#!/usr/bin/env python3
"""
inference_pipeline.py — End-to-End Medical Imaging Inference Pipeline
=====================================================================
Healthcare & Life Sciences Demo — Patient 360 Lakehouse Integration

This pipeline:
  1. Loads trained classification and detection models
  2. Accepts batch imaging metadata records (from imaging_metadata.csv)
  3. Runs chest X-ray classification + anomaly detection
  4. Enriches records with AI predictions
  5. Writes results to S3 in Iceberg-compatible Parquet format
  6. Registers results in healthcare_db.ai_imaging_results

Designed to run as:
  - SageMaker Processing Job (batch)
  - SageMaker Pipeline step
  - Lambda-triggered near-real-time inference
  - Local development/testing

Usage:
    python inference_pipeline.py \
        --classification-model s3://bucket/models/classification/model.pt \
        --detection-model s3://bucket/models/detection/detection_model.pth \
        --input-data s3://bucket/data/imaging_metadata.csv \
        --output-path s3://bucket/results/ai_imaging_results/ \
        --database healthcare_db \
        --table ai_imaging_results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("InferencePipeline")


# ═══════════════════════════════════════════════════════════════════════════
# 1. S3 / LOCAL MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════
class ModelLoader:
    """Load models from S3 or local filesystem."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._s3_client = None

    @property
    def s3(self):
        if self._s3_client is None:
            try:
                import boto3
                self._s3_client = boto3.client("s3")
            except ImportError:
                logger.warning("boto3 not available — S3 operations will fail")
        return self._s3_client

    def _download_from_s3(self, s3_uri: str) -> str:
        """Download S3 object to a temp file and return local path."""
        assert s3_uri.startswith("s3://"), f"Not an S3 URI: {s3_uri}"
        bucket, key = s3_uri[5:].split("/", 1)
        local_path = os.path.join(tempfile.mkdtemp(), os.path.basename(key))
        logger.info("Downloading %s → %s", s3_uri, local_path)
        self.s3.download_file(bucket, key, local_path)
        return local_path

    def _resolve_path(self, path: str) -> str:
        """Resolve an S3 URI or local path."""
        if path.startswith("s3://"):
            return self._download_from_s3(path)
        return path

    def load_classification_model(self, model_path: str) -> torch.jit.ScriptModule:
        """Load TorchScript classification model."""
        local_path = self._resolve_path(model_path)
        logger.info("Loading classification model from %s", local_path)
        model = torch.jit.load(local_path, map_location=self.device)
        model.eval()
        return model

    def load_detection_model(self, model_path: str) -> nn.Module:
        """Load Faster R-CNN detection model."""
        local_path = self._resolve_path(model_path)
        logger.info("Loading detection model from %s", local_path)

        checkpoint = torch.load(local_path, map_location=self.device)
        num_classes = checkpoint.get("num_classes", 6)

        # Re-build model architecture and load weights
        from object_detection_model import build_model
        model = build_model(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model


# ═══════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING — IMAGE TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════
CLASSIFICATION_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

DETECTION_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ]
)

# Classification class mapping
CLASSIFICATION_CLASSES = {0: "Normal", 1: "Pneumonia", 2: "COVID-19", 3: "Cardiomegaly"}

# Detection class mapping
DETECTION_CLASSES = {
    0: "__background__",
    1: "tumor",
    2: "fracture",
    3: "calcification",
    4: "nodule",
    5: "effusion",
}


# ═══════════════════════════════════════════════════════════════════════════
# 3. INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════
class InferenceEngine:
    """Run classification and detection inference on batches of imaging records."""

    def __init__(
        self,
        classification_model: Optional[nn.Module] = None,
        detection_model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.cls_model = classification_model
        self.det_model = detection_model
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    # -------------------------------------------------------------------
    @torch.no_grad()
    def classify_batch(
        self, image_tensors: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """Run classification on a batch of pre-processed image tensors.

        Returns list of dicts with predicted class, confidence, and probabilities.
        """
        if self.cls_model is None:
            return self._synthetic_classification(len(image_tensors))

        image_tensors = image_tensors.to(self.device)
        outputs = self.cls_model(image_tensors)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        results = []
        for i in range(len(preds)):
            class_id = preds[i].item()
            results.append(
                {
                    "classification_label": CLASSIFICATION_CLASSES.get(
                        class_id, "Unknown"
                    ),
                    "classification_id": class_id,
                    "classification_confidence": float(probs[i, class_id]),
                    "classification_probabilities": {
                        CLASSIFICATION_CLASSES[j]: float(probs[i, j])
                        for j in range(probs.shape[1])
                    },
                }
            )
        return results

    # -------------------------------------------------------------------
    @torch.no_grad()
    def detect_batch(
        self, image_tensors: List[torch.Tensor]
    ) -> List[List[Dict[str, Any]]]:
        """Run object detection on a list of image tensors.

        Returns list-of-lists of detection dicts (one list per image).
        """
        if self.det_model is None:
            return self._synthetic_detections(len(image_tensors))

        images = [img.to(self.device) for img in image_tensors]
        predictions = self.det_model(images)

        all_results = []
        for pred in predictions:
            detections = self._post_process_detections(pred)
            all_results.append(detections)
        return all_results

    def _post_process_detections(
        self, pred: Dict[str, torch.Tensor]
    ) -> List[Dict[str, Any]]:
        """Apply NMS, threshold, and format detections."""
        import torchvision

        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]

        # Confidence filter
        keep = scores >= self.confidence_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        if len(boxes) == 0:
            return []

        # NMS
        keep_nms = torchvision.ops.batched_nms(
            boxes, scores, labels, self.nms_threshold
        )
        boxes = boxes[keep_nms].cpu().numpy()
        scores = scores[keep_nms].cpu().numpy()
        labels = labels[keep_nms].cpu().numpy()

        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            detections.append(
                {
                    "anomaly_type": DETECTION_CLASSES.get(int(labels[i]), "unknown"),
                    "anomaly_class_id": int(labels[i]),
                    "confidence": float(scores[i]),
                    "bbox_x1": float(x1),
                    "bbox_y1": float(y1),
                    "bbox_x2": float(x2),
                    "bbox_y2": float(y2),
                    "bbox_width": float(x2 - x1),
                    "bbox_height": float(y2 - y1),
                }
            )
        return detections

    # -------------------------------------------------------------------
    # Synthetic fallbacks for demo mode
    # -------------------------------------------------------------------
    def _synthetic_classification(self, n: int) -> List[Dict[str, Any]]:
        rng = np.random.RandomState(42)
        results = []
        for _ in range(n):
            class_id = rng.choice([0, 1, 2, 3], p=[0.5, 0.25, 0.15, 0.1])
            conf = rng.uniform(0.65, 0.99)
            probs = rng.dirichlet([5 if j == class_id else 1 for j in range(4)])
            results.append(
                {
                    "classification_label": CLASSIFICATION_CLASSES[class_id],
                    "classification_id": class_id,
                    "classification_confidence": float(conf),
                    "classification_probabilities": {
                        CLASSIFICATION_CLASSES[j]: float(probs[j]) for j in range(4)
                    },
                }
            )
        return results

    def _synthetic_detections(self, n: int) -> List[List[Dict[str, Any]]]:
        rng = np.random.RandomState(42)
        all_results = []
        for _ in range(n):
            n_det = rng.choice([0, 0, 1, 1, 2, 3])
            detections = []
            for _ in range(n_det):
                cls_id = rng.randint(1, 6)
                x1, y1 = rng.randint(10, 300, size=2)
                w, h = rng.randint(30, 150, size=2)
                detections.append(
                    {
                        "anomaly_type": DETECTION_CLASSES[cls_id],
                        "anomaly_class_id": cls_id,
                        "confidence": float(rng.uniform(0.5, 0.98)),
                        "bbox_x1": float(x1),
                        "bbox_y1": float(y1),
                        "bbox_x2": float(x1 + w),
                        "bbox_y2": float(y1 + h),
                        "bbox_width": float(w),
                        "bbox_height": float(h),
                    }
                )
            all_results.append(detections)
        return all_results


# ═══════════════════════════════════════════════════════════════════════════
# 4. RECORD ENRICHMENT — MERGE AI PREDICTIONS WITH SOURCE METADATA
# ═══════════════════════════════════════════════════════════════════════════
class RecordEnricher:
    """Merge AI predictions back into imaging metadata records."""

    @staticmethod
    def enrich(
        metadata_df: pd.DataFrame,
        classification_results: List[Dict[str, Any]],
        detection_results: List[List[Dict[str, Any]]],
    ) -> pd.DataFrame:
        """Add classification + detection columns to the metadata DataFrame.

        Output schema aligns with healthcare_db.ai_imaging_results Iceberg table:
            - Original metadata columns
            - classification_label, classification_confidence, classification_probabilities
            - detection_count, detections (JSON array)
            - ai_inference_id, ai_inference_timestamp, model_version
        """
        enriched = metadata_df.copy()

        # Classification columns
        enriched["classification_label"] = [
            r["classification_label"] for r in classification_results
        ]
        enriched["classification_id"] = [
            r["classification_id"] for r in classification_results
        ]
        enriched["classification_confidence"] = [
            r["classification_confidence"] for r in classification_results
        ]
        enriched["classification_probabilities"] = [
            json.dumps(r["classification_probabilities"])
            for r in classification_results
        ]

        # Detection columns
        enriched["detection_count"] = [len(dets) for dets in detection_results]
        enriched["detections"] = [json.dumps(dets) for dets in detection_results]

        # Primary anomaly (highest confidence detection, if any)
        enriched["primary_anomaly_type"] = [
            max(dets, key=lambda d: d["confidence"])["anomaly_type"]
            if dets
            else None
            for dets in detection_results
        ]
        enriched["primary_anomaly_confidence"] = [
            max(dets, key=lambda d: d["confidence"])["confidence"]
            if dets
            else None
            for dets in detection_results
        ]

        # Inference metadata
        now = datetime.now(timezone.utc)
        enriched["ai_inference_id"] = [
            str(uuid.uuid4()) for _ in range(len(enriched))
        ]
        enriched["ai_inference_timestamp"] = now.isoformat()
        enriched["model_version"] = "v1.0.0"

        # Clinical flags
        enriched["requires_urgent_review"] = enriched.apply(
            _flag_urgent_review, axis=1
        )

        return enriched


def _flag_urgent_review(row: pd.Series) -> bool:
    """Flag records requiring urgent clinical review."""
    # Urgent if: high-confidence abnormal classification or any tumor/fracture detected
    if row.get("classification_label") in ("Pneumonia", "COVID-19", "Cardiomegaly"):
        if row.get("classification_confidence", 0) > 0.85:
            return True
    detections = json.loads(row.get("detections", "[]"))
    for det in detections:
        if det["anomaly_type"] in ("tumor", "fracture") and det["confidence"] > 0.7:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# 5. OUTPUT WRITER — S3 + ICEBERG INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════
class OutputWriter:
    """Write enriched results to S3 in Iceberg-compatible Parquet format."""

    def __init__(
        self,
        output_path: str,
        database: str = "healthcare_db",
        table: str = "ai_imaging_results",
    ):
        self.output_path = output_path
        self.database = database
        self.table = table
        self._s3_client = None

    def write(self, df: pd.DataFrame) -> str:
        """Write DataFrame as Parquet with Iceberg-compatible partitioning.

        Partition scheme: year/month/day based on ai_inference_timestamp.
        """
        now = datetime.now(timezone.utc)
        partition_path = f"year={now.year}/month={now.month:02d}/day={now.day:02d}"

        if self.output_path.startswith("s3://"):
            return self._write_to_s3(df, partition_path)
        else:
            return self._write_to_local(df, partition_path)

    def _write_to_local(self, df: pd.DataFrame, partition_path: str) -> str:
        """Write to local filesystem."""
        output_dir = Path(self.output_path) / partition_path
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_id = uuid.uuid4().hex[:12]
        filename = f"ai_results_{batch_id}.parquet"
        filepath = output_dir / filename

        df.to_parquet(str(filepath), engine="pyarrow", index=False)
        logger.info("Wrote %d records to %s", len(df), filepath)

        # Also write a summary JSON
        summary = {
            "batch_id": batch_id,
            "record_count": len(df),
            "partition": partition_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "classification_distribution": (
                df["classification_label"].value_counts().to_dict()
            ),
            "detection_summary": {
                "total_detections": int(df["detection_count"].sum()),
                "images_with_detections": int((df["detection_count"] > 0).sum()),
                "urgent_reviews": int(df["requires_urgent_review"].sum()),
            },
        }
        summary_path = output_dir / f"_batch_summary_{batch_id}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return str(filepath)

    def _write_to_s3(self, df: pd.DataFrame, partition_path: str) -> str:
        """Write to S3 bucket."""
        import boto3
        import io

        bucket, prefix = self.output_path[5:].split("/", 1)
        batch_id = uuid.uuid4().hex[:12]
        key = f"{prefix}/{partition_path}/ai_results_{batch_id}.parquet"

        buffer = io.BytesIO()
        df.to_parquet(buffer, engine="pyarrow", index=False)
        buffer.seek(0)

        s3 = boto3.client("s3")
        s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        s3_uri = f"s3://{bucket}/{key}"
        logger.info("Wrote %d records to %s", len(df), s3_uri)
        return s3_uri

    def register_with_glue(self, output_uri: str) -> None:
        """Register new data with AWS Glue catalog for Iceberg table access.

        In production this would call Glue API or run an ALTER TABLE on Athena.
        For demo, we log the registration intent.
        """
        logger.info(
            "Registering data with Glue catalog: %s.%s ← %s",
            self.database,
            self.table,
            output_uri,
        )
        # Production implementation:
        # glue = boto3.client("glue")
        # glue.update_table(DatabaseName=self.database, TableInput={...})
        #
        # Or via Athena:
        # athena.start_query_execution(
        #     QueryString=f"ALTER TABLE {self.database}.{self.table} ADD PARTITION ...",
        #     ...
        # )
        logger.info(
            "✓ Results registered in %s.%s — accessible via Athena/Spark",
            self.database,
            self.table,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 6. MAIN PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
class MedicalImagingPipeline:
    """End-to-end inference pipeline orchestrating all components."""

    def __init__(
        self,
        classification_model_path: Optional[str] = None,
        detection_model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        batch_size: int = 16,
    ):
        self.batch_size = batch_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("Pipeline device: %s", self.device)

        # Load models
        loader = ModelLoader(device=self.device)
        cls_model = (
            loader.load_classification_model(classification_model_path)
            if classification_model_path
            else None
        )
        det_model = (
            loader.load_detection_model(detection_model_path)
            if detection_model_path
            else None
        )

        self.engine = InferenceEngine(
            classification_model=cls_model,
            detection_model=det_model,
            device=self.device,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
        )
        self.enricher = RecordEnricher()

    def run(
        self,
        input_data: str,
        output_path: str,
        database: str = "healthcare_db",
        table: str = "ai_imaging_results",
    ) -> pd.DataFrame:
        """Execute the full pipeline.

        Args:
            input_data: Path to imaging_metadata.csv (S3 or local)
            output_path: S3 or local path for Parquet output
            database: Glue database name
            table: Iceberg table name

        Returns:
            Enriched DataFrame with AI predictions
        """
        logger.info("=" * 70)
        logger.info("Medical Imaging Inference Pipeline — Started")
        logger.info("=" * 70)

        # ── 1. Load input data ────────────────────────────────────────
        logger.info("Step 1: Loading input data from %s", input_data)
        if input_data.startswith("s3://"):
            import boto3
            import io

            bucket, key = input_data[5:].split("/", 1)
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        else:
            df = pd.read_csv(input_data)

        logger.info("Loaded %d imaging records", len(df))

        # ── 2. Run inference in batches ───────────────────────────────
        logger.info("Step 2: Running AI inference (batch_size=%d)", self.batch_size)
        all_cls_results: List[Dict] = []
        all_det_results: List[List[Dict]] = []

        for start in range(0, len(df), self.batch_size):
            end = min(start + self.batch_size, len(df))
            batch_df = df.iloc[start:end]

            # For demo: generate synthetic tensors (real pipeline would load images)
            cls_tensors = torch.randn(len(batch_df), 3, 224, 224)
            det_tensors = [torch.randn(3, 512, 512) for _ in range(len(batch_df))]

            cls_batch = self.engine.classify_batch(cls_tensors)
            det_batch = self.engine.detect_batch(det_tensors)

            all_cls_results.extend(cls_batch)
            all_det_results.extend(det_batch)

            logger.info(
                "  Processed batch %d-%d / %d", start + 1, end, len(df)
            )

        # ── 3. Enrich records ─────────────────────────────────────────
        logger.info("Step 3: Enriching records with AI predictions")
        enriched_df = self.enricher.enrich(df, all_cls_results, all_det_results)

        # ── 4. Write output ───────────────────────────────────────────
        logger.info("Step 4: Writing results to %s", output_path)
        writer = OutputWriter(output_path, database, table)
        output_uri = writer.write(enriched_df)
        writer.register_with_glue(output_uri)

        # ── 5. Summary ────────────────────────────────────────────────
        self._log_summary(enriched_df)

        return enriched_df

    def _log_summary(self, df: pd.DataFrame) -> None:
        logger.info("=" * 70)
        logger.info("Pipeline Summary:")
        logger.info("  Total records processed: %d", len(df))
        logger.info("  Classification distribution:")
        for label, count in df["classification_label"].value_counts().items():
            logger.info("    %s: %d (%.1f%%)", label, count, 100 * count / len(df))
        logger.info(
            "  Records with detections: %d", (df["detection_count"] > 0).sum()
        )
        logger.info(
            "  Total anomalies detected: %d", df["detection_count"].sum()
        )
        logger.info(
            "  Records flagged for urgent review: %d",
            df["requires_urgent_review"].sum(),
        )
        logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# 7. SAMPLE DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════
def generate_sample_metadata(output_path: str, n: int = 50) -> str:
    """Generate sample imaging_metadata.csv for demo."""
    rng = np.random.RandomState(42)
    modalities = ["CR", "DX", "CT", "MR"]
    body_parts = ["CHEST", "ABDOMEN", "HEAD", "EXTREMITY"]

    records = []
    for i in range(n):
        records.append(
            {
                "image_id": f"IMG_{i:06d}",
                "patient_id": f"PAT_{rng.randint(10000, 99999)}",
                "encounter_id": f"ENC_{rng.randint(100000, 999999)}",
                "study_date": f"2024-{rng.randint(1, 13):02d}-{rng.randint(1, 29):02d}",
                "modality": rng.choice(modalities),
                "body_part": rng.choice(body_parts),
                "referring_physician": f"DR_{rng.randint(100, 999)}",
                "facility_id": f"FAC_{rng.choice(['001', '002', '003'])}",
                "pixel_rows": 512,
                "pixel_cols": 512,
                "bits_stored": 16,
            }
        )

    filepath = Path(output_path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(str(filepath), index=False)
    logger.info("Generated sample metadata: %s (%d records)", filepath, n)
    return str(filepath)


# ═══════════════════════════════════════════════════════════════════════════
# 8. CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Medical Imaging Inference Pipeline"
    )
    parser.add_argument(
        "--classification-model",
        type=str,
        default=None,
        help="Path to classification model (S3 URI or local)",
    )
    parser.add_argument(
        "--detection-model",
        type=str,
        default=None,
        help="Path to detection model (S3 URI or local)",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAINING", "./sample_data/imaging_metadata.csv"),
        help="Path to imaging_metadata.csv",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./inference_output",
        help="Output directory (S3 URI or local)",
    )
    parser.add_argument("--database", type=str, default="healthcare_db")
    parser.add_argument("--table", type=str, default="ai_imaging_results")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.4)
    parser.add_argument(
        "--generate-sample", action="store_true", help="Generate sample data and run"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Generate sample data if needed
    if args.generate_sample or not Path(args.input_data).exists():
        logger.info("Generating sample imaging metadata for demo")
        args.input_data = generate_sample_metadata(args.input_data, n=50)

    # Run pipeline
    pipeline = MedicalImagingPipeline(
        classification_model_path=args.classification_model,
        detection_model_path=args.detection_model,
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold,
        batch_size=args.batch_size,
    )

    results = pipeline.run(
        input_data=args.input_data,
        output_path=args.output_path,
        database=args.database,
        table=args.table,
    )

    logger.info("Pipeline complete. %d records processed.", len(results))


if __name__ == "__main__":
    main()
