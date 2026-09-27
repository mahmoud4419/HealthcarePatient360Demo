#!/usr/bin/env python3
"""
object_detection_model.py — SageMaker-Compatible Medical Anomaly Detection
===========================================================================
Healthcare & Life Sciences Demo — Patient 360 Lakehouse Integration

Detects and localizes anomalies in medical images using Faster R-CNN with a
ResNet50-FPN backbone. Multi-class detection for:
  0: background (implicit)
  1: tumor
  2: fracture
  3: calcification
  4: nodule
  5: effusion

Outputs bounding boxes, class labels, and confidence scores in COCO-compatible
format for downstream integration with the Patient 360 lakehouse.

Usage (SageMaker):
    estimator = PyTorch(
        entry_point="object_detection_model.py",
        role=role,
        instance_count=1,
        instance_type="ml.p3.2xlarge",
        framework_version="2.1",
        hyperparameters={
            "epochs": 25,
            "batch-size": 4,
            "learning-rate": 5e-5,
            "confidence-threshold": 0.5,
            "nms-threshold": 0.4,
        },
    )
    estimator.fit({"training": s3_data_uri})
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision.models.detection import (
    FasterRCNN,
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MedicalObjectDetection")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ANOMALY_CLASSES: Dict[int, str] = {
    0: "__background__",
    1: "tumor",
    2: "fracture",
    3: "calcification",
    4: "nodule",
    5: "effusion",
}
NUM_CLASSES = len(ANOMALY_CLASSES)  # includes background
IMAGE_SIZE = 512


# ═══════════════════════════════════════════════════════════════════════════
# 1. COCO-FORMAT ANNOTATION HANDLER
# ═══════════════════════════════════════════════════════════════════════════
class COCOAnnotationParser:
    """Parse and validate COCO-format annotations for medical imaging.

    Expected COCO JSON structure:
    {
        "images": [{"id": 1, "file_name": "scan_001.png", "width": 512, "height": 512}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1,
             "bbox": [x, y, width, height], "area": 1024, "iscrowd": 0}
        ],
        "categories": [
            {"id": 1, "name": "tumor"}, ...
        ]
    }
    """

    def __init__(self, annotation_path: str) -> None:
        with open(annotation_path, "r") as f:
            self.coco = json.load(f)

        self.images = {img["id"]: img for img in self.coco["images"]}
        self.categories = {cat["id"]: cat["name"] for cat in self.coco["categories"]}

        # Group annotations by image_id
        self.img_to_anns: Dict[int, List[Dict]] = defaultdict(list)
        for ann in self.coco["annotations"]:
            self.img_to_anns[ann["image_id"]].append(ann)

        logger.info(
            "Loaded COCO annotations: %d images, %d annotations, %d categories",
            len(self.images),
            len(self.coco["annotations"]),
            len(self.categories),
        )

    def get_image_annotations(
        self, image_id: int
    ) -> Tuple[List[List[float]], List[int]]:
        """Return (boxes_xyxy, labels) for an image."""
        anns = self.img_to_anns.get(image_id, [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            # Convert COCO [x, y, w, h] → [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
        return boxes, labels


# ═══════════════════════════════════════════════════════════════════════════
# 2. DATASET
# ═══════════════════════════════════════════════════════════════════════════
class MedicalDetectionDataset(Dataset):
    """Dataset for medical anomaly detection with COCO annotations.

    Falls back to synthetic annotations when real data is unavailable.
    """

    def __init__(
        self,
        image_dir: str,
        annotation_path: Optional[str] = None,
        split: str = "train",
        image_size: int = IMAGE_SIZE,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.split = split

        if annotation_path and Path(annotation_path).exists():
            self.parser = COCOAnnotationParser(annotation_path)
            self.image_ids = list(self.parser.images.keys())
        else:
            logger.warning("No annotations found — using synthetic data for demo")
            self.parser = None
            self.image_ids = list(range(100 if split == "train" else 20))

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_id = self.image_ids[idx]

        # Load or synthesize image
        if self.parser:
            img_info = self.parser.images[image_id]
            img_path = self.image_dir / img_info["file_name"]
            if img_path.exists():
                image = Image.open(img_path).convert("RGB")
            else:
                image = self._synthetic_image(image_id)
            boxes, labels = self.parser.get_image_annotations(image_id)
        else:
            image = self._synthetic_image(image_id)
            boxes, labels = self._synthetic_annotations(image_id)

        # Resize
        image = image.resize((self.image_size, self.image_size))
        image_tensor = F.to_tensor(image)

        # Build target dict (Faster R-CNN expected format)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        # Data augmentation for training
        if self.split == "train":
            image_tensor, target = self._augment(image_tensor, target)

        return image_tensor, target

    def _synthetic_image(self, seed: int) -> Image.Image:
        rng = np.random.RandomState(seed % 2**31)
        arr = rng.randint(30, 200, (self.image_size, self.image_size, 3), dtype=np.uint8)
        return Image.fromarray(arr, "RGB")

    def _synthetic_annotations(
        self, seed: int
    ) -> Tuple[List[List[float]], List[int]]:
        rng = np.random.RandomState(seed % 2**31)
        n_objects = rng.randint(0, 4)
        boxes, labels = [], []
        for _ in range(n_objects):
            x1 = rng.randint(10, self.image_size - 100)
            y1 = rng.randint(10, self.image_size - 100)
            w = rng.randint(30, 120)
            h = rng.randint(30, 120)
            boxes.append([x1, y1, x1 + w, y1 + h])
            labels.append(rng.randint(1, NUM_CLASSES))
        return boxes, labels

    def _augment(
        self, image: torch.Tensor, target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Simple augmentation: horizontal flip with 50% probability."""
        if torch.rand(1).item() > 0.5:
            image = F.hflip(image)
            if target["boxes"].numel() > 0:
                boxes = target["boxes"]
                w = image.shape[-1]
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict]]
) -> Tuple[List[torch.Tensor], List[Dict]]:
    """Custom collate for variable-size targets."""
    return tuple(zip(*batch))


# ═══════════════════════════════════════════════════════════════════════════
# 3. MODEL — FASTER R-CNN WITH CUSTOM HEAD
# ═══════════════════════════════════════════════════════════════════════════
def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> FasterRCNN:
    """Build Faster R-CNN with ResNet50-FPN backbone, customized for medical imaging.

    The final classification head is replaced to predict our anomaly classes.
    """
    if pretrained:
        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )
    else:
        model = fasterrcnn_resnet50_fpn_v2(weights=None)

    # Replace the classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Faster R-CNN built: %s trainable parameters", f"{trainable:,}")

    return model


# ═══════════════════════════════════════════════════════════════════════════
# 4. MEAN AVERAGE PRECISION (mAP) EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
class MAPEvaluator:
    """Compute COCO-style mean Average Precision for object detection.

    Implements IoU-based matching at thresholds [0.5, 0.55, ..., 0.95]
    (COCO mAP convention), plus mAP@0.5 and mAP@0.75.
    """

    def __init__(self, iou_thresholds: Optional[List[float]] = None) -> None:
        self.iou_thresholds = iou_thresholds or [
            0.5 + 0.05 * i for i in range(10)
        ]
        self.all_predictions: List[Dict] = []
        self.all_ground_truths: List[Dict] = []

    def update(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        ground_truths: List[Dict[str, torch.Tensor]],
    ) -> None:
        for pred, gt in zip(predictions, ground_truths):
            self.all_predictions.append(
                {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in pred.items()}
            )
            self.all_ground_truths.append(
                {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in gt.items()}
            )

    @staticmethod
    def _compute_iou(
        box1: np.ndarray, box2: np.ndarray
    ) -> np.ndarray:
        """Compute IoU between two sets of boxes [N, 4] and [M, 4]."""
        x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
        y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
        x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
        y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union = area1[:, None] + area2[None, :] - inter
        return inter / (union + 1e-6)

    def compute_ap_per_class(
        self, class_id: int, iou_threshold: float = 0.5
    ) -> float:
        """Compute Average Precision for a single class at a given IoU threshold."""
        all_scores, all_tp = [], []
        n_gt_total = 0

        for pred, gt in zip(self.all_predictions, self.all_ground_truths):
            pred_mask = pred["labels"] == class_id
            gt_mask = gt["labels"] == class_id

            pred_boxes = pred["boxes"][pred_mask]
            pred_scores = pred["scores"][pred_mask]
            gt_boxes = gt["boxes"][gt_mask]

            n_gt_total += len(gt_boxes)

            if len(pred_boxes) == 0:
                continue
            if len(gt_boxes) == 0:
                all_scores.extend(pred_scores.tolist())
                all_tp.extend([0] * len(pred_scores))
                continue

            ious = self._compute_iou(pred_boxes, gt_boxes)
            gt_matched = set()

            # Sort predictions by score descending
            sorted_idx = np.argsort(-pred_scores)
            for idx in sorted_idx:
                best_iou_idx = np.argmax(ious[idx])
                best_iou = ious[idx, best_iou_idx]

                all_scores.append(pred_scores[idx])
                if best_iou >= iou_threshold and best_iou_idx not in gt_matched:
                    all_tp.append(1)
                    gt_matched.add(best_iou_idx)
                else:
                    all_tp.append(0)

        if n_gt_total == 0:
            return 0.0

        # Sort by score and compute precision-recall curve
        sorted_idx = np.argsort(-np.array(all_scores))
        tp_cumsum = np.cumsum(np.array(all_tp)[sorted_idx])
        fp_cumsum = np.arange(1, len(sorted_idx) + 1) - tp_cumsum

        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall = tp_cumsum / (n_gt_total + 1e-6)

        # 11-point interpolation
        ap = 0.0
        for r_threshold in np.linspace(0, 1, 11):
            precisions_above = precision[recall >= r_threshold]
            ap += (precisions_above.max() if len(precisions_above) > 0 else 0.0)
        return ap / 11.0

    def compute(self) -> Dict[str, float]:
        """Compute mAP across all classes and IoU thresholds."""
        results: Dict[str, float] = {}

        # mAP@0.5
        aps_50 = []
        for cls_id in range(1, NUM_CLASSES):
            ap = self.compute_ap_per_class(cls_id, iou_threshold=0.5)
            aps_50.append(ap)
            results[f"AP@0.5_{ANOMALY_CLASSES[cls_id]}"] = ap
        results["mAP@0.5"] = np.mean(aps_50) if aps_50 else 0.0

        # mAP@0.75
        aps_75 = []
        for cls_id in range(1, NUM_CLASSES):
            ap = self.compute_ap_per_class(cls_id, iou_threshold=0.75)
            aps_75.append(ap)
        results["mAP@0.75"] = np.mean(aps_75) if aps_75 else 0.0

        # COCO mAP (average over IoU thresholds 0.5:0.95)
        aps_coco = []
        for cls_id in range(1, NUM_CLASSES):
            class_aps = [
                self.compute_ap_per_class(cls_id, iou_threshold=t)
                for t in self.iou_thresholds
            ]
            aps_coco.append(np.mean(class_aps))
        results["mAP@[0.5:0.95]"] = np.mean(aps_coco) if aps_coco else 0.0

        return results

    def reset(self) -> None:
        self.all_predictions.clear()
        self.all_ground_truths.clear()


# ═══════════════════════════════════════════════════════════════════════════
# 5. CONFIDENCE THRESHOLDING + NON-MAXIMUM SUPPRESSION
# ═══════════════════════════════════════════════════════════════════════════
def apply_nms_and_threshold(
    predictions: Dict[str, torch.Tensor],
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.4,
) -> Dict[str, torch.Tensor]:
    """Post-process detections: filter low-confidence, apply NMS."""
    boxes = predictions["boxes"]
    scores = predictions["scores"]
    labels = predictions["labels"]

    # Confidence filter
    keep_conf = scores >= confidence_threshold
    boxes = boxes[keep_conf]
    scores = scores[keep_conf]
    labels = labels[keep_conf]

    if len(boxes) == 0:
        return {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros((0,)),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

    # Per-class NMS
    keep_indices = torchvision.ops.batched_nms(boxes, scores, labels, nms_threshold)

    return {
        "boxes": boxes[keep_indices],
        "scores": scores[keep_indices],
        "labels": labels[keep_indices],
    }


def format_predictions_coco(
    predictions: Dict[str, torch.Tensor],
    image_id: int,
) -> List[Dict[str, Any]]:
    """Format detections as COCO-style result records for downstream integration."""
    results = []
    boxes = predictions["boxes"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        results.append(
            {
                "image_id": int(image_id),
                "category_id": int(labels[i]),
                "category_name": ANOMALY_CLASSES.get(int(labels[i]), "unknown"),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(scores[i]),
            }
        )
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 6. TRAINING ENGINE
# ═══════════════════════════════════════════════════════════════════════════
class DetectionTrainer:
    """Training loop for Faster R-CNN with mAP evaluation."""

    def __init__(
        self,
        model: FasterRCNN,
        device: torch.device,
        learning_rate: float = 5e-5,
        weight_decay: float = 1e-4,
        patience: int = 5,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.patience = patience

        # Separate learning rates for backbone vs head
        params = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": learning_rate * 0.1,  # lower LR for backbone
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ],
                "lr": learning_rate,
            },
        ]
        self.optimizer = optim.SGD(
            params, momentum=0.9, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=8, gamma=0.5
        )

        self.best_map = 0.0
        self.best_model_state: Optional[Dict[str, Any]] = None
        self.epochs_no_improve = 0

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_idx, (images, targets) in enumerate(loader):
            images = [img.to(self.device) for img in images]
            targets = [
                {k: v.to(self.device) for k, v in t.items()} for t in targets
            ]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += losses.item()

            if (batch_idx + 1) % 20 == 0:
                loss_str = " | ".join(
                    f"{k}: {v.item():.4f}" for k, v in loss_dict.items()
                )
                logger.info("  Batch %d/%d — %s", batch_idx + 1, len(loader), loss_str)

        self.scheduler.step()
        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        evaluator = MAPEvaluator()

        for images, targets in loader:
            images = [img.to(self.device) for img in images]
            predictions = self.model(images)

            # Post-process each prediction
            processed = []
            for pred in predictions:
                pred = apply_nms_and_threshold(
                    pred, self.confidence_threshold, self.nms_threshold
                )
                processed.append(pred)

            evaluator.update(processed, targets)

        return evaluator.compute()

    def check_early_stopping(self, current_map: float) -> bool:
        if current_map > self.best_map:
            self.best_map = current_map
            self.best_model_state = {
                k: v.cpu().clone() for k, v in self.model.state_dict().items()
            }
            self.epochs_no_improve = 0
            return False
        else:
            self.epochs_no_improve += 1
            return self.epochs_no_improve >= self.patience

    def restore_best_model(self) -> None:
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Restored best model (mAP@0.5=%.4f)", self.best_map)


# ═══════════════════════════════════════════════════════════════════════════
# 7. MODEL EXPORT
# ═══════════════════════════════════════════════════════════════════════════
def save_detection_model(
    model: FasterRCNN,
    output_dir: str,
    metrics: Dict[str, float],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save state dict (Faster R-CNN doesn't trace cleanly to TorchScript)
    model_path = output_path / "detection_model.pth"
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "num_classes": NUM_CLASSES,
            "anomaly_classes": ANOMALY_CLASSES,
            "image_size": IMAGE_SIZE,
        },
        str(model_path),
    )
    logger.info("Saved detection model to %s", model_path)

    # Save metrics
    metrics_path = output_path / "detection_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# 8. SYNTHETIC COCO DATA GENERATOR (demo)
# ═══════════════════════════════════════════════════════════════════════════
def generate_demo_coco(data_dir: str, n_images: int = 100) -> str:
    """Generate synthetic COCO annotations for demo."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    images_list, annotations_list = [], []
    ann_id = 1

    for img_id in range(1, n_images + 1):
        images_list.append(
            {
                "id": img_id,
                "file_name": f"scan_{img_id:06d}.png",
                "width": IMAGE_SIZE,
                "height": IMAGE_SIZE,
            }
        )
        n_objects = rng.randint(0, 5)
        for _ in range(n_objects):
            x = rng.randint(10, IMAGE_SIZE - 100)
            y = rng.randint(10, IMAGE_SIZE - 100)
            w = rng.randint(20, 100)
            h = rng.randint(20, 100)
            annotations_list.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": rng.randint(1, NUM_CLASSES),
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "area": int(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [
        {"id": k, "name": v}
        for k, v in ANOMALY_CLASSES.items()
        if k > 0
    ]

    coco_data = {
        "images": images_list,
        "annotations": annotations_list,
        "categories": categories,
    }

    ann_path = data_path / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    logger.info("Generated demo COCO annotations: %s", ann_path)
    return str(ann_path)


# ═══════════════════════════════════════════════════════════════════════════
# 9. MAIN — SAGEMAKER ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Medical Anomaly Detector (SageMaker compatible)"
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.4)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAINING", "./sample_data"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "./model_output"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("=" * 70)
    logger.info("Medical Anomaly Detection — Training Started")
    logger.info("=" * 70)
    logger.info("Hyperparameters: %s", vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Data ──────────────────────────────────────────────────────────
    ann_path = Path(args.data_dir) / "annotations.json"
    if not ann_path.exists():
        ann_path = Path(generate_demo_coco(args.data_dir))

    image_dir = Path(args.data_dir) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    train_ds = MedicalDetectionDataset(str(image_dir), str(ann_path), split="train")
    val_ds = MedicalDetectionDataset(str(image_dir), str(ann_path), split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(num_classes=NUM_CLASSES, pretrained=True)

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = DetectionTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold,
    )

    # ── Training Loop ─────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = trainer.train_epoch(train_loader)
        eval_metrics = trainer.evaluate(val_loader)
        elapsed = time.time() - t0

        logger.info(
            "Epoch %d/%d (%.1fs) — train_loss=%.4f | mAP@0.5=%.4f mAP@[.5:.95]=%.4f",
            epoch,
            args.epochs,
            elapsed,
            train_loss,
            eval_metrics.get("mAP@0.5", 0),
            eval_metrics.get("mAP@[0.5:0.95]", 0),
        )

        if trainer.check_early_stopping(eval_metrics.get("mAP@0.5", 0)):
            logger.info("Early stopping at epoch %d", epoch)
            break

    # ── Final ─────────────────────────────────────────────────────────
    trainer.restore_best_model()
    final_metrics = trainer.evaluate(val_loader)

    logger.info("=" * 70)
    logger.info("Final Detection Metrics:")
    for k, v in sorted(final_metrics.items()):
        logger.info("  %s: %.4f", k, v)

    save_detection_model(model, args.output_dir, final_metrics)
    logger.info("Training complete. Model saved to %s", args.output_dir)


# ═══════════════════════════════════════════════════════════════════════════
# SageMaker hosting functions
# ═══════════════════════════════════════════════════════════════════════════
def model_fn(model_dir: str) -> FasterRCNN:
    checkpoint = torch.load(
        os.path.join(model_dir, "detection_model.pth"),
        map_location="cpu",
    )
    model = build_model(
        num_classes=checkpoint["num_classes"], pretrained=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    main()
