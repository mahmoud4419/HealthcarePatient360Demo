#!/usr/bin/env python3
"""
image_classification_model.py — SageMaker-Compatible Chest X-Ray Classification
================================================================================
Healthcare & Life Sciences Demo — Patient 360 Lakehouse Integration

Classifies chest X-ray images into four categories:
  0: Normal
  1: Pneumonia
  2: COVID-19
  3: Cardiomegaly

Architecture: ResNet50 (ImageNet pre-trained) with frozen early layers, fine-tuned
classification head, and medical-imaging-specific data augmentation.

Usage (SageMaker):
    estimator = PyTorch(
        entry_point="image_classification_model.py",
        role=role,
        instance_count=1,
        instance_type="ml.p3.2xlarge",
        framework_version="2.1",
        hyperparameters={
            "epochs": 30,
            "batch-size": 32,
            "learning-rate": 1e-4,
            "weight-decay": 1e-5,
            "patience": 5,
            "num-classes": 4,
            "freeze-layers": 7,
        },
    )
    estimator.fit({"training": s3_data_uri})

Usage (Local):
    python image_classification_model.py \
        --epochs 10 --batch-size 16 --learning-rate 1e-4 \
        --data-dir ./sample_data --output-dir ./model_output
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# torchvision for pre-trained backbone and transforms
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ChestXRayClassifier")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES: List[str] = ["Normal", "Pneumonia", "COVID-19", "Cardiomegaly"]
IMAGE_SIZE: int = 224  # ResNet50 expected input


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA AUGMENTATION TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════
def get_train_transforms() -> transforms.Compose:
    """Medical-imaging-specific augmentation pipeline.

    Includes conservative rotations (chest X-rays have anatomical orientation),
    horizontal flip (left/right lung symmetry), and contrast adjustments to
    simulate varying acquisition protocols.
    """
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),  # conservative for anatomy
            transforms.ColorJitter(
                brightness=0.2, contrast=0.3, saturation=0.1, hue=0.0
            ),
            transforms.RandomAffine(
                degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)
            ),
            transforms.RandomAutocontrast(p=0.3),
            transforms.ToTensor(),
            # ImageNet normalization (pre-trained backbone expects this)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_val_transforms() -> transforms.Compose:
    """Deterministic transform for validation/test sets."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. DATASET — LOADS FROM CSV METADATA + IMAGE FILES / FEATURES
# ═══════════════════════════════════════════════════════════════════════════
class ChestXRayDataset(Dataset):
    """Dataset backed by a CSV manifest and image files (or pre-extracted features).

    Expected CSV columns:
        image_id        — unique identifier (maps to <image_dir>/<image_id>.png)
        patient_id      — FK to Patient 360
        study_date      — DICOM study date
        label           — integer class label (0-3)
        split           — 'train' | 'val' | 'test'

    If images are not present on disk the dataset falls back to generating
    synthetic tensors (for demo/unit-test purposes).
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform or get_val_transforms()

        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        logger.info(
            "Loaded %d records for split='%s' from %s", len(self.df), split, csv_path
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label = int(row["label"])

        img_path = self.image_dir / f"{image_id}.png"
        if img_path.exists():
            image = Image.open(img_path).convert("RGB")
        else:
            # Fallback: generate synthetic 224×224 image for demo
            rng = np.random.RandomState(hash(image_id) % 2**31)
            arr = rng.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            image = Image.fromarray(arr, "RGB")

        image = self.transform(image)
        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency weights for imbalanced classes."""
        counts = self.df["label"].value_counts().sort_index().values.astype(float)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(counts)
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> List[float]:
        """Per-sample weight for WeightedRandomSampler."""
        class_weights = self.get_class_weights().numpy()
        return [class_weights[label] for label in self.df["label"].values]


# ═══════════════════════════════════════════════════════════════════════════
# 3. MODEL — ResNet50 WITH TRANSFER LEARNING
# ═══════════════════════════════════════════════════════════════════════════
class ChestXRayClassifier(nn.Module):
    """ResNet50-based classifier with configurable frozen layers.

    Architecture:
        ResNet50 backbone (ImageNet pre-trained)
        → AdaptiveAvgPool2d
        → Dropout(0.5)
        → Linear(2048, 512)  → ReLU → Dropout(0.3)
        → Linear(512, num_classes)
    """

    def __init__(self, num_classes: int = 4, freeze_layers: int = 7) -> None:
        super().__init__()

        # Load pre-trained ResNet50
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Freeze early layers (conv1 through layer<freeze_layers>)
        all_children = list(backbone.children())
        layers_to_freeze = min(freeze_layers, len(all_children) - 1)
        for i, child in enumerate(all_children[:layers_to_freeze]):
            for param in child.parameters():
                param.requires_grad = False
        logger.info("Froze first %d layer groups of ResNet50", layers_to_freeze)

        # Remove the original FC layer; keep everything up to avgpool
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        # Count trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "Model params: %s trainable / %s total (%.1f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100.0 * trainable / total,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# 4. TRAINING ENGINE
# ═══════════════════════════════════════════════════════════════════════════
class Trainer:
    """Training loop with validation, early stopping, and rich metrics."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        patience: int = 5,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.patience = patience

        # Loss with class weighting for imbalanced medical datasets
        weight = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weight)

        # Optimizer — only update trainable parameters
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = optim.AdamW(
            trainable_params, lr=learning_rate, weight_decay=weight_decay
        )

        # LR scheduler — reduce on plateau of val loss
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2, verbose=True
        )

        # Early stopping state
        self.best_val_loss = float("inf")
        self.best_model_state: Optional[Dict[str, Any]] = None
        self.epochs_no_improve = 0

    # -------------------------------------------------------------------
    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    "  Batch %d/%d — loss: %.4f",
                    batch_idx + 1,
                    len(loader),
                    loss.item(),
                )

        epoch_loss = running_loss / len(loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        return {"loss": epoch_loss, "accuracy": acc}

    # -------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        epoch_loss = running_loss / len(loader.dataset)

        # Compute comprehensive metrics
        metrics: Dict[str, Any] = {
            "loss": epoch_loss,
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision_macro": precision_score(
                all_labels, all_preds, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                all_labels, all_preds, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                all_labels, all_preds, average="macro", zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                all_labels, all_preds, labels=list(range(self.num_classes))
            ).tolist(),
        }

        # AUC-ROC (one-vs-rest)
        try:
            if self.num_classes == 2:
                metrics["auc_roc"] = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                metrics["auc_roc"] = roc_auc_score(
                    all_labels, all_probs, multi_class="ovr", average="macro"
                )
        except ValueError:
            metrics["auc_roc"] = 0.0  # not enough classes in batch

        # Per-class report
        metrics["classification_report"] = classification_report(
            all_labels,
            all_preds,
            target_names=CLASS_NAMES[: self.num_classes],
            output_dict=True,
            zero_division=0,
        )

        return metrics

    # -------------------------------------------------------------------
    def check_early_stopping(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_state = {
                k: v.cpu().clone() for k, v in self.model.state_dict().items()
            }
            self.epochs_no_improve = 0
            return False
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement",
                    self.patience,
                )
                return True
            return False

    def restore_best_model(self) -> None:
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Restored best model (val_loss=%.4f)", self.best_val_loss)


# ═══════════════════════════════════════════════════════════════════════════
# 5. MODEL EXPORT
# ═══════════════════════════════════════════════════════════════════════════
def save_model(
    model: nn.Module,
    output_dir: str,
    num_classes: int,
    metrics: Dict[str, Any],
) -> str:
    """Save model in TorchScript format for SageMaker deployment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    model_cpu = model.cpu()

    # TorchScript via tracing
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    scripted_model = torch.jit.trace(model_cpu, dummy_input)
    ts_path = output_path / "model.pt"
    scripted_model.save(str(ts_path))
    logger.info("Saved TorchScript model to %s", ts_path)

    # Also save standard state dict as backup
    sd_path = output_path / "model_state_dict.pth"
    torch.save(
        {
            "model_state_dict": model_cpu.state_dict(),
            "num_classes": num_classes,
            "class_names": CLASS_NAMES[:num_classes],
            "image_size": IMAGE_SIZE,
        },
        str(sd_path),
    )
    logger.info("Saved state dict to %s", sd_path)

    # Save metrics alongside model
    metrics_path = output_path / "training_metrics.json"
    serializable = _make_serializable(metrics)
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    return str(ts_path)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# 6. SYNTHETIC DATA GENERATOR (for demo when no real data available)
# ═══════════════════════════════════════════════════════════════════════════
def generate_demo_data(data_dir: str, n_samples: int = 200) -> str:
    """Create a synthetic CSV manifest for demo/testing."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    records = []
    for i in range(n_samples):
        label = rng.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.15, 0.15])
        split = rng.choice(["train", "val", "test"], p=[0.7, 0.15, 0.15])
        records.append(
            {
                "image_id": f"CXR_{i:06d}",
                "patient_id": f"PAT_{rng.randint(1000, 9999)}",
                "study_date": f"2024-{rng.randint(1, 13):02d}-{rng.randint(1, 29):02d}",
                "label": label,
                "split": split,
            }
        )

    csv_path = data_path / "imaging_manifest.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    logger.info("Generated demo manifest: %s (%d samples)", csv_path, n_samples)
    return str(csv_path)


# ═══════════════════════════════════════════════════════════════════════════
# 7. MAIN — SAGEMAKER ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Chest X-Ray Classifier (SageMaker compatible)"
    )

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--freeze-layers", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=4)

    # SageMaker environment paths (with local fallbacks)
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
    parser.add_argument(
        "--image-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_IMAGES", "./sample_data/images"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("=" * 70)
    logger.info("Chest X-Ray Classification — Training Started")
    logger.info("=" * 70)
    logger.info("Hyperparameters: %s", vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Data ──────────────────────────────────────────────────────────
    csv_path = Path(args.data_dir) / "imaging_manifest.csv"
    if not csv_path.exists():
        logger.warning("No manifest found — generating synthetic demo data")
        csv_path = Path(generate_demo_data(args.data_dir))

    train_ds = ChestXRayDataset(
        str(csv_path), args.image_dir, split="train", transform=get_train_transforms()
    )
    val_ds = ChestXRayDataset(
        str(csv_path), args.image_dir, split="val", transform=get_val_transforms()
    )

    # Weighted sampler for class imbalance
    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = ChestXRayClassifier(
        num_classes=args.num_classes, freeze_layers=args.freeze_layers
    )
    class_weights = train_ds.get_class_weights()

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        device=device,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        class_weights=class_weights,
    )

    # ── Training Loop ─────────────────────────────────────────────────
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)
        elapsed = time.time() - t0

        trainer.scheduler.step(val_metrics["loss"])

        logger.info(
            "Epoch %d/%d (%.1fs) — "
            "train_loss=%.4f train_acc=%.4f | "
            "val_loss=%.4f val_acc=%.4f val_f1=%.4f val_auc=%.4f",
            epoch,
            args.epochs,
            elapsed,
            train_metrics["loss"],
            train_metrics["accuracy"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["f1_macro"],
            val_metrics["auc_roc"],
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_f1": val_metrics["f1_macro"],
                "val_auc": val_metrics["auc_roc"],
            }
        )

        if trainer.check_early_stopping(val_metrics["loss"]):
            break

    # ── Restore best & final evaluation ───────────────────────────────
    trainer.restore_best_model()
    final_metrics = trainer.validate(val_loader)

    logger.info("=" * 70)
    logger.info("Final Validation Metrics:")
    logger.info("  Accuracy:  %.4f", final_metrics["accuracy"])
    logger.info("  Precision: %.4f", final_metrics["precision_macro"])
    logger.info("  Recall:    %.4f", final_metrics["recall_macro"])
    logger.info("  F1 Macro:  %.4f", final_metrics["f1_macro"])
    logger.info("  AUC-ROC:   %.4f", final_metrics["auc_roc"])
    logger.info("  Confusion Matrix:")
    for i, row in enumerate(final_metrics["confusion_matrix"]):
        logger.info("    %s: %s", CLASS_NAMES[i], row)

    # ── Save ──────────────────────────────────────────────────────────
    final_metrics["training_history"] = history
    save_model(model, args.output_dir, args.num_classes, final_metrics)

    logger.info("=" * 70)
    logger.info("Training complete. Model saved to %s", args.output_dir)
    logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# SageMaker model_fn / input_fn / predict_fn for inference hosting
# ═══════════════════════════════════════════════════════════════════════════
def model_fn(model_dir: str) -> torch.jit.ScriptModule:
    """Load TorchScript model for SageMaker inference."""
    model_path = os.path.join(model_dir, "model.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def input_fn(request_body: bytes, content_type: str = "application/json") -> torch.Tensor:
    """Deserialize input for SageMaker inference."""
    if content_type == "application/json":
        data = json.loads(request_body)
        tensor = torch.tensor(data["instances"], dtype=torch.float32)
        return tensor
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(
    input_data: torch.Tensor, model: torch.jit.ScriptModule
) -> Dict[str, Any]:
    """Run prediction and return class probabilities."""
    device = next(model.parameters()).device
    input_data = input_data.to(device)
    with torch.no_grad():
        outputs = model(input_data)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

    return {
        "predictions": preds.cpu().numpy().tolist(),
        "probabilities": probs.cpu().numpy().tolist(),
        "class_names": CLASS_NAMES,
    }


if __name__ == "__main__":
    main()
