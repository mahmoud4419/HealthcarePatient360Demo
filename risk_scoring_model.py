#!/usr/bin/env python3
"""
risk_scoring_model.py — Patient Readmission / Deterioration Risk Scoring
=========================================================================
Healthcare & Life Sciences Demo — Patient 360 Lakehouse Integration

XGBoost-based risk classification model that predicts patient deterioration
risk using clinical features derived from the Patient 360 lakehouse:
  - Demographics (age, gender)
  - Chronic conditions count
  - Admission history (frequency, recent days)
  - Lab abnormalities
  - Vital sign trends
  - AI imaging findings (from inference_pipeline.py)

Output:
  risk_score      — continuous 0.0–1.0
  risk_category   — Low / Medium / High / Critical
  top_factors     — SHAP-based feature importance explanations

Inspired by dynamic pricing ML patterns, adapted for clinical risk:
  - Feature engineering with temporal windows
  - Gradient boosted trees with regularization
  - Explainable AI (SHAP) for clinical trust
  - Threshold-based risk stratification

Usage (SageMaker):
    estimator = SKLearn(
        entry_point="risk_scoring_model.py",
        role=role,
        instance_type="ml.m5.xlarge",
        framework_version="1.2-1",
        hyperparameters={
            "n-estimators": 300,
            "max-depth": 6,
            "learning-rate": 0.05,
            "min-child-weight": 5,
            "subsample": 0.8,
            "colsample-bytree": 0.8,
        },
    )
    estimator.fit({"training": s3_data_uri})
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# XGBoost for gradient boosting
import xgboost as xgb
from xgboost import XGBClassifier

# SHAP for explainability
import shap

# Scikit-learn utilities
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("RiskScoringModel")
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RISK_CATEGORIES = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
RISK_THRESHOLDS = {"Low": 0.25, "Medium": 0.50, "High": 0.75}  # score boundaries

FEATURE_COLUMNS = [
    # Demographics
    "age",
    "gender_encoded",
    "bmi",
    # Clinical history
    "chronic_conditions_count",
    "admission_count_12m",
    "days_since_last_admission",
    "emergency_visits_6m",
    "length_of_stay_avg",
    # Lab results
    "lab_abnormal_count",
    "lab_critical_count",
    "hemoglobin_latest",
    "wbc_latest",
    "creatinine_latest",
    "glucose_latest",
    # Vital sign trends
    "heart_rate_mean_24h",
    "heart_rate_std_24h",
    "bp_systolic_mean_24h",
    "bp_systolic_trend",  # slope over 24h
    "temperature_max_24h",
    "oxygen_saturation_min_24h",
    "respiratory_rate_mean_24h",
    # Imaging findings (from AI inference)
    "imaging_abnormal_flag",
    "imaging_detection_count",
    "imaging_max_confidence",
    # Medication
    "active_medications_count",
    "high_risk_medications_flag",
    # Social/behavioral
    "smoking_status_encoded",
    "insurance_type_encoded",
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
class FeatureEngineer:
    """Transform raw Patient 360 data into model-ready features.

    Mirrors dynamic pricing feature engineering patterns:
    - Temporal windowing (24h vitals, 6m/12m history)
    - Interaction features
    - Missing value imputation
    - Categorical encoding
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and return transformed features."""
        df = self._base_features(df)
        df = self._encode_categoricals(df, fit=True)
        df = self._engineer_interactions(df)
        df = self._impute_missing(df)

        feature_df = df[FEATURE_COLUMNS].copy()
        feature_df[FEATURE_COLUMNS] = self.scaler.fit_transform(feature_df)
        self._fitted = True
        return feature_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted encoders/scaler."""
        assert self._fitted, "Must call fit_transform first"
        df = self._base_features(df)
        df = self._encode_categoricals(df, fit=False)
        df = self._engineer_interactions(df)
        df = self._impute_missing(df)

        feature_df = df[FEATURE_COLUMNS].copy()
        feature_df[FEATURE_COLUMNS] = self.scaler.transform(feature_df)
        return feature_df

    def _base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute base features from raw data."""
        df = df.copy()

        # Ensure numeric types
        for col in [
            "age", "chronic_conditions_count", "admission_count_12m",
            "lab_abnormal_count", "imaging_detection_count",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Derived features
        if "days_since_last_admission" not in df.columns:
            df["days_since_last_admission"] = np.random.randint(1, 365, len(df))

        if "bp_systolic_trend" not in df.columns:
            # Slope of systolic BP over 24h (positive = rising)
            df["bp_systolic_trend"] = np.random.normal(0, 5, len(df))

        return df

    def _encode_categoricals(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        categorical_mappings = {
            "gender": "gender_encoded",
            "smoking_status": "smoking_status_encoded",
            "insurance_type": "insurance_type_encoded",
        }

        for src_col, dst_col in categorical_mappings.items():
            if src_col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[dst_col] = le.fit_transform(df[src_col].fillna("Unknown"))
                    self.label_encoders[src_col] = le
                else:
                    le = self.label_encoders.get(src_col)
                    if le:
                        # Handle unseen labels
                        known = set(le.classes_)
                        df[dst_col] = df[src_col].fillna("Unknown").apply(
                            lambda x: le.transform([x])[0] if x in known else -1
                        )
                    else:
                        df[dst_col] = 0
            elif dst_col not in df.columns:
                df[dst_col] = 0

        return df

    def _engineer_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        df = df.copy()

        # High-risk medication with many conditions = compounded risk
        if "high_risk_medications_flag" in df.columns and "chronic_conditions_count" in df.columns:
            df["med_condition_interaction"] = (
                df["high_risk_medications_flag"] * df["chronic_conditions_count"]
            )

        # Vital instability score
        vital_std_cols = [c for c in df.columns if "_std_" in c]
        if vital_std_cols:
            df["vital_instability"] = df[vital_std_cols].mean(axis=1)

        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values with median (numeric) or mode (categorical)."""
        df = df.copy()
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
            else:
                if df[col].dtype in [np.float64, np.float32, np.int64]:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(0)
        return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. RISK SCORING MODEL
# ═══════════════════════════════════════════════════════════════════════════
class PatientRiskModel:
    """XGBoost-based patient risk scoring with SHAP explainability.

    Multi-class classification: Low (0) / Medium (1) / High (2) / Critical (3)
    Also outputs a continuous risk_score via calibrated probability.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        min_child_weight: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        scale_pos_weight: float = 1.0,
    ) -> None:
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",  # fast histogram-based
        )
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.feature_engineer = FeatureEngineer()
        self.shap_explainer: Optional[shap.TreeExplainer] = None

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        target_col: str = "risk_label",
    ) -> Dict[str, Any]:
        """Train the risk model and return metrics."""
        logger.info("Preparing features…")
        X_train = self.feature_engineer.fit_transform(train_df)
        y_train = train_df[target_col].values

        eval_set = []
        if val_df is not None:
            X_val = self.feature_engineer.transform(val_df)
            y_val = val_df[target_col].values
            eval_set = [(X_val, y_val)]

        # ── Train XGBoost ─────────────────────────────────────────────
        logger.info("Training XGBoost model…")
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set or [(X_train, y_train)],
            verbose=50,
        )

        # ── Probability calibration ──────────────────────────────────
        logger.info("Calibrating probabilities (isotonic regression)…")
        self.calibrated_model = CalibratedClassifierCV(
            self.model, method="isotonic", cv=3
        )
        self.calibrated_model.fit(X_train, y_train)

        # ── Cross-validation ──────────────────────────────────────────
        logger.info("Running 5-fold cross-validation…")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.model, X_train, y_train, cv=cv, scoring="f1_macro"
        )

        # ── Compute metrics ───────────────────────────────────────────
        metrics = self._compute_metrics(X_train, y_train, "train")
        metrics["cv_f1_mean"] = float(cv_scores.mean())
        metrics["cv_f1_std"] = float(cv_scores.std())

        if val_df is not None:
            val_metrics = self._compute_metrics(X_val, y_val, "val")
            metrics.update(val_metrics)

        # ── Initialize SHAP ───────────────────────────────────────────
        logger.info("Initializing SHAP explainer…")
        self.shap_explainer = shap.TreeExplainer(self.model)

        # Feature importance
        importance = dict(
            zip(FEATURE_COLUMNS, self.model.feature_importances_.tolist())
        )
        metrics["feature_importance"] = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        return metrics

    def predict(
        self, df: pd.DataFrame, explain: bool = True
    ) -> pd.DataFrame:
        """Generate risk predictions with optional SHAP explanations.

        Returns DataFrame with columns:
          risk_score, risk_category, risk_probabilities, top_contributing_factors
        """
        X = self.feature_engineer.transform(df)

        # Calibrated probabilities
        if self.calibrated_model:
            probs = self.calibrated_model.predict_proba(X)
        else:
            probs = self.model.predict_proba(X)

        # Risk score: weighted sum of class probabilities
        # Weights: Low=0.0, Medium=0.33, High=0.67, Critical=1.0
        risk_weights = np.array([0.0, 0.33, 0.67, 1.0])
        risk_scores = (probs * risk_weights).sum(axis=1)

        # Risk category from thresholds
        risk_categories = np.digitize(
            risk_scores,
            bins=[RISK_THRESHOLDS["Low"], RISK_THRESHOLDS["Medium"], RISK_THRESHOLDS["High"]],
        )

        results = pd.DataFrame(
            {
                "risk_score": risk_scores,
                "risk_category": [RISK_CATEGORIES[c] for c in risk_categories],
                "risk_category_id": risk_categories,
                "prob_low": probs[:, 0],
                "prob_medium": probs[:, 1],
                "prob_high": probs[:, 2],
                "prob_critical": probs[:, 3],
            }
        )

        # SHAP explanations
        if explain and self.shap_explainer is not None:
            explanations = self._explain(X)
            results["top_contributing_factors"] = explanations

        return results

    def _explain(self, X: pd.DataFrame) -> List[str]:
        """Generate SHAP-based explanations for each prediction."""
        shap_values = self.shap_explainer.shap_values(X)

        explanations = []
        for i in range(len(X)):
            # Get SHAP values for predicted class
            if isinstance(shap_values, list):
                # Multi-class: list of arrays
                pred_class = self.model.predict(X.iloc[[i]])[0]
                sv = shap_values[pred_class][i]
            else:
                sv = shap_values[i]

            # Top 5 contributing features
            top_indices = np.argsort(np.abs(sv))[-5:][::-1]
            factors = []
            for idx in top_indices:
                feature_name = FEATURE_COLUMNS[idx]
                shap_val = sv[idx]
                direction = "↑ increases" if shap_val > 0 else "↓ decreases"
                raw_val = X.iloc[i, idx]
                factors.append(
                    f"{feature_name}={raw_val:.2f} ({direction} risk, SHAP={shap_val:.3f})"
                )

            explanations.append(json.dumps(factors))

        return explanations

    def _compute_metrics(
        self, X: pd.DataFrame, y: np.ndarray, prefix: str
    ) -> Dict[str, Any]:
        """Compute classification metrics."""
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)

        metrics = {
            f"{prefix}_accuracy": float(accuracy_score(y, y_pred)),
            f"{prefix}_f1_macro": float(
                f1_score(y, y_pred, average="macro", zero_division=0)
            ),
            f"{prefix}_log_loss": float(log_loss(y, y_proba)),
            f"{prefix}_confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        }

        try:
            metrics[f"{prefix}_auc_roc"] = float(
                roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics[f"{prefix}_auc_roc"] = 0.0

        metrics[f"{prefix}_classification_report"] = classification_report(
            y,
            y_pred,
            target_names=list(RISK_CATEGORIES.values()),
            output_dict=True,
            zero_division=0,
        )

        return metrics

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------
    def save(self, output_dir: str, metrics: Optional[Dict] = None) -> None:
        """Save model, feature engineer, and metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # XGBoost native format
        self.model.save_model(str(output_path / "risk_model.xgb"))
        logger.info("Saved XGBoost model to %s", output_path / "risk_model.xgb")

        # Pickle the full pipeline (model + feature engineer + calibration)
        pipeline = {
            "model": self.model,
            "calibrated_model": self.calibrated_model,
            "feature_engineer": self.feature_engineer,
            "feature_columns": FEATURE_COLUMNS,
            "risk_categories": RISK_CATEGORIES,
            "risk_thresholds": RISK_THRESHOLDS,
        }
        with open(output_path / "risk_pipeline.pkl", "wb") as f:
            pickle.dump(pipeline, f)
        logger.info("Saved full pipeline to %s", output_path / "risk_pipeline.pkl")

        if metrics:
            with open(output_path / "risk_metrics.json", "w") as f:
                json.dump(_make_serializable(metrics), f, indent=2)

    @classmethod
    def load(cls, model_dir: str) -> "PatientRiskModel":
        """Load a saved model pipeline."""
        model_path = Path(model_dir)
        with open(model_path / "risk_pipeline.pkl", "rb") as f:
            pipeline = pickle.load(f)

        instance = cls.__new__(cls)
        instance.model = pipeline["model"]
        instance.calibrated_model = pipeline["calibrated_model"]
        instance.feature_engineer = pipeline["feature_engineer"]
        instance.shap_explainer = shap.TreeExplainer(instance.model)
        return instance


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# 3. SYNTHETIC DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════
def generate_synthetic_patient_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic patient data for training/demo.

    Risk labels are derived from a combination of features to ensure
    clinical plausibility (not purely random).
    """
    rng = np.random.RandomState(seed)

    data = {
        # Demographics
        "patient_id": [f"PAT_{i:06d}" for i in range(n)],
        "age": rng.normal(65, 15, n).clip(18, 100).astype(int),
        "gender": rng.choice(["M", "F"], n),
        "bmi": rng.normal(27, 6, n).clip(15, 50),
        # Clinical history
        "chronic_conditions_count": rng.poisson(2, n).clip(0, 10),
        "admission_count_12m": rng.poisson(1.5, n).clip(0, 8),
        "days_since_last_admission": rng.exponential(90, n).clip(1, 365).astype(int),
        "emergency_visits_6m": rng.poisson(0.8, n).clip(0, 5),
        "length_of_stay_avg": rng.exponential(5, n).clip(1, 30),
        # Lab results
        "lab_abnormal_count": rng.poisson(3, n).clip(0, 15),
        "lab_critical_count": rng.poisson(0.5, n).clip(0, 5),
        "hemoglobin_latest": rng.normal(13, 2.5, n).clip(5, 20),
        "wbc_latest": rng.normal(7.5, 3, n).clip(1, 25),
        "creatinine_latest": rng.exponential(1.2, n).clip(0.3, 10),
        "glucose_latest": rng.normal(110, 40, n).clip(50, 400),
        # Vitals
        "heart_rate_mean_24h": rng.normal(78, 15, n).clip(40, 150),
        "heart_rate_std_24h": rng.exponential(8, n).clip(1, 40),
        "bp_systolic_mean_24h": rng.normal(130, 20, n).clip(80, 200),
        "bp_systolic_trend": rng.normal(0, 5, n),
        "temperature_max_24h": rng.normal(37.0, 0.8, n).clip(35, 41),
        "oxygen_saturation_min_24h": rng.normal(96, 3, n).clip(70, 100),
        "respiratory_rate_mean_24h": rng.normal(16, 4, n).clip(8, 40),
        # Imaging
        "imaging_abnormal_flag": rng.choice([0, 1], n, p=[0.6, 0.4]),
        "imaging_detection_count": rng.poisson(0.5, n).clip(0, 5),
        "imaging_max_confidence": rng.uniform(0, 1, n),
        # Medication
        "active_medications_count": rng.poisson(5, n).clip(0, 20),
        "high_risk_medications_flag": rng.choice([0, 1], n, p=[0.7, 0.3]),
        # Social
        "smoking_status": rng.choice(
            ["Never", "Former", "Current"], n, p=[0.5, 0.3, 0.2]
        ),
        "insurance_type": rng.choice(
            ["Private", "Medicare", "Medicaid", "Uninsured"],
            n,
            p=[0.4, 0.3, 0.2, 0.1],
        ),
    }

    df = pd.DataFrame(data)

    # Derive risk labels from clinical logic (not random)
    risk_score = (
        0.15 * (df["age"] - 18) / 82
        + 0.12 * df["chronic_conditions_count"] / 10
        + 0.10 * df["admission_count_12m"] / 8
        + 0.08 * df["lab_abnormal_count"] / 15
        + 0.10 * df["lab_critical_count"] / 5
        + 0.08 * (df["heart_rate_mean_24h"] - 60) / 90
        + 0.07 * (200 - df["bp_systolic_mean_24h"]).clip(0) / 120  # low BP = risk
        + 0.08 * (100 - df["oxygen_saturation_min_24h"]) / 30
        + 0.06 * df["imaging_abnormal_flag"]
        + 0.05 * df["imaging_detection_count"] / 5
        + 0.05 * df["active_medications_count"] / 20
        + 0.06 * df["high_risk_medications_flag"]
    )

    # Add noise and clip
    risk_score += rng.normal(0, 0.08, n)
    risk_score = risk_score.clip(0, 1)

    # Assign categories with clinical bias
    df["risk_label"] = np.digitize(
        risk_score,
        bins=[RISK_THRESHOLDS["Low"], RISK_THRESHOLDS["Medium"], RISK_THRESHOLDS["High"]],
    )

    # Train/val/test split
    df["split"] = rng.choice(
        ["train", "val", "test"], n, p=[0.7, 0.15, 0.15]
    )

    logger.info("Generated %d synthetic patient records", n)
    logger.info("Risk distribution: %s", df["risk_label"].value_counts().to_dict())

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. MAIN — SAGEMAKER ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Patient Risk Scoring Model (SageMaker compatible)"
    )

    # Hyperparameters
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-child-weight", type=int, default=5)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.1)
    parser.add_argument("--reg-lambda", type=float, default=1.0)

    # Data & output paths
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
        "--generate-data",
        action="store_true",
        help="Generate synthetic training data",
    )
    parser.add_argument("--n-samples", type=int, default=2000)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("=" * 70)
    logger.info("Patient Risk Scoring Model — Training Started")
    logger.info("=" * 70)
    logger.info("Hyperparameters: %s", vars(args))

    # ── Load or generate data ─────────────────────────────────────────
    data_path = Path(args.data_dir) / "patient_risk_data.csv"
    if data_path.exists() and not args.generate_data:
        logger.info("Loading data from %s", data_path)
        df = pd.read_csv(str(data_path))
    else:
        logger.info("Generating synthetic data (%d samples)", args.n_samples)
        df = generate_synthetic_patient_data(n=args.n_samples)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(data_path), index=False)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    logger.info(
        "Data splits: train=%d, val=%d, test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    # ── Build & Train ─────────────────────────────────────────────────
    model = PatientRiskModel(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
    )

    metrics = model.train(train_df, val_df, target_col="risk_label")

    # ── Evaluate on test set ──────────────────────────────────────────
    logger.info("Evaluating on test set…")
    test_predictions = model.predict(test_df, explain=True)

    # Merge predictions with test data for analysis
    test_results = test_df.reset_index(drop=True).copy()
    test_results = pd.concat(
        [test_results, test_predictions.reset_index(drop=True)], axis=1
    )

    # Test metrics
    test_acc = accuracy_score(
        test_df["risk_label"].values, test_predictions["risk_category_id"].values
    )
    test_f1 = f1_score(
        test_df["risk_label"].values,
        test_predictions["risk_category_id"].values,
        average="macro",
        zero_division=0,
    )
    metrics["test_accuracy"] = float(test_acc)
    metrics["test_f1_macro"] = float(test_f1)

    # ── Log results ───────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Training Results:")
    logger.info("  Train accuracy:  %.4f", metrics.get("train_accuracy", 0))
    logger.info("  Val accuracy:    %.4f", metrics.get("val_accuracy", 0))
    logger.info("  Test accuracy:   %.4f", test_acc)
    logger.info("  Test F1 (macro): %.4f", test_f1)
    logger.info("  CV F1 (mean±std): %.4f ± %.4f", metrics["cv_f1_mean"], metrics["cv_f1_std"])
    logger.info("")
    logger.info("  Top 10 features by importance:")
    for i, (feat, imp) in enumerate(
        list(metrics["feature_importance"].items())[:10]
    ):
        logger.info("    %2d. %-35s %.4f", i + 1, feat, imp)

    logger.info("")
    logger.info("  Risk prediction distribution (test set):")
    for cat, count in test_predictions["risk_category"].value_counts().items():
        logger.info("    %-10s %d (%.1f%%)", cat, count, 100 * count / len(test_predictions))

    # ── Save ──────────────────────────────────────────────────────────
    model.save(args.output_dir, metrics)
    logger.info("=" * 70)
    logger.info("Training complete. Model saved to %s", args.output_dir)


# ═══════════════════════════════════════════════════════════════════════════
# SageMaker model_fn for hosting
# ═══════════════════════════════════════════════════════════════════════════
def model_fn(model_dir: str) -> PatientRiskModel:
    """Load model for SageMaker inference."""
    return PatientRiskModel.load(model_dir)


def input_fn(request_body: bytes, content_type: str = "application/json") -> pd.DataFrame:
    if content_type == "application/json":
        data = json.loads(request_body)
        return pd.DataFrame(data["instances"])
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(
    input_data: pd.DataFrame, model: PatientRiskModel
) -> Dict[str, Any]:
    predictions = model.predict(input_data, explain=True)
    return {
        "predictions": predictions.to_dict(orient="records"),
    }


if __name__ == "__main__":
    main()
