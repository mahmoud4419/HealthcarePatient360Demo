# %% [markdown]
# # 🏥 Healthcare AI — Patient Risk Scoring & Medical Imaging Analysis
#
# **SageMaker Notebook: End-to-End ML Pipeline for Patient 360 Lakehouse**
#
# This notebook demonstrates:
# 1. Loading sample data from Iceberg tables via Athena
# 2. Feature engineering for patient risk scoring
# 3. Training the XGBoost risk model
# 4. Deploying to a SageMaker endpoint
# 5. Running batch inference
# 6. Visualizing results with clinical context
#
# ---
# **Prerequisites:**
# - SageMaker notebook instance with `conda_pytorch_p310` kernel
# - IAM role with Athena, S3, and Glue permissions
# - Patient 360 lakehouse configured (`healthcare_db`)

# %% [markdown]
# ## 1. Environment Setup

# %%
import os
import json
import time
import warnings
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.figsize"] = (12, 6)
matplotlib.rcParams["font.size"] = 11
sns.set_style("whitegrid")

# SageMaker SDK
import sagemaker
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.sklearn import SKLearn
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter

print(f"SageMaker SDK version: {sagemaker.__version__}")
print(f"Timestamp: {datetime.now().isoformat()}")

# %%
# Configuration
REGION = boto3.Session().region_name
ROLE = get_execution_role()
SESSION = sagemaker.Session()
BUCKET = SESSION.default_bucket()
PREFIX = "healthcare-demo/patient360"

# Lakehouse config
DATABASE = "healthcare_db"
IMAGING_TABLE = "imaging_metadata"
PATIENT_TABLE = "patient_demographics"
LABS_TABLE = "lab_results"
RESULTS_TABLE = "ai_imaging_results"

print(f"Region:  {REGION}")
print(f"Role:    {ROLE}")
print(f"Bucket:  s3://{BUCKET}/{PREFIX}")

# %% [markdown]
# ## 2. Load Data from Iceberg Tables via Athena
#
# Query the Patient 360 lakehouse to build our training dataset.
# The lakehouse stores data in Apache Iceberg format, queryable via Athena.

# %%
# Athena query helper
athena_client = boto3.client("athena", region_name=REGION)


def run_athena_query(query: str, database: str = DATABASE) -> pd.DataFrame:
    """Execute Athena query and return results as DataFrame."""
    output_location = f"s3://{BUCKET}/{PREFIX}/athena-results/"

    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_location},
    )
    query_id = response["QueryExecutionId"]
    print(f"Query started: {query_id}")

    # Wait for completion
    while True:
        status = athena_client.get_query_execution(QueryExecutionId=query_id)
        state = status["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(2)

    if state != "SUCCEEDED":
        reason = status["QueryExecution"]["Status"].get("StateChangeReason", "")
        raise RuntimeError(f"Query {state}: {reason}")

    # Read results
    result_uri = f"{output_location}{query_id}.csv"
    result_path = f"/tmp/athena_{query_id}.csv"

    s3 = boto3.client("s3")
    result_bucket, result_key = result_uri[5:].split("/", 1)
    s3.download_file(result_bucket, result_key, result_path)

    return pd.read_csv(result_path)


# %%
# Query 1: Patient demographics + admission history
patient_query = """
SELECT
    p.patient_id,
    p.age,
    p.gender,
    p.bmi,
    p.smoking_status,
    p.insurance_type,
    p.chronic_conditions_count,
    COALESCE(a.admission_count_12m, 0)     AS admission_count_12m,
    COALESCE(a.emergency_visits_6m, 0)     AS emergency_visits_6m,
    COALESCE(a.length_of_stay_avg, 0)      AS length_of_stay_avg,
    COALESCE(a.days_since_last_admission, 365) AS days_since_last_admission
FROM healthcare_db.patient_demographics p
LEFT JOIN (
    SELECT
        patient_id,
        COUNT(CASE WHEN admission_date > CURRENT_DATE - INTERVAL '12' MONTH THEN 1 END) AS admission_count_12m,
        COUNT(CASE WHEN admission_type = 'EMERGENCY' AND admission_date > CURRENT_DATE - INTERVAL '6' MONTH THEN 1 END) AS emergency_visits_6m,
        AVG(length_of_stay_days)          AS length_of_stay_avg,
        DATE_DIFF('day', MAX(admission_date), CURRENT_DATE) AS days_since_last_admission
    FROM healthcare_db.admissions
    GROUP BY patient_id
) a ON p.patient_id = a.patient_id
LIMIT 5000
"""

print("Querying patient demographics + admissions...")
# In production: patients_df = run_athena_query(patient_query)
# For demo, generate synthetic data:
from risk_scoring_model import generate_synthetic_patient_data

patients_df = generate_synthetic_patient_data(n=2000)
print(f"Loaded {len(patients_df)} patient records")
patients_df.head()

# %%
# Query 2: Lab results (aggregated)
labs_query = """
SELECT
    patient_id,
    COUNT(CASE WHEN is_abnormal = true THEN 1 END) AS lab_abnormal_count,
    COUNT(CASE WHEN is_critical = true THEN 1 END) AS lab_critical_count,
    MAX(CASE WHEN test_name = 'Hemoglobin' THEN result_value END) AS hemoglobin_latest,
    MAX(CASE WHEN test_name = 'WBC' THEN result_value END) AS wbc_latest,
    MAX(CASE WHEN test_name = 'Creatinine' THEN result_value END) AS creatinine_latest,
    MAX(CASE WHEN test_name = 'Glucose' THEN result_value END) AS glucose_latest
FROM healthcare_db.lab_results
WHERE collection_date > CURRENT_DATE - INTERVAL '30' DAY
GROUP BY patient_id
"""
print("Lab results already included in synthetic data")

# %%
# Query 3: AI imaging results (from inference_pipeline.py)
imaging_query = """
SELECT
    patient_id,
    MAX(CASE WHEN classification_label != 'Normal' THEN 1 ELSE 0 END) AS imaging_abnormal_flag,
    SUM(detection_count)     AS imaging_detection_count,
    MAX(primary_anomaly_confidence) AS imaging_max_confidence
FROM healthcare_db.ai_imaging_results
WHERE ai_inference_timestamp > CURRENT_DATE - INTERVAL '90' DAY
GROUP BY patient_id
"""
print("Imaging results already included in synthetic data")

# %% [markdown]
# ## 3. Exploratory Data Analysis

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Patient Risk Factor Distributions", fontsize=16, fontweight="bold")

# Age distribution by risk
risk_map = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
patients_df["risk_name"] = patients_df["risk_label"].map(risk_map)

for i, (col, title) in enumerate([
    ("age", "Age Distribution"),
    ("chronic_conditions_count", "Chronic Conditions"),
    ("admission_count_12m", "Admissions (12mo)"),
    ("lab_abnormal_count", "Abnormal Labs"),
    ("oxygen_saturation_min_24h", "Min SpO₂ (24h)"),
    ("imaging_detection_count", "Imaging Detections"),
]):
    ax = axes[i // 3, i % 3]
    for risk in ["Low", "Medium", "High", "Critical"]:
        subset = patients_df[patients_df["risk_name"] == risk]
        ax.hist(subset[col], alpha=0.5, label=risk, bins=20)
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("risk_factor_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Saved risk_factor_distributions.png")

# %%
# Correlation matrix of key features
feature_cols = [
    "age", "chronic_conditions_count", "admission_count_12m",
    "lab_abnormal_count", "lab_critical_count", "heart_rate_mean_24h",
    "bp_systolic_mean_24h", "oxygen_saturation_min_24h",
    "imaging_detection_count", "active_medications_count", "risk_label",
]
corr = patients_df[feature_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, ax=ax)
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("feature_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Feature Engineering

# %%
from risk_scoring_model import FeatureEngineer, FEATURE_COLUMNS

fe = FeatureEngineer()

train_df = patients_df[patients_df["split"] == "train"].copy()
val_df = patients_df[patients_df["split"] == "val"].copy()
test_df = patients_df[patients_df["split"] == "test"].copy()

X_train = fe.fit_transform(train_df)
X_val = fe.transform(val_df)
X_test = fe.transform(test_df)

y_train = train_df["risk_label"].values
y_val = val_df["risk_label"].values
y_test = test_df["risk_label"].values

print(f"Feature matrix shape: {X_train.shape}")
print(f"Features: {list(X_train.columns)}")
print(f"\nTraining set risk distribution:")
print(pd.Series(y_train).value_counts().sort_index().rename(risk_map))

# %% [markdown]
# ## 5. Train Risk Scoring Model (Local)

# %%
from risk_scoring_model import PatientRiskModel

# Train locally first to validate
model = PatientRiskModel(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
)

metrics = model.train(train_df, val_df, target_col="risk_label")

print("\n" + "=" * 60)
print("LOCAL TRAINING RESULTS")
print("=" * 60)
print(f"Train Accuracy:  {metrics.get('train_accuracy', 0):.4f}")
print(f"Val Accuracy:    {metrics.get('val_accuracy', 0):.4f}")
print(f"CV F1 (mean±std): {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
print(f"\nTop 10 Features:")
for i, (feat, imp) in enumerate(list(metrics["feature_importance"].items())[:10]):
    print(f"  {i+1:2d}. {feat:<35s} {imp:.4f}")

# %%
# Feature importance visualization
importance = metrics["feature_importance"]
top_features = dict(list(importance.items())[:15])

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(
    list(reversed(top_features.keys())),
    list(reversed(top_features.values())),
    color=sns.color_palette("viridis", len(top_features)),
)
ax.set_xlabel("Feature Importance (Gain)")
ax.set_title("Top 15 Risk Factors — XGBoost Feature Importance", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Deploy to SageMaker Endpoint

# %%
# Upload training data to S3
train_data_path = f"s3://{BUCKET}/{PREFIX}/training-data/patient_risk_data.csv"
patients_df.to_csv("/tmp/patient_risk_data.csv", index=False)
SESSION.upload_data(
    path="/tmp/patient_risk_data.csv",
    bucket=BUCKET,
    key_prefix=f"{PREFIX}/training-data",
)
print(f"Training data uploaded to {train_data_path}")

# %%
# SageMaker Training Job
sklearn_estimator = SKLearn(
    entry_point="risk_scoring_model.py",
    source_dir="./",
    role=ROLE,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    framework_version="1.2-1",
    py_version="py3",
    hyperparameters={
        "n-estimators": 300,
        "max-depth": 6,
        "learning-rate": 0.05,
        "min-child-weight": 5,
        "subsample": 0.8,
        "colsample-bytree": 0.8,
        "generate-data": "",  # generate if not found
        "n-samples": 5000,
    },
    output_path=f"s3://{BUCKET}/{PREFIX}/model-artifacts/",
    base_job_name="healthcare-risk-scoring",
)

# Uncomment to run on SageMaker:
# sklearn_estimator.fit({"training": train_data_path})
print("✓ Estimator configured (uncomment .fit() to launch training job)")

# %%
# Hyperparameter Tuning (optional)
tuner = HyperparameterTuner(
    estimator=sklearn_estimator,
    objective_metric_name="val:f1_macro",
    objective_type="Maximize",
    hyperparameter_ranges={
        "n-estimators": IntegerParameter(100, 500),
        "max-depth": IntegerParameter(3, 10),
        "learning-rate": ContinuousParameter(0.01, 0.3),
        "min-child-weight": IntegerParameter(1, 10),
        "subsample": ContinuousParameter(0.6, 1.0),
        "colsample-bytree": ContinuousParameter(0.6, 1.0),
    },
    max_jobs=20,
    max_parallel_jobs=4,
    base_tuning_job_name="healthcare-risk-tuning",
)
# Uncomment to launch: tuner.fit({"training": train_data_path})
print("✓ Tuner configured (uncomment .fit() to launch HPO)")

# %%
# Deploy the model
# model = sklearn_estimator.create_model()
# predictor = model.deploy(
#     initial_instance_count=1,
#     instance_type="ml.m5.large",
#     endpoint_name="healthcare-risk-scoring-endpoint",
# )
print("✓ Deployment config ready (uncomment to deploy endpoint)")

# %% [markdown]
# ## 7. Batch Inference

# %%
# Run predictions on test set
test_predictions = model.predict(test_df, explain=True)

# Merge with original data
results = test_df.reset_index(drop=True).copy()
results = pd.concat([results, test_predictions.reset_index(drop=True)], axis=1)

print(f"\nBatch inference complete: {len(results)} patients scored")
print(f"\nRisk Distribution:")
print(results["risk_category"].value_counts().sort_index())

# Preview
results[
    ["patient_id", "age", "chronic_conditions_count", "risk_score", "risk_category"]
].head(10)

# %% [markdown]
# ## 8. Visualize Results

# %%
# Risk score distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 8a. Risk score histogram
colors = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c", "Critical": "#8e44ad"}
for cat in ["Low", "Medium", "High", "Critical"]:
    subset = results[results["risk_category"] == cat]
    axes[0].hist(subset["risk_score"], bins=20, alpha=0.7, label=cat, color=colors[cat])
axes[0].set_xlabel("Risk Score")
axes[0].set_ylabel("Count")
axes[0].set_title("Risk Score Distribution")
axes[0].legend()
axes[0].axvline(0.25, color="gray", linestyle="--", alpha=0.5)
axes[0].axvline(0.50, color="gray", linestyle="--", alpha=0.5)
axes[0].axvline(0.75, color="gray", linestyle="--", alpha=0.5)

# 8b. Risk category pie chart
cat_counts = results["risk_category"].value_counts()
axes[1].pie(
    cat_counts.values,
    labels=cat_counts.index,
    colors=[colors.get(c, "gray") for c in cat_counts.index],
    autopct="%1.1f%%",
    startangle=90,
)
axes[1].set_title("Risk Category Distribution")

# 8c. Age vs Risk Score scatter
scatter = axes[2].scatter(
    results["age"],
    results["risk_score"],
    c=results["risk_category_id"],
    cmap="RdYlGn_r",
    alpha=0.5,
    s=20,
)
axes[2].set_xlabel("Age")
axes[2].set_ylabel("Risk Score")
axes[2].set_title("Age vs. Risk Score")
plt.colorbar(scatter, ax=axes[2], label="Risk Level")

plt.suptitle("Patient Risk Scoring Results", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("risk_scoring_results.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(test_df["risk_label"].values, test_predictions["risk_category_id"].values)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(cm, display_labels=["Low", "Medium", "High", "Critical"])
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Risk Prediction — Confusion Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# SHAP summary plot (for explainability demo)
import shap

explainer = shap.TreeExplainer(model.model)
shap_values = explainer.shap_values(X_test)

fig = plt.figure(figsize=(12, 8))
# For multi-class, show the "Critical" class SHAP values
if isinstance(shap_values, list) and len(shap_values) > 3:
    shap.summary_plot(shap_values[3], X_test, feature_names=FEATURE_COLUMNS, show=False)
else:
    shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLUMNS, show=False)
plt.title("SHAP Feature Importance — Critical Risk Class", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Write Results to Iceberg Table
#
# Output enriched predictions to the Patient 360 lakehouse for downstream
# consumption by clinical dashboards, alerting systems, and care teams.

# %%
# Prepare output for Iceberg
output_df = results[
    [
        "patient_id", "age", "gender", "chronic_conditions_count",
        "risk_score", "risk_category", "risk_category_id",
        "prob_low", "prob_medium", "prob_high", "prob_critical",
        "top_contributing_factors",
    ]
].copy()

output_df["inference_timestamp"] = datetime.now().isoformat()
output_df["model_version"] = "risk-v1.0.0"
output_df["model_type"] = "xgboost"

# Write as Parquet (Iceberg-compatible)
output_path = f"/tmp/ai_risk_results.parquet"
output_df.to_parquet(output_path, engine="pyarrow", index=False)
print(f"Wrote {len(output_df)} records to {output_path}")

# Upload to S3
s3_output = f"s3://{BUCKET}/{PREFIX}/results/risk-scoring/"
SESSION.upload_data(
    path=output_path,
    bucket=BUCKET,
    key_prefix=f"{PREFIX}/results/risk-scoring",
)
print(f"Uploaded to {s3_output}")

# %%
# Register with Glue / Iceberg (production code)
register_query = f"""
ALTER TABLE {DATABASE}.patient_risk_scores
ADD IF NOT EXISTS PARTITION (
    year = '{datetime.now().year}',
    month = '{datetime.now().month:02d}'
)
LOCATION '{s3_output}'
"""
print("Glue registration query (execute in production):")
print(register_query)
# run_athena_query(register_query)

# %% [markdown]
# ## 10. Summary & Next Steps
#
# ### Results
# | Metric | Value |
# |--------|-------|
# | Test Accuracy | {test_acc:.4f} |
# | Test F1 (macro) | {test_f1:.4f} |
# | CV F1 (mean±std) | {cv_mean:.4f} ± {cv_std:.4f} |
#
# ### Integration Points
# - **Iceberg tables**: `healthcare_db.ai_imaging_results`, `healthcare_db.patient_risk_scores`
# - **SageMaker endpoint**: `healthcare-risk-scoring-endpoint`
# - **Downstream consumers**: Clinical dashboard, care team alerts, EHR integration
#
# ### Next Steps
# 1. **Production deployment** — Enable auto-scaling on the SageMaker endpoint
# 2. **Model monitoring** — Set up SageMaker Model Monitor for data drift
# 3. **A/B testing** — Shadow deploy new model versions
# 4. **Feature store** — Migrate features to SageMaker Feature Store
# 5. **Real-time scoring** — Integrate with EventBridge for admission events

# %%
print("=" * 60)
print("🏥 Healthcare AI Demo — Notebook Complete")
print("=" * 60)
print(f"  Patients scored:    {len(results)}")
print(f"  High/Critical risk: {(results['risk_category'].isin(['High', 'Critical'])).sum()}")
print(f"  Model saved to:     {s3_output}")
print(f"  Timestamp:          {datetime.now().isoformat()}")
