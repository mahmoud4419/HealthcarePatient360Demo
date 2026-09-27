# Healthcare Patient 360 — Lakehouse Demo Scripts

> **Architecture Pattern**: S3 → Glue/Spark → Apache Iceberg → Athena  
> **Inspired by**: Petra Ride (bronze→silver→gold layering) & Gulf Air Falcon Eye (CDC merge, dual-sink streaming)

---

## 📁 File Inventory

| # | File | Type | Purpose |
|---|------|------|---------|
| 1 | `transform_patients.py` | PySpark/Glue | Patient dimension table — schema validation, cleansing, dedup → `dim_patients` |
| 2 | `transform_admissions.py` | PySpark/Glue | Admissions fact — LOS calc, severity categorization, CDC MERGE → `fact_admissions` |
| 3 | `transform_imaging.py` | PySpark/Glue | Imaging fact — radiology enrichment, patient demographics → `fact_imaging` |
| 4 | `transform_labs.py` | PySpark/Glue | Lab results — abnormal flag derivation, trend analysis → `fact_lab_results` |
| 5 | `transform_omics.py` | PySpark/Glue | Genomic variants — ClinVar annotation, pharmacogenomics → `fact_omics` |
| 6 | `stream_vitals.py` | Python/Kinesis | Streaming vitals — 5-min windowed aggregation, dual-sink (Iceberg + OpenSearch) → `fact_vitals` |
| 7 | `dag_healthcare_pipeline.py` | Airflow DAG | End-to-end orchestration — sensors → transforms → 360 view → risk scoring → dashboard |
| 8 | `create_iceberg_tables.sql` | Athena SQL | DDL for all 7 Iceberg tables with partitioning and table properties |
| 9 | `patient_360_view.sql` | Athena SQL | Unified Patient 360 CTAS view joining all fact tables with composite risk scoring |

---

## 🏗️ Architecture Overview

```
                           ┌──────────────────────────────────┐
                           │     Amazon S3 (Raw Zone)         │
                           │  patients.csv, admissions.csv,   │
                           │  imaging.csv, labs.csv, omics.csv│
                           └──────────────┬───────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
              ┌─────▼─────┐        ┌──────▼──────┐       ┌─────▼──────┐
              │ S3 Sensors│        │ Kinesis Data │       │            │
              │ (Airflow) │        │   Streams    │       │            │
              └─────┬─────┘        └──────┬───────┘       │            │
                    │                     │               │            │
              ┌─────▼─────────────┐ ┌─────▼──────────┐   │            │
              │  AWS Glue Jobs    │ │ stream_vitals  │   │            │
              │  (PySpark ETL)    │ │ (Python consumer)│  │            │
              │                   │ │                │   │            │
              │ • Cleanse         │ │ • 5-min windows│   │            │
              │ • Validate        │ │ • Aggregation  │   │            │
              │ • Enrich          │ │ • Dual-sink    │   │            │
              │ • CDC Merge       │ │                │   │            │
              └─────┬─────────────┘ └──┬─────────┬───┘   │            │
                    │                  │         │       │            │
                    ▼                  ▼         ▼       │            │
         ┌──────────────────────────────────┐ ┌──▼────────▼──┐
         │  Apache Iceberg on S3            │ │ OpenSearch   │
         │  (Glue Data Catalog)             │ │ (Real-time)  │
         │                                  │ └──────────────┘
         │  • dim_patients                  │
         │  • fact_admissions               │
         │  • fact_imaging                  │
         │  • fact_lab_results              │
         │  • fact_omics                    │
         │  • fact_vitals                   │
         │  • patient_360_view              │
         └──────────────┬───────────────────┘
                        │
                  ┌─────▼─────┐
                  │  Amazon   │
                  │  Athena   │
                  │  Queries  │
                  └─────┬─────┘
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
        ┌───────────┐     ┌──────────────┐
        │ QuickSight│     │  SageMaker   │
        │ Dashboard │     │ Risk Scoring │
        └───────────┘     └──────────────┘
```

---

## 🚀 Deployment Guide

### Prerequisites

1. **S3 Bucket**: Create `s3://healthcare-patient360-demo/` with sub-folders:
   ```
   raw/patients/
   raw/admissions/
   raw/imaging/
   raw/labs/
   raw/omics/
   warehouse/
   scripts/
   athena-results/
   spark-logs/
   ```

2. **IAM Role**: `AWSGlueServiceRole-Healthcare360` with permissions for:
   - S3 read/write on the bucket
   - Glue Data Catalog access
   - CloudWatch Logs

3. **Glue Catalog Database**: Created by `create_iceberg_tables.sql`

### Step 1: Create Iceberg Tables

Run the DDL in Athena:
```bash
# Via AWS CLI
aws athena start-query-execution \
  --query-string "$(cat create_iceberg_tables.sql)" \
  --result-configuration OutputLocation=s3://healthcare-patient360-demo/athena-results/ \
  --work-group primary
```

Or paste the contents of `create_iceberg_tables.sql` into the Athena Query Editor.

### Step 2: Upload Transform Scripts to S3

```bash
aws s3 cp transform_patients.py   s3://healthcare-patient360-demo/scripts/
aws s3 cp transform_admissions.py s3://healthcare-patient360-demo/scripts/
aws s3 cp transform_imaging.py    s3://healthcare-patient360-demo/scripts/
aws s3 cp transform_labs.py       s3://healthcare-patient360-demo/scripts/
aws s3 cp transform_omics.py      s3://healthcare-patient360-demo/scripts/
```

### Step 3: Upload Source CSV Data

Place your generated CSV files in the raw zone:
```bash
aws s3 cp patients.csv          s3://healthcare-patient360-demo/raw/patients/
aws s3 cp admissions.csv        s3://healthcare-patient360-demo/raw/admissions/
aws s3 cp imaging_metadata.csv  s3://healthcare-patient360-demo/raw/imaging/
aws s3 cp lab_results.csv       s3://healthcare-patient360-demo/raw/labs/
aws s3 cp omics_data.csv        s3://healthcare-patient360-demo/raw/omics/
```

### Step 4: Run Glue Jobs (Manual / One-Off)

```bash
# Example: Run patients transform
aws glue start-job-run \
  --job-name healthcare_transform_patients \
  --arguments '{
    "--S3_INPUT_PATH": "s3://healthcare-patient360-demo/raw/patients/patients.csv",
    "--S3_WAREHOUSE": "s3://healthcare-patient360-demo/warehouse/",
    "--DATABASE_NAME": "healthcare_db"
  }'
```

### Step 5: Deploy Airflow DAG (Automated Pipeline)

**Option A — Amazon MWAA:**
```bash
aws s3 cp dag_healthcare_pipeline.py s3://your-mwaa-dags-bucket/dags/
```
The DAG will appear in the Airflow UI within 30 seconds.

**Option B — SageMaker Unified Studio (SMUS):**
SMUS Workflows are visual-only and do not support DAG file imports. Recreate the pipeline visually:
1. Open SMUS → Workflows → Create New
2. Add steps matching the DAG tasks:
   - S3 Sensor → Run Glue Job (patients) → Run Glue Job (admissions) → …
3. Wire dependencies to match the topology in the DAG docstring

### Step 6: Start Vitals Stream (Optional)

```bash
# Create the Kinesis stream
aws kinesis create-stream \
  --stream-name healthcare-vitals-stream \
  --shard-count 2

# Start the consumer
python stream_vitals.py \
  --stream-name healthcare-vitals-stream \
  --region us-east-1 \
  --s3-warehouse s3://healthcare-patient360-demo/warehouse/ \
  --database-name healthcare_db

# For dual-sink mode (Iceberg + OpenSearch):
ENABLE_OPENSEARCH=true python stream_vitals.py \
  --stream-name healthcare-vitals-stream \
  --region us-east-1 \
  --s3-warehouse s3://healthcare-patient360-demo/warehouse/ \
  --database-name healthcare_db \
  --opensearch-endpoint https://your-domain.us-east-1.es.amazonaws.com
```

### Step 7: Build Patient 360 View

Run in Athena or via the Airflow DAG:
```bash
aws athena start-query-execution \
  --query-string "$(cat patient_360_view.sql)" \
  --result-configuration OutputLocation=s3://healthcare-patient360-demo/athena-results/
```

---

## 📊 Key Design Patterns

### Petra Ride Pattern — Layered Architecture
- **Raw (Bronze)**: CSV files in S3 `raw/` prefix
- **Cleansed (Silver)**: Validated, standardized data in Iceberg fact tables
- **Curated (Gold)**: `patient_360_view` — unified, business-ready view

### Falcon Eye Pattern — CDC Merge
- All PySpark scripts use `MERGE INTO` for incremental updates
- `record_hash` (MD5) enables change detection — only modified records are updated
- Admission status changes (Admitted → Discharged) flow through cleanly

### Falcon Eye Pattern — Dual-Sink Streaming
- `stream_vitals.py` writes to **both** Iceberg (analytics) and OpenSearch (real-time)
- 5-minute tumbling windows with threshold-based alerting
- Graceful shutdown with window flush

### Partitioning Strategy
| Table | Partition Keys | Rationale |
|-------|---------------|-----------|
| `dim_patients` | None | Small dimension; full scans are fast |
| `fact_admissions` | `department`, `admission_year`, `admission_month` | Department-level queries + time-range pruning |
| `fact_imaging` | `study_year`, `study_month` | Time-series analysis |
| `fact_lab_results` | `test_year`, `test_month` | Time-series trends |
| `fact_omics` | `sample_year`, `sample_month` | Time-partitioned genomic data |
| `fact_vitals` | `window_year`, `window_month` | High-volume streaming data |

---

## 🧪 Validation Queries

After the pipeline runs, verify data quality with these Athena queries:

```sql
-- Record counts per table
SELECT 'dim_patients' AS tbl, COUNT(*) AS cnt FROM healthcare_db.dim_patients
UNION ALL
SELECT 'fact_admissions', COUNT(*) FROM healthcare_db.fact_admissions
UNION ALL
SELECT 'fact_imaging', COUNT(*) FROM healthcare_db.fact_imaging
UNION ALL
SELECT 'fact_lab_results', COUNT(*) FROM healthcare_db.fact_lab_results
UNION ALL
SELECT 'fact_omics', COUNT(*) FROM healthcare_db.fact_omics
UNION ALL
SELECT 'fact_vitals', COUNT(*) FROM healthcare_db.fact_vitals
UNION ALL
SELECT 'patient_360_view', COUNT(*) FROM healthcare_db.patient_360_view;

-- Risk distribution in Patient 360
SELECT composite_risk_level, COUNT(*) AS patients
FROM healthcare_db.patient_360_view
GROUP BY composite_risk_level
ORDER BY CASE composite_risk_level
    WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2
    WHEN 'MEDIUM' THEN 3 ELSE 4 END;

-- Top 10 highest-risk patients
SELECT patient_id, full_name, age, composite_risk_level,
       numeric_risk_score, total_admissions, critical_lab_count,
       pathogenic_variants, active_vital_alert
FROM healthcare_db.patient_360_view
ORDER BY numeric_risk_score DESC
LIMIT 10;
```

---

## 📝 Notes

- **Iceberg format**: All tables use Parquet with ZSTD compression (except vitals which uses Snappy for faster writes)
- **Glue version**: Scripts target Glue 4.0 with Iceberg connector (built-in via `--datalake-formats iceberg`)
- **Idempotency**: All transforms use MERGE for safe re-runs; the 360 view uses DROP + CTAS
- **Error handling**: Each script logs warnings for non-critical issues (missing FK references, etc.) and continues processing
- **Compaction**: Patient transform includes `rewrite_data_files` call for Iceberg small-file compaction
