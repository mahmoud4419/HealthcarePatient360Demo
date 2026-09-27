"""
===============================================================================
dag_healthcare_pipeline.py — Healthcare Patient 360 Lakehouse Demo
===============================================================================
Apache Airflow DAG for orchestrating the end-to-end Healthcare Patient 360
data pipeline.

*** SMUS (SageMaker Unified Studio) COMPATIBILITY NOTE ***
SMUS Workflows are visual-only and do NOT support importing DAG files directly.
You have two options:
  (a) Deploy this DAG to a standalone Airflow or Amazon MWAA (Managed Workflows
      for Apache Airflow) environment by placing it in the DAGs S3 bucket.
  (b) Recreate the pipeline visually in SMUS Workflows by chaining notebook
      steps that call the same transform scripts. Each GlueJobOperator below
      maps to a "Run Glue Job" step in SMUS; each sensor maps to a "Wait for
      condition" step.

Pipeline topology:
  ┌─────────────────────────────────────────────────────────────┐
  │  S3 Sensor (wait for files)                                 │
  │       ↓                                                     │
  │  ingest_patients (dim_patients)                             │
  │       ↓                                                     │
  │  ingest_admissions (fact_admissions)                        │
  │       ↓                                                     │
  │  ┌──────────────┬──────────────┬──────────────┐            │
  │  │ ingest_imaging│ ingest_labs  │ ingest_omics │  (parallel)│
  │  └──────┬───────┴──────┬───────┴──────┬───────┘            │
  │         └──────────────┼──────────────┘                     │
  │                        ↓                                    │
  │             build_patient_360_view                          │
  │                        ↓                                    │
  │             run_risk_scoring_model                          │
  │                        ↓                                    │
  │              refresh_dashboard                              │
  └─────────────────────────────────────────────────────────────┘

Architecture: Airflow/MWAA → Glue Jobs (PySpark) → Iceberg → Athena
Modeled after: Petra Ride orchestration + Falcon Eye dependency chains
===============================================================================
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from airflow.providers.amazon.aws.operators.athena import AthenaOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerProcessingOperator
from airflow.utils.task_group import TaskGroup


# ===========================================================================
# CONFIGURATION
# ===========================================================================
# Override these via Airflow Variables or environment for your deployment
S3_BUCKET = "healthcare-patient360-demo"
S3_RAW_PREFIX = "raw"
S3_WAREHOUSE = f"s3://{S3_BUCKET}/warehouse/"
DATABASE_NAME = "healthcare_db"
GLUE_IAM_ROLE = "AWSGlueServiceRole-Healthcare360"
AWS_CONN_ID = "aws_default"
ATHENA_OUTPUT = f"s3://{S3_BUCKET}/athena-results/"
SAGEMAKER_ROLE = "arn:aws:iam::role/SageMakerExecutionRole-Healthcare360"

# Glue job defaults
GLUE_DEFAULTS = {
    "GlueVersion": "4.0",
    "WorkerType": "G.1X",
    "NumberOfWorkers": 4,
    "Timeout": 60,  # minutes
    "DefaultArguments": {
        "--S3_WAREHOUSE": S3_WAREHOUSE,
        "--DATABASE_NAME": DATABASE_NAME,
        "--enable-metrics": "true",
        "--enable-continuous-cloudwatch-log": "true",
        "--enable-spark-ui": "true",
        "--spark-event-logs-path": f"s3://{S3_BUCKET}/spark-logs/",
        "--conf": "spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        "--datalake-formats": "iceberg",
    },
}


# ===========================================================================
# DAG DEFINITION
# ===========================================================================
default_args = {
    "owner": "healthcare-data-team",
    "depends_on_past": False,
    "email": ["data-team@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

with DAG(
    dag_id="healthcare_patient_360_pipeline",
    default_args=default_args,
    description=(
        "End-to-end Patient 360 lakehouse pipeline: "
        "S3 CSV → Glue/Spark → Iceberg → Athena → Dashboard"
    ),
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["healthcare", "patient-360", "lakehouse", "iceberg"],
    doc_md=__doc__,
) as dag:

    # -----------------------------------------------------------------------
    # PHASE 0: S3 FILE ARRIVAL SENSORS
    # -----------------------------------------------------------------------
    # Wait for today's data files to land in S3 before triggering pipeline.
    # Follows Petra Ride pattern: gate pipeline start on data availability.
    # -----------------------------------------------------------------------
    with TaskGroup("wait_for_source_files") as sensor_group:

        sensor_patients = S3KeySensor(
            task_id="sense_patients_csv",
            bucket_name=S3_BUCKET,
            bucket_key=f"{S3_RAW_PREFIX}/patients/patients.csv",
            aws_conn_id=AWS_CONN_ID,
            poke_interval=60,        # check every 60s
            timeout=3600,            # wait up to 1 hour
            mode="poke",
        )

        sensor_admissions = S3KeySensor(
            task_id="sense_admissions_csv",
            bucket_name=S3_BUCKET,
            bucket_key=f"{S3_RAW_PREFIX}/admissions/admissions.csv",
            aws_conn_id=AWS_CONN_ID,
            poke_interval=60,
            timeout=3600,
            mode="poke",
        )

        sensor_imaging = S3KeySensor(
            task_id="sense_imaging_csv",
            bucket_name=S3_BUCKET,
            bucket_key=f"{S3_RAW_PREFIX}/imaging/imaging_metadata.csv",
            aws_conn_id=AWS_CONN_ID,
            poke_interval=60,
            timeout=3600,
            mode="reschedule",  # reschedule mode frees the worker slot
        )

        sensor_labs = S3KeySensor(
            task_id="sense_labs_csv",
            bucket_name=S3_BUCKET,
            bucket_key=f"{S3_RAW_PREFIX}/labs/lab_results.csv",
            aws_conn_id=AWS_CONN_ID,
            poke_interval=60,
            timeout=3600,
            mode="reschedule",
        )

        sensor_omics = S3KeySensor(
            task_id="sense_omics_csv",
            bucket_name=S3_BUCKET,
            bucket_key=f"{S3_RAW_PREFIX}/omics/omics_data.csv",
            aws_conn_id=AWS_CONN_ID,
            poke_interval=60,
            timeout=3600,
            mode="reschedule",
        )

    # -----------------------------------------------------------------------
    # PHASE 1: DIMENSION TABLE — Patients (must complete first)
    # -----------------------------------------------------------------------
    ingest_patients = GlueJobOperator(
        task_id="ingest_patients",
        job_name="healthcare_transform_patients",
        script_location=f"s3://{S3_BUCKET}/scripts/transform_patients.py",
        iam_role_name=GLUE_IAM_ROLE,
        create_job_kwargs={
            **GLUE_DEFAULTS,
            "DefaultArguments": {
                **GLUE_DEFAULTS["DefaultArguments"],
                "--S3_INPUT_PATH": f"s3://{S3_BUCKET}/{S3_RAW_PREFIX}/patients/patients.csv",
            },
        },
        script_args={
            "--S3_INPUT_PATH": f"s3://{S3_BUCKET}/{S3_RAW_PREFIX}/patients/patients.csv",
        },
        aws_conn_id=AWS_CONN_ID,
        region_name="us-east-1",
        wait_for_completion=True,
        verbose=True,
    )

    # -----------------------------------------------------------------------
    # PHASE 2: ADMISSIONS — depends on patients for FK validation
    # -----------------------------------------------------------------------
    ingest_admissions = GlueJobOperator(
        task_id="ingest_admissions",
        job_name="healthcare_transform_admissions",
        script_location=f"s3://{S3_BUCKET}/scripts/transform_admissions.py",
        iam_role_name=GLUE_IAM_ROLE,
        create_job_kwargs={
            **GLUE_DEFAULTS,
            "DefaultArguments": {
                **GLUE_DEFAULTS["DefaultArguments"],
                "--S3_INPUT_PATH": f"s3://{S3_BUCKET}/{S3_RAW_PREFIX}/admissions/admissions.csv",
            },
        },
        script_args={
            "--S3_INPUT_PATH": f"s3://{S3_BUCKET}/{S3_RAW_PREFIX}/admissions/admissions.csv",
        },
        aws_conn_id=AWS_CONN_ID,
        region_name="us-east-1",
        wait_for_completion=True,
        verbose=True,
    )

    # -----------------------------------------------------------------------
    # PHASE 3: PARALLEL FACT TABLE LOADS (imaging, labs, omics)
    # -----------------------------------------------------------------------
    with TaskGroup("parallel_fact_loads") as parallel_facts:

        ingest_imaging = GlueJobOperator(
            task_id="ingest_imaging",
            job_name="healthcare_transform_imaging",
            script_location=f"s3://{S3_BUCKET}/scripts/transform_imaging.py",
            iam_role_name=GLUE_IAM_ROLE,
            create_job_kwargs={
                **GLUE_DEFAULTS,
                "DefaultArguments": {
                    **GLUE_DEFAULTS["DefaultArguments"],
                    "--S3_INPUT_PATH": f"s3://{S3_BUCKET}/{S3_RAW_PREFIX}/imaging/imaging_metadata.csv",
                },
            },
            script_args={
                "--S3_INPUT_PATH": f"s3://{S3_BUCKET}/{S3_RAW_PREFIX}/imaging/imaging_metadata.csv",
            },
            aws_conn_id=AWS_CONN_ID,
            region_name="us-east-1",
            wait_for_completion=True,
        )

        ingest_labs = GlueJobOperator(
            task_id="ingest_labs",
            job_name="healthcare_transform_labs",
            script_location=f"s3://{S3_BUCKET}/scripts/transform_labs.py",
            iam_role_name=GLUE_IAM_ROLE,
            create_job_kwargs={
                **GLUE_DEFAULTS,
                "DefaultArguments": {
                    **GLUE_DEFAULTS["DefaultArguments"],
                    "--S3_INPUT_PATH": f"s3://{S3_BUCKET}/{S3_RAW_PREFIX}/labs/lab_results.csv",
                },
            },
            script_args={
                "--S3_INPUT_PATH": f"s3://{S3_BUCKET}/{S3_RAW_PREFIX}/labs/lab_results.csv",
            },
            aws_conn_id=AWS_CONN_ID,
            region_name="us-east-1",
            wait_for_completion=True,
        )

        ingest_omics = GlueJobOperator(
            task_id="ingest_omics",
            job_name="healthcare_transform_omics",
            script_location=f"s3://{S3_BUCKET}/scripts/transform_omics.py",
            iam_role_name=GLUE_IAM_ROLE,
            create_job_kwargs={
                **GLUE_DEFAULTS,
                "DefaultArguments": {
                    **GLUE_DEFAULTS["DefaultArguments"],
                    "--S3_INPUT_PATH": f"s3://{S3_BUCKET}/{S3_RAW_PREFIX}/omics/omics_data.csv",
                },
            },
            script_args={
                "--S3_INPUT_PATH": f"s3://{S3_BUCKET}/{S3_RAW_PREFIX}/omics/omics_data.csv",
            },
            aws_conn_id=AWS_CONN_ID,
            region_name="us-east-1",
            wait_for_completion=True,
        )

    # -----------------------------------------------------------------------
    # PHASE 4: BUILD PATIENT 360 VIEW (CTAS / materialized view)
    # -----------------------------------------------------------------------
    build_patient_360_view = AthenaOperator(
        task_id="build_patient_360_view",
        query="""
            -- Drop and recreate the Patient 360 materialized view (CTAS)
            -- See patient_360_view.sql for the full query
            DROP TABLE IF EXISTS healthcare_db.patient_360_view;

            CREATE TABLE healthcare_db.patient_360_view
            WITH (
                table_type = 'ICEBERG',
                location = 's3://{{ params.s3_bucket }}/warehouse/healthcare_db.db/patient_360_view/',
                is_external = false,
                format = 'PARQUET'
            ) AS
            SELECT
                p.patient_id,
                p.full_name,
                p.age,
                p.age_group,
                p.gender,
                p.chronic_conditions,
                p.insurance_provider,
                -- Admission summary
                a.total_admissions,
                a.current_status,
                a.last_admission_date,
                a.avg_length_of_stay,
                a.max_severity,
                -- Lab summary
                l.total_lab_tests,
                l.abnormal_lab_count,
                l.critical_lab_count,
                l.last_lab_date,
                -- Imaging summary
                i.total_imaging_studies,
                i.abnormal_imaging_count,
                i.cumulative_radiation,
                -- Omics summary
                o.pathogenic_variants,
                o.pharmacogenomic_flags,
                -- Risk composite
                CASE
                    WHEN COALESCE(a.max_severity, 'LOW') = 'CRITICAL'
                         OR COALESCE(l.critical_lab_count, 0) > 2
                         OR COALESCE(o.pathogenic_variants, 0) > 3 THEN 'HIGH'
                    WHEN COALESCE(a.max_severity, 'LOW') IN ('HIGH', 'CRITICAL')
                         OR COALESCE(l.abnormal_lab_count, 0) > 5 THEN 'MEDIUM'
                    ELSE 'LOW'
                END AS composite_risk_level,
                current_timestamp AS view_refreshed_at
            FROM healthcare_db.dim_patients p
            LEFT JOIN (
                SELECT patient_id,
                       COUNT(*) AS total_admissions,
                       MAX(status) AS current_status,
                       MAX(admission_date) AS last_admission_date,
                       ROUND(AVG(length_of_stay), 1) AS avg_length_of_stay,
                       MAX(severity_category) AS max_severity
                FROM healthcare_db.fact_admissions
                GROUP BY patient_id
            ) a ON p.patient_id = a.patient_id
            LEFT JOIN (
                SELECT patient_id,
                       COUNT(*) AS total_lab_tests,
                       COUNT(CASE WHEN abnormal_flag != 'NORMAL' THEN 1 END) AS abnormal_lab_count,
                       COUNT(CASE WHEN critical_flag = 'CRITICAL' THEN 1 END) AS critical_lab_count,
                       MAX(test_date) AS last_lab_date
                FROM healthcare_db.fact_lab_results
                GROUP BY patient_id
            ) l ON p.patient_id = l.patient_id
            LEFT JOIN (
                SELECT patient_id,
                       COUNT(*) AS total_imaging_studies,
                       COUNT(CASE WHEN abnormal_flag = 'Y' THEN 1 END) AS abnormal_imaging_count,
                       SUM(radiation_dose_mgy) AS cumulative_radiation
                FROM healthcare_db.fact_imaging
                GROUP BY patient_id
            ) i ON p.patient_id = i.patient_id
            LEFT JOIN (
                SELECT patient_id,
                       COUNT(CASE WHEN clinical_significance_std = 'PATHOGENIC/LIKELY_PATHOGENIC' THEN 1 END)
                           AS pathogenic_variants,
                       COUNT(CASE WHEN is_pharmacogenomic = true THEN 1 END)
                           AS pharmacogenomic_flags
                FROM healthcare_db.fact_omics
                GROUP BY patient_id
            ) o ON p.patient_id = o.patient_id
        """,
        database=DATABASE_NAME,
        output_location=ATHENA_OUTPUT,
        aws_conn_id=AWS_CONN_ID,
        params={"s3_bucket": S3_BUCKET},
    )

    # -----------------------------------------------------------------------
    # PHASE 5: RISK SCORING MODEL (SageMaker Processing Job)
    # -----------------------------------------------------------------------
    # Calls a SageMaker Processing job that reads from patient_360_view,
    # runs a risk scoring model, and writes risk scores back.
    # For demo simplicity, this is a PythonOperator placeholder.
    def _run_risk_scoring(**kwargs):
        """
        Placeholder for SageMaker risk scoring model invocation.
        In production, replace with SageMakerProcessingOperator or
        SageMakerEndpointOperator for real-time inference.
        """
        import logging
        log = logging.getLogger("risk_scoring")
        log.info("Risk scoring model triggered.")
        log.info("Reading from healthcare_db.patient_360_view ...")
        log.info("Model: XGBoost readmission risk + clinical deterioration")
        log.info("Output: healthcare_db.patient_risk_scores Iceberg table")
        # In production:
        # 1. Read patient_360_view via Athena/Spark
        # 2. Call SageMaker endpoint for batch inference
        # 3. Write scored results to patient_risk_scores table
        log.info("Risk scoring completed (placeholder).")

    run_risk_scoring_model = PythonOperator(
        task_id="run_risk_scoring_model",
        python_callable=_run_risk_scoring,
    )

    # -----------------------------------------------------------------------
    # PHASE 6: REFRESH DASHBOARD
    # -----------------------------------------------------------------------
    def _refresh_dashboard(**kwargs):
        """
        Placeholder for QuickSight / Grafana / OpenSearch dashboard refresh.
        Options:
          - QuickSight: Trigger SPICE dataset refresh via boto3
          - Grafana: Dashboards query Athena live (no explicit refresh needed)
          - OpenSearch: Vitals are already streamed via dual-sink
        """
        import logging
        import boto3
        log = logging.getLogger("refresh_dashboard")

        # Example: QuickSight SPICE refresh
        # qs = boto3.client("quicksight")
        # qs.create_ingestion(
        #     DataSetId="patient-360-dataset",
        #     IngestionId=f"refresh-{kwargs['ds']}",
        #     AwsAccountId="123456789012",
        # )

        log.info(f"Dashboard refresh triggered for execution date: {kwargs['ds']}")
        log.info("QuickSight SPICE dataset refresh initiated (placeholder).")

    refresh_dashboard = PythonOperator(
        task_id="refresh_dashboard",
        python_callable=_refresh_dashboard,
    )

    # -----------------------------------------------------------------------
    # PIPELINE COMPLETION MARKER
    # -----------------------------------------------------------------------
    pipeline_complete = DummyOperator(
        task_id="pipeline_complete",
        trigger_rule="all_success",
    )

    # -----------------------------------------------------------------------
    # DAG DEPENDENCY WIRING
    # -----------------------------------------------------------------------
    # Phase 0: Wait for files → Phase 1: Patients
    sensor_group >> ingest_patients

    # Phase 1 → Phase 2: Admissions (needs dim_patients for FK join)
    ingest_patients >> ingest_admissions

    # Phase 2 → Phase 3: Parallel fact loads
    ingest_admissions >> parallel_facts

    # Phase 3 → Phase 4: Build unified Patient 360 view
    parallel_facts >> build_patient_360_view

    # Phase 4 → Phase 5: Risk scoring
    build_patient_360_view >> run_risk_scoring_model

    # Phase 5 → Phase 6: Dashboard refresh → Done
    run_risk_scoring_model >> refresh_dashboard >> pipeline_complete
