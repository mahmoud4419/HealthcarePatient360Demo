"""
===============================================================================
transform_admissions.py — Healthcare Patient 360 Lakehouse Demo
===============================================================================
PySpark / AWS Glue script that reads admissions.csv from S3, joins with
dim_patients for referential integrity, calculates length_of_stay and
severity_category, and writes to `healthcare_db.fact_admissions` Iceberg table.

Includes CDC MERGE logic modeled after the Gulf Air Falcon Eye pattern:
  - Uses admission_id as the merge key
  - Updates status and discharge_date for existing records (late-arriving data)
  - Inserts new admission records

Architecture: S3 (raw CSV) → Glue/Spark → Iceberg → Athena
===============================================================================
"""

import sys
import logging

from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, trim, upper, when, lit, current_timestamp, to_date, to_timestamp,
    datediff, year, month, dayofmonth, coalesce, round as spark_round,
    broadcast, expr, concat_ws, md5
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, DateType
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("transform_admissions")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
))
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Glue boilerplate
# ---------------------------------------------------------------------------
args = getResolvedOptions(sys.argv, [
    "JOB_NAME",
    "S3_INPUT_PATH",
    "S3_WAREHOUSE",
    "DATABASE_NAME",
])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# Iceberg catalog configuration
spark.conf.set("spark.sql.catalog.glue_catalog", "org.apache.iceberg.spark.SparkCatalog")
spark.conf.set("spark.sql.catalog.glue_catalog.warehouse", args["S3_WAREHOUSE"])
spark.conf.set("spark.sql.catalog.glue_catalog.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog")
spark.conf.set("spark.sql.catalog.glue_catalog.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
spark.conf.set("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")

S3_INPUT_PATH = args["S3_INPUT_PATH"]
DATABASE_NAME = args["DATABASE_NAME"]
FACT_TABLE = f"glue_catalog.{DATABASE_NAME}.fact_admissions"
DIM_PATIENTS_TABLE = f"glue_catalog.{DATABASE_NAME}.dim_patients"

logger.info(f"Starting admissions transform from {S3_INPUT_PATH}")

# ---------------------------------------------------------------------------
# 1. INGEST — Read raw CSV
# ---------------------------------------------------------------------------
ADMISSIONS_SCHEMA = StructType([
    StructField("admission_id", StringType(), False),
    StructField("patient_id", StringType(), False),
    StructField("admission_date", StringType(), True),
    StructField("discharge_date", StringType(), True),
    StructField("department", StringType(), True),
    StructField("diagnosis_code", StringType(), True),
    StructField("diagnosis_description", StringType(), True),
    StructField("attending_physician", StringType(), True),
    StructField("room_number", StringType(), True),
    StructField("bed_number", StringType(), True),
    StructField("admission_type", StringType(), True),       # Emergency / Elective / Urgent
    StructField("status", StringType(), True),               # Admitted / Discharged / Transferred
    StructField("severity_score", DoubleType(), True),       # 0.0 – 10.0
    StructField("total_charges", DoubleType(), True),
    StructField("insurance_covered", DoubleType(), True),
])

try:
    raw_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .schema(ADMISSIONS_SCHEMA)
        .csv(S3_INPUT_PATH)
    )
    raw_count = raw_df.count()
    logger.info(f"Raw admission records: {raw_count}")
except Exception as e:
    logger.error(f"Failed to read admissions CSV: {e}")
    raise

# ---------------------------------------------------------------------------
# 2. VALIDATION — ensure required fields present
# ---------------------------------------------------------------------------
valid_df = raw_df.filter(
    col("admission_id").isNotNull() &
    col("patient_id").isNotNull() &
    col("admission_date").isNotNull()
)
rejected = raw_count - valid_df.count()
if rejected > 0:
    logger.warning(f"Rejected {rejected} rows with null keys")

# ---------------------------------------------------------------------------
# 3. CLEANSE AND ENRICH
# ---------------------------------------------------------------------------

# Parse dates
enriched_df = valid_df.withColumn(
    "admission_date", to_date(col("admission_date"), "yyyy-MM-dd")
).withColumn(
    "discharge_date", to_date(col("discharge_date"), "yyyy-MM-dd")
)

# Standardize text fields
enriched_df = enriched_df.withColumn(
    "department", upper(trim(col("department")))
).withColumn(
    "admission_type", upper(trim(col("admission_type")))
).withColumn(
    "status", upper(trim(col("status")))
)

# Calculate Length of Stay (LOS)
# If still admitted (no discharge_date), LOS = days since admission to now
enriched_df = enriched_df.withColumn(
    "length_of_stay",
    when(
        col("discharge_date").isNotNull(),
        datediff(col("discharge_date"), col("admission_date"))
    ).otherwise(
        datediff(current_timestamp(), col("admission_date"))
    )
).withColumn(
    # Guard against negative LOS from bad data
    "length_of_stay",
    when(col("length_of_stay") < 0, lit(0)).otherwise(col("length_of_stay"))
)

# Severity category derived from severity_score
enriched_df = enriched_df.withColumn(
    "severity_category",
    when(col("severity_score") <= 2.5, "LOW")
    .when(col("severity_score") <= 5.0, "MEDIUM")
    .when(col("severity_score") <= 7.5, "HIGH")
    .when(col("severity_score") > 7.5, "CRITICAL")
    .otherwise("UNKNOWN")
)

# Out-of-pocket cost
enriched_df = enriched_df.withColumn(
    "out_of_pocket",
    spark_round(
        coalesce(col("total_charges"), lit(0)) - coalesce(col("insurance_covered"), lit(0)),
        2
    )
)

# Partitioning columns
enriched_df = enriched_df.withColumn(
    "admission_year", year(col("admission_date"))
).withColumn(
    "admission_month", month(col("admission_date"))
)

# ---------------------------------------------------------------------------
# 4. REFERENTIAL INTEGRITY — Join with dim_patients
# ---------------------------------------------------------------------------
try:
    patients_df = spark.sql(f"""
        SELECT patient_id, full_name, age_group, insurance_provider
        FROM {DIM_PATIENTS_TABLE}
    """)

    # Left join — keep admissions even if patient not yet in dim (orphan tracking)
    enriched_df = enriched_df.join(
        broadcast(patients_df),
        on="patient_id",
        how="left"
    ).withColumn(
        "patient_found_in_dim",
        when(patients_df["full_name"].isNotNull(), lit(True)).otherwise(lit(False))
    )

    orphan_count = enriched_df.filter(~col("patient_found_in_dim")).count()
    if orphan_count > 0:
        logger.warning(f"Found {orphan_count} admissions with no matching patient in dim_patients")

except Exception as e:
    logger.warning(f"Could not join with dim_patients ({e}). Proceeding without patient enrichment.")
    enriched_df = enriched_df.withColumn("patient_found_in_dim", lit(False))
    enriched_df = enriched_df.withColumn("full_name", lit(None).cast(StringType()))
    enriched_df = enriched_df.withColumn("age_group", lit(None).cast(StringType()))
    enriched_df = enriched_df.withColumn("insurance_provider", lit(None).cast(StringType()))

# Audit columns
final_df = enriched_df.withColumn(
    "record_hash", md5(concat_ws("||",
        col("admission_id"), col("patient_id"),
        col("admission_date").cast(StringType()),
        col("discharge_date").cast(StringType()),
        col("status"), col("severity_score").cast(StringType()),
        col("total_charges").cast(StringType()),
    ))
).withColumn(
    "etl_loaded_at", current_timestamp()
).withColumn(
    "etl_source", lit("admissions.csv")
)

logger.info(f"Enriched admission records: {final_df.count()}")

# ---------------------------------------------------------------------------
# 5. CDC MERGE — Falcon Eye pattern (upsert on admission_id)
# ---------------------------------------------------------------------------
# The MERGE handles:
#   - New admissions → INSERT
#   - Status changes (e.g. Admitted→Discharged) → UPDATE status, discharge_date, LOS
#   - Charge corrections → UPDATE charges
# ---------------------------------------------------------------------------
final_df.createOrReplaceTempView("incoming_admissions")

try:
    merge_sql = f"""
    MERGE INTO {FACT_TABLE} AS target
    USING incoming_admissions AS source
    ON target.admission_id = source.admission_id

    WHEN MATCHED AND target.record_hash != source.record_hash THEN UPDATE SET
        target.patient_id             = source.patient_id,
        target.admission_date         = source.admission_date,
        target.discharge_date         = source.discharge_date,
        target.department             = source.department,
        target.diagnosis_code         = source.diagnosis_code,
        target.diagnosis_description  = source.diagnosis_description,
        target.attending_physician    = source.attending_physician,
        target.room_number            = source.room_number,
        target.bed_number             = source.bed_number,
        target.admission_type         = source.admission_type,
        target.status                 = source.status,
        target.severity_score         = source.severity_score,
        target.severity_category      = source.severity_category,
        target.total_charges          = source.total_charges,
        target.insurance_covered      = source.insurance_covered,
        target.out_of_pocket          = source.out_of_pocket,
        target.length_of_stay         = source.length_of_stay,
        target.full_name              = source.full_name,
        target.age_group              = source.age_group,
        target.insurance_provider     = source.insurance_provider,
        target.patient_found_in_dim   = source.patient_found_in_dim,
        target.admission_year         = source.admission_year,
        target.admission_month        = source.admission_month,
        target.record_hash            = source.record_hash,
        target.etl_loaded_at          = source.etl_loaded_at,
        target.etl_source             = source.etl_source

    WHEN NOT MATCHED THEN INSERT *
    """
    spark.sql(merge_sql)
    logger.info("MERGE INTO fact_admissions completed successfully (Falcon Eye CDC pattern).")

except Exception as e:
    logger.warning(f"MERGE failed ({e}), falling back to overwrite.")
    final_df.writeTo(FACT_TABLE).using("iceberg").createOrReplace()
    logger.info("Overwrite write to fact_admissions completed.")

# ---------------------------------------------------------------------------
# 6. POST-WRITE STATS
# ---------------------------------------------------------------------------
stats = spark.sql(f"""
    SELECT
        COUNT(*)                                           AS total_records,
        COUNT(DISTINCT patient_id)                         AS unique_patients,
        COUNT(CASE WHEN status = 'ADMITTED' THEN 1 END)   AS currently_admitted,
        ROUND(AVG(length_of_stay), 1)                      AS avg_los_days,
        COUNT(CASE WHEN severity_category = 'CRITICAL' THEN 1 END) AS critical_count
    FROM {FACT_TABLE}
""").collect()[0]

logger.info(
    f"fact_admissions stats: "
    f"total={stats['total_records']}, "
    f"patients={stats['unique_patients']}, "
    f"admitted={stats['currently_admitted']}, "
    f"avg_los={stats['avg_los_days']}d, "
    f"critical={stats['critical_count']}"
)

# Iceberg table maintenance
try:
    spark.sql(f"CALL glue_catalog.system.rewrite_data_files(table => '{DATABASE_NAME}.fact_admissions')")
    spark.sql(f"CALL glue_catalog.system.expire_snapshots(table => '{DATABASE_NAME}.fact_admissions', older_than => TIMESTAMP '{{}}')"
              .format((datetime.now()).strftime('%Y-%m-%d %H:%M:%S')))
    logger.info("Iceberg maintenance completed.")
except Exception as e:
    logger.warning(f"Iceberg maintenance skipped: {e}")

job.commit()
logger.info("transform_admissions job finished successfully.")
