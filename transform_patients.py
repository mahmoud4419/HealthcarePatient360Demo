"""
===============================================================================
transform_patients.py — Healthcare Patient 360 Lakehouse Demo
===============================================================================
PySpark / AWS Glue script that reads patients.csv from S3, applies schema
validation, data cleansing, deduplication, and writes to the Iceberg table
`healthcare_db.dim_patients`.

Architecture pattern: S3 (raw CSV) → Glue/Spark → Iceberg → Athena
Modeled after: Petra Ride bronze→silver→gold layering and Gulf Air Falcon Eye
               incremental-load patterns.

Usage (Glue job parameters):
  --S3_INPUT_PATH    s3://your-bucket/raw/patients/patients.csv
  --S3_WAREHOUSE     s3://your-bucket/warehouse/
  --DATABASE_NAME    healthcare_db
===============================================================================
"""

import sys
import re
import logging
from datetime import datetime

from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame

from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, trim, initcap, lower, upper, regexp_replace, when, lit, current_timestamp,
    to_date, datediff, year, month, dayofmonth, split, array_distinct,
    row_number, md5, concat_ws, coalesce, expr
)
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DateType,
    DoubleType, ArrayType, TimestampType
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("transform_patients")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
))
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Glue job boilerplate
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

# Configure Spark for Iceberg
spark.conf.set("spark.sql.catalog.glue_catalog", "org.apache.iceberg.spark.SparkCatalog")
spark.conf.set("spark.sql.catalog.glue_catalog.warehouse", args["S3_WAREHOUSE"])
spark.conf.set("spark.sql.catalog.glue_catalog.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog")
spark.conf.set("spark.sql.catalog.glue_catalog.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
spark.conf.set("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")

S3_INPUT_PATH = args["S3_INPUT_PATH"]
DATABASE_NAME = args["DATABASE_NAME"]
TABLE_NAME = f"glue_catalog.{DATABASE_NAME}.dim_patients"

logger.info(f"Starting patient dimension transform from {S3_INPUT_PATH}")

# ---------------------------------------------------------------------------
# 1. INGEST — Read raw CSV
# ---------------------------------------------------------------------------
EXPECTED_SCHEMA = StructType([
    StructField("patient_id", StringType(), False),
    StructField("first_name", StringType(), True),
    StructField("last_name", StringType(), True),
    StructField("date_of_birth", StringType(), True),
    StructField("gender", StringType(), True),
    StructField("blood_type", StringType(), True),
    StructField("email", StringType(), True),
    StructField("phone", StringType(), True),
    StructField("address", StringType(), True),
    StructField("city", StringType(), True),
    StructField("state", StringType(), True),
    StructField("zip_code", StringType(), True),
    StructField("insurance_provider", StringType(), True),
    StructField("insurance_id", StringType(), True),
    StructField("primary_physician", StringType(), True),
    StructField("chronic_conditions", StringType(), True),   # pipe-delimited
    StructField("registration_date", StringType(), True),
])

try:
    raw_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .schema(EXPECTED_SCHEMA)
        .csv(S3_INPUT_PATH)
    )
    raw_count = raw_df.count()
    logger.info(f"Raw records read: {raw_count}")
except Exception as e:
    logger.error(f"Failed to read source CSV: {e}")
    raise

# ---------------------------------------------------------------------------
# 2. SCHEMA VALIDATION — reject rows with null keys or corrupt data
# ---------------------------------------------------------------------------
valid_df = raw_df.filter(
    col("patient_id").isNotNull() &
    (col("patient_id") != "") &
    col("first_name").isNotNull() &
    col("last_name").isNotNull()
)

rejected_count = raw_count - valid_df.count()
if rejected_count > 0:
    logger.warning(f"Rejected {rejected_count} rows with missing required fields")

# ---------------------------------------------------------------------------
# 3. DATA CLEANSING
# ---------------------------------------------------------------------------

# 3a. Standardize names — trim whitespace, apply InitCap
cleansed_df = valid_df.select(
    trim(col("patient_id")).alias("patient_id"),
    initcap(trim(col("first_name"))).alias("first_name"),
    initcap(trim(col("last_name"))).alias("last_name"),
    col("date_of_birth"),
    upper(trim(col("gender"))).alias("gender"),
    upper(trim(col("blood_type"))).alias("blood_type"),
    lower(trim(col("email"))).alias("email"),
    regexp_replace(trim(col("phone")), r"[^0-9+]", "").alias("phone"),
    trim(col("address")).alias("address"),
    initcap(trim(col("city"))).alias("city"),
    upper(trim(col("state"))).alias("state"),
    trim(col("zip_code")).alias("zip_code"),
    trim(col("insurance_provider")).alias("insurance_provider"),
    trim(col("insurance_id")).alias("insurance_id"),
    initcap(trim(col("primary_physician"))).alias("primary_physician"),
    col("chronic_conditions"),
    col("registration_date"),
)

# 3b. Validate and parse dates
cleansed_df = cleansed_df.withColumn(
    "date_of_birth",
    to_date(col("date_of_birth"), "yyyy-MM-dd")
).withColumn(
    "registration_date",
    to_date(col("registration_date"), "yyyy-MM-dd")
)

# Filter out obviously bad DOBs (future dates or > 120 years old)
cleansed_df = cleansed_df.filter(
    (col("date_of_birth").isNotNull()) &
    (col("date_of_birth") < current_timestamp()) &
    (datediff(current_timestamp(), col("date_of_birth")) < 120 * 365)
)

# 3c. Standardize gender
cleansed_df = cleansed_df.withColumn(
    "gender",
    when(col("gender").isin("M", "MALE"), "M")
    .when(col("gender").isin("F", "FEMALE"), "F")
    .when(col("gender").isin("NB", "NON-BINARY", "NONBINARY"), "NB")
    .otherwise("U")  # Unknown
)

# 3d. Validate blood type
VALID_BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
cleansed_df = cleansed_df.withColumn(
    "blood_type",
    when(col("blood_type").isin(*VALID_BLOOD_TYPES), col("blood_type"))
    .otherwise(lit(None))
)

# 3e. Parse chronic_conditions from pipe-delimited string → array
cleansed_df = cleansed_df.withColumn(
    "chronic_conditions_array",
    when(
        col("chronic_conditions").isNotNull() & (col("chronic_conditions") != ""),
        array_distinct(split(col("chronic_conditions"), r"\|"))
    ).otherwise(lit(None).cast(ArrayType(StringType())))
).drop("chronic_conditions").withColumnRenamed(
    "chronic_conditions_array", "chronic_conditions"
)

# 3f. Compute derived fields
cleansed_df = cleansed_df.withColumn(
    "age",
    (datediff(current_timestamp(), col("date_of_birth")) / 365.25).cast(IntegerType())
).withColumn(
    "age_group",
    when(col("age") < 18, "Pediatric")
    .when(col("age") < 40, "Young Adult")
    .when(col("age") < 65, "Adult")
    .otherwise("Senior")
).withColumn(
    "full_name",
    concat_ws(" ", col("first_name"), col("last_name"))
)

# ---------------------------------------------------------------------------
# 4. DEDUPLICATION — keep most recently registered record per patient_id
# ---------------------------------------------------------------------------
dedup_window = Window.partitionBy("patient_id").orderBy(
    col("registration_date").desc_nulls_last()
)

deduped_df = (
    cleansed_df
    .withColumn("_row_num", row_number().over(dedup_window))
    .filter(col("_row_num") == 1)
    .drop("_row_num")
)

# Add audit / SCD-2 metadata columns
final_df = deduped_df.withColumn(
    "record_hash", md5(concat_ws("||",
        col("first_name"), col("last_name"), col("date_of_birth").cast(StringType()),
        col("gender"), col("blood_type"), col("email"), col("phone"),
        col("address"), col("city"), col("state"), col("zip_code"),
        col("insurance_provider"), col("insurance_id"), col("primary_physician"),
    ))
).withColumn(
    "etl_loaded_at", current_timestamp()
).withColumn(
    "etl_source", lit("patients.csv")
)

logger.info(f"Records after cleansing & dedup: {final_df.count()}")

# ---------------------------------------------------------------------------
# 5. WRITE — Upsert (MERGE) into Iceberg dim_patients
# ---------------------------------------------------------------------------
# Register as temp view for SparkSQL MERGE
final_df.createOrReplaceTempView("incoming_patients")

try:
    # Attempt MERGE (for incremental runs) — CDC merge pattern à la Falcon Eye
    merge_sql = f"""
    MERGE INTO {TABLE_NAME} AS target
    USING incoming_patients AS source
    ON target.patient_id = source.patient_id
    WHEN MATCHED AND target.record_hash != source.record_hash THEN UPDATE SET
        target.first_name          = source.first_name,
        target.last_name           = source.last_name,
        target.full_name           = source.full_name,
        target.date_of_birth       = source.date_of_birth,
        target.age                 = source.age,
        target.age_group           = source.age_group,
        target.gender              = source.gender,
        target.blood_type          = source.blood_type,
        target.email               = source.email,
        target.phone               = source.phone,
        target.address             = source.address,
        target.city                = source.city,
        target.state               = source.state,
        target.zip_code            = source.zip_code,
        target.insurance_provider  = source.insurance_provider,
        target.insurance_id        = source.insurance_id,
        target.primary_physician   = source.primary_physician,
        target.chronic_conditions  = source.chronic_conditions,
        target.registration_date   = source.registration_date,
        target.record_hash         = source.record_hash,
        target.etl_loaded_at       = source.etl_loaded_at,
        target.etl_source          = source.etl_source
    WHEN NOT MATCHED THEN INSERT *
    """
    spark.sql(merge_sql)
    logger.info("MERGE INTO dim_patients completed successfully.")
except Exception as e:
    # First run — table may be empty or not yet created; fall back to append/overwrite
    logger.warning(f"MERGE failed ({e}), falling back to overwrite write.")
    final_df.writeTo(TABLE_NAME).using("iceberg").createOrReplace()
    logger.info("Overwrite write to dim_patients completed.")

# ---------------------------------------------------------------------------
# 6. POST-WRITE VALIDATION
# ---------------------------------------------------------------------------
result_count = spark.sql(f"SELECT COUNT(*) AS cnt FROM {TABLE_NAME}").collect()[0]["cnt"]
logger.info(f"dim_patients now contains {result_count} records.")

# Compact small files (Iceberg maintenance)
try:
    spark.sql(f"CALL glue_catalog.system.rewrite_data_files(table => '{DATABASE_NAME}.dim_patients')")
    logger.info("Iceberg compaction completed.")
except Exception as e:
    logger.warning(f"Compaction skipped: {e}")

job.commit()
logger.info("transform_patients job finished successfully.")
