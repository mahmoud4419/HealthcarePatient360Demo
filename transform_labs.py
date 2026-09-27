"""
===============================================================================
transform_labs.py — Healthcare Patient 360 Lakehouse Demo
===============================================================================
PySpark / AWS Glue script that processes lab_results.csv with abnormal flag
derivation, per-patient/test trend calculation (moving average, delta from
previous), and writes to `healthcare_db.fact_lab_results` Iceberg table.

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
from pyspark.sql.functions import (
    col, trim, upper, lower, when, lit, current_timestamp, to_date, to_timestamp,
    year, month, lag, lead, avg as spark_avg, count, round as spark_round,
    row_number, percentile_approx, stddev, abs as spark_abs,
    concat_ws, md5, coalesce, broadcast
)
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("transform_labs")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s"))
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Glue boilerplate
# ---------------------------------------------------------------------------
args = getResolvedOptions(sys.argv, [
    "JOB_NAME", "S3_INPUT_PATH", "S3_WAREHOUSE", "DATABASE_NAME",
])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

spark.conf.set("spark.sql.catalog.glue_catalog", "org.apache.iceberg.spark.SparkCatalog")
spark.conf.set("spark.sql.catalog.glue_catalog.warehouse", args["S3_WAREHOUSE"])
spark.conf.set("spark.sql.catalog.glue_catalog.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog")
spark.conf.set("spark.sql.catalog.glue_catalog.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
spark.conf.set("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")

S3_INPUT_PATH = args["S3_INPUT_PATH"]
DATABASE_NAME = args["DATABASE_NAME"]
FACT_TABLE = f"glue_catalog.{DATABASE_NAME}.fact_lab_results"
DIM_PATIENTS = f"glue_catalog.{DATABASE_NAME}.dim_patients"

logger.info(f"Starting lab results transform from {S3_INPUT_PATH}")

# ---------------------------------------------------------------------------
# 1. INGEST
# ---------------------------------------------------------------------------
LAB_SCHEMA = StructType([
    StructField("lab_result_id", StringType(), False),
    StructField("patient_id", StringType(), False),
    StructField("test_date", StringType(), True),
    StructField("test_code", StringType(), True),
    StructField("test_name", StringType(), True),
    StructField("test_category", StringType(), True),     # Hematology, Chemistry, etc.
    StructField("result_value", DoubleType(), True),
    StructField("result_unit", StringType(), True),
    StructField("reference_range_low", DoubleType(), True),
    StructField("reference_range_high", DoubleType(), True),
    StructField("ordering_physician", StringType(), True),
    StructField("specimen_type", StringType(), True),     # Blood, Urine, etc.
    StructField("lab_facility", StringType(), True),
    StructField("collection_time", StringType(), True),
    StructField("reporting_time", StringType(), True),
])

try:
    raw_df = spark.read.option("header", "true").schema(LAB_SCHEMA).csv(S3_INPUT_PATH)
    raw_count = raw_df.count()
    logger.info(f"Raw lab records: {raw_count}")
except Exception as e:
    logger.error(f"Failed to read lab CSV: {e}")
    raise

# ---------------------------------------------------------------------------
# 2. VALIDATE & CLEANSE
# ---------------------------------------------------------------------------
valid_df = raw_df.filter(
    col("lab_result_id").isNotNull() &
    col("patient_id").isNotNull() &
    col("test_date").isNotNull() &
    col("result_value").isNotNull()
)

cleansed_df = valid_df.withColumn(
    "test_date", to_date(col("test_date"), "yyyy-MM-dd")
).withColumn(
    "collection_time", to_timestamp(col("collection_time"), "yyyy-MM-dd HH:mm:ss")
).withColumn(
    "reporting_time", to_timestamp(col("reporting_time"), "yyyy-MM-dd HH:mm:ss")
).withColumn(
    "test_code", upper(trim(col("test_code")))
).withColumn(
    "test_category", upper(trim(col("test_category")))
).withColumn(
    "specimen_type", upper(trim(col("specimen_type")))
).withColumn(
    "test_year", year(col("test_date"))
).withColumn(
    "test_month", month(col("test_date"))
)

# ---------------------------------------------------------------------------
# 3. ABNORMAL FLAG DERIVATION
# ---------------------------------------------------------------------------
# Compare result_value against reference range [low, high]
cleansed_df = cleansed_df.withColumn(
    "abnormal_flag",
    when(
        (col("reference_range_low").isNotNull()) & (col("reference_range_high").isNotNull()),
        when(col("result_value") < col("reference_range_low"), "LOW")
        .when(col("result_value") > col("reference_range_high"), "HIGH")
        .otherwise("NORMAL")
    ).otherwise("UNKNOWN")  # No reference range available
)

# Severity of abnormality: how far outside the range
cleansed_df = cleansed_df.withColumn(
    "deviation_pct",
    when(
        col("abnormal_flag") == "LOW",
        spark_round(
            (col("reference_range_low") - col("result_value")) / col("reference_range_low") * 100, 2
        )
    ).when(
        col("abnormal_flag") == "HIGH",
        spark_round(
            (col("result_value") - col("reference_range_high")) / col("reference_range_high") * 100, 2
        )
    ).otherwise(lit(0.0))
)

# Critical flag — deviation > 50% warrants alert
cleansed_df = cleansed_df.withColumn(
    "critical_flag",
    when(spark_abs(col("deviation_pct")) > 50, "CRITICAL")
    .when(spark_abs(col("deviation_pct")) > 25, "WARNING")
    .otherwise("OK")
)

# ---------------------------------------------------------------------------
# 4. TREND CALCULATION — per patient + test over time
# ---------------------------------------------------------------------------
# Window: all results for the same patient & test, ordered by date
trend_window = Window.partitionBy("patient_id", "test_code").orderBy("test_date")

# 3-reading moving average
moving_avg_window = (
    Window.partitionBy("patient_id", "test_code")
    .orderBy("test_date")
    .rowsBetween(-2, 0)  # current + 2 prior
)

trend_df = cleansed_df.withColumn(
    "previous_value", lag("result_value", 1).over(trend_window)
).withColumn(
    "delta_from_previous",
    spark_round(col("result_value") - coalesce(col("previous_value"), col("result_value")), 4)
).withColumn(
    "pct_change_from_previous",
    when(
        col("previous_value").isNotNull() & (col("previous_value") != 0),
        spark_round((col("result_value") - col("previous_value")) / col("previous_value") * 100, 2)
    ).otherwise(lit(None))
).withColumn(
    "moving_avg_3", spark_round(spark_avg("result_value").over(moving_avg_window), 4)
).withColumn(
    "test_sequence_num", row_number().over(trend_window)
).withColumn(
    "trend_direction",
    when(col("delta_from_previous") > 0, "RISING")
    .when(col("delta_from_previous") < 0, "FALLING")
    .when(col("delta_from_previous") == 0, "STABLE")
    .otherwise("BASELINE")  # First reading
)

# Turnaround time (collection → reporting)
trend_df = trend_df.withColumn(
    "turnaround_hours",
    when(
        col("collection_time").isNotNull() & col("reporting_time").isNotNull(),
        spark_round(
            (col("reporting_time").cast("long") - col("collection_time").cast("long")) / 3600, 2
        )
    ).otherwise(lit(None))
)

logger.info("Trend calculations completed.")

# ---------------------------------------------------------------------------
# 5. ENRICH WITH PATIENT DEMOGRAPHICS
# ---------------------------------------------------------------------------
try:
    patients_df = spark.sql(f"""
        SELECT patient_id, age_group, gender FROM {DIM_PATIENTS}
    """)
    final_df = trend_df.join(broadcast(patients_df), on="patient_id", how="left")
    logger.info("Joined with dim_patients.")
except Exception as e:
    logger.warning(f"Could not join patients: {e}")
    final_df = trend_df
    final_df = final_df.withColumn("age_group", lit(None).cast(StringType()))
    final_df = final_df.withColumn("gender", lit(None).cast(StringType()))

# Audit columns
final_df = final_df.withColumn(
    "record_hash", md5(concat_ws("||",
        col("lab_result_id"), col("result_value").cast(StringType()),
        col("abnormal_flag"), col("test_date").cast(StringType()),
    ))
).withColumn(
    "etl_loaded_at", current_timestamp()
).withColumn(
    "etl_source", lit("lab_results.csv")
)

logger.info(f"Final lab records: {final_df.count()}")

# ---------------------------------------------------------------------------
# 6. WRITE — MERGE INTO fact_lab_results
# ---------------------------------------------------------------------------
final_df.createOrReplaceTempView("incoming_labs")

try:
    merge_sql = f"""
    MERGE INTO {FACT_TABLE} AS target
    USING incoming_labs AS source
    ON target.lab_result_id = source.lab_result_id
    WHEN MATCHED AND target.record_hash != source.record_hash THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
    """
    spark.sql(merge_sql)
    logger.info("MERGE INTO fact_lab_results completed.")
except Exception as e:
    logger.warning(f"MERGE failed ({e}), falling back to overwrite.")
    final_df.writeTo(FACT_TABLE).using("iceberg").createOrReplace()

# ---------------------------------------------------------------------------
# 7. POST-WRITE STATS
# ---------------------------------------------------------------------------
stats = spark.sql(f"""
    SELECT
        COUNT(*)                                            AS total,
        COUNT(CASE WHEN abnormal_flag != 'NORMAL' THEN 1 END) AS abnormal,
        COUNT(CASE WHEN critical_flag = 'CRITICAL' THEN 1 END) AS critical,
        COUNT(DISTINCT patient_id)                           AS patients,
        COUNT(DISTINCT test_code)                            AS test_types
    FROM {FACT_TABLE}
""").collect()[0]

logger.info(
    f"fact_lab_results: total={stats['total']}, "
    f"abnormal={stats['abnormal']}, critical={stats['critical']}, "
    f"patients={stats['patients']}, tests={stats['test_types']}"
)

job.commit()
logger.info("transform_labs job finished successfully.")
