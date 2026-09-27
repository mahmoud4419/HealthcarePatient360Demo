"""
===============================================================================
transform_imaging.py — Healthcare Patient 360 Lakehouse Demo
===============================================================================
PySpark / AWS Glue script that processes imaging_metadata.csv, enriches with
patient demographics from dim_patients, creates aggregated imaging summary
per patient, and writes to `healthcare_db.fact_imaging` Iceberg table.

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
    year, month, count, sum as spark_sum, avg as spark_avg,
    max as spark_max, min as spark_min, first, collect_list, broadcast,
    concat_ws, md5, row_number, datediff, regexp_replace
)
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, DateType
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("transform_imaging")
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
FACT_TABLE = f"glue_catalog.{DATABASE_NAME}.fact_imaging"
DIM_PATIENTS = f"glue_catalog.{DATABASE_NAME}.dim_patients"

logger.info(f"Starting imaging transform from {S3_INPUT_PATH}")

# ---------------------------------------------------------------------------
# 1. INGEST
# ---------------------------------------------------------------------------
IMAGING_SCHEMA = StructType([
    StructField("imaging_id", StringType(), False),
    StructField("patient_id", StringType(), False),
    StructField("study_date", StringType(), True),
    StructField("modality", StringType(), True),          # CT, MRI, X-RAY, ULTRASOUND, PET
    StructField("body_part", StringType(), True),
    StructField("indication", StringType(), True),        # Clinical reason for the study
    StructField("findings", StringType(), True),
    StructField("impression", StringType(), True),        # Radiologist's summary
    StructField("radiologist", StringType(), True),
    StructField("contrast_used", StringType(), True),     # Yes / No
    StructField("num_images", IntegerType(), True),
    StructField("radiation_dose_mgy", DoubleType(), True),
    StructField("study_duration_min", DoubleType(), True),
    StructField("abnormal_flag", StringType(), True),     # Y / N
    StructField("priority", StringType(), True),          # STAT / ROUTINE / URGENT
    StructField("referring_physician", StringType(), True),
    StructField("facility", StringType(), True),
])

try:
    raw_df = spark.read.option("header", "true").schema(IMAGING_SCHEMA).csv(S3_INPUT_PATH)
    raw_count = raw_df.count()
    logger.info(f"Raw imaging records: {raw_count}")
except Exception as e:
    logger.error(f"Failed to read imaging CSV: {e}")
    raise

# ---------------------------------------------------------------------------
# 2. VALIDATE & CLEANSE
# ---------------------------------------------------------------------------
valid_df = raw_df.filter(
    col("imaging_id").isNotNull() &
    col("patient_id").isNotNull() &
    col("study_date").isNotNull()
)

cleansed_df = valid_df.withColumn(
    "study_date", to_date(col("study_date"), "yyyy-MM-dd")
).withColumn(
    "modality", upper(trim(col("modality")))
).withColumn(
    "body_part", upper(trim(col("body_part")))
).withColumn(
    "priority", upper(trim(col("priority")))
).withColumn(
    "contrast_used",
    when(upper(trim(col("contrast_used"))).isin("YES", "Y", "TRUE"), "YES")
    .otherwise("NO")
).withColumn(
    "abnormal_flag",
    when(upper(trim(col("abnormal_flag"))).isin("Y", "YES", "TRUE", "ABNORMAL"), "Y")
    .otherwise("N")
).withColumn(
    "study_year", year(col("study_date"))
).withColumn(
    "study_month", month(col("study_date"))
)

# Categorize radiation dose
cleansed_df = cleansed_df.withColumn(
    "dose_category",
    when(col("radiation_dose_mgy").isNull(), "N/A")
    .when(col("radiation_dose_mgy") == 0, "NONE")
    .when(col("radiation_dose_mgy") < 5, "LOW")
    .when(col("radiation_dose_mgy") < 20, "MODERATE")
    .otherwise("HIGH")
)

logger.info(f"Cleansed imaging records: {cleansed_df.count()}")

# ---------------------------------------------------------------------------
# 3. ENRICH WITH PATIENT DEMOGRAPHICS
# ---------------------------------------------------------------------------
try:
    patients_df = spark.sql(f"""
        SELECT patient_id, full_name, date_of_birth, age, age_group, gender
        FROM {DIM_PATIENTS}
    """)

    enriched_df = cleansed_df.join(
        broadcast(patients_df),
        on="patient_id",
        how="left"
    )
    logger.info("Joined with dim_patients for demographics.")
except Exception as e:
    logger.warning(f"Could not enrich with patient data: {e}")
    enriched_df = cleansed_df
    for c in ["full_name", "date_of_birth", "age", "age_group", "gender"]:
        enriched_df = enriched_df.withColumn(c, lit(None).cast(StringType()))

# ---------------------------------------------------------------------------
# 4. IMAGING SUMMARY — aggregate stats per patient
# ---------------------------------------------------------------------------
# This creates a per-patient summary for the Patient 360 view
imaging_summary_df = (
    enriched_df
    .groupBy("patient_id")
    .agg(
        count("*").alias("total_studies"),
        count(when(col("abnormal_flag") == "Y", True)).alias("abnormal_studies"),
        spark_sum("radiation_dose_mgy").alias("cumulative_radiation_mgy"),
        spark_avg("study_duration_min").alias("avg_study_duration_min"),
        spark_max("study_date").alias("last_study_date"),
        spark_min("study_date").alias("first_study_date"),
        collect_list("modality").alias("modalities_used"),
    )
    .withColumn("abnormal_rate",
        when(col("total_studies") > 0,
             (col("abnormal_studies") / col("total_studies") * 100))
        .otherwise(lit(0.0))
    )
)
# Cache summary for potential downstream use
imaging_summary_df.createOrReplaceTempView("imaging_summary")
logger.info("Imaging summary per patient computed.")

# ---------------------------------------------------------------------------
# 5. ADD AUDIT COLUMNS AND WRITE
# ---------------------------------------------------------------------------
final_df = enriched_df.withColumn(
    "record_hash", md5(concat_ws("||",
        col("imaging_id"), col("patient_id"), col("study_date").cast(StringType()),
        col("modality"), col("findings"), col("abnormal_flag"),
    ))
).withColumn(
    "etl_loaded_at", current_timestamp()
).withColumn(
    "etl_source", lit("imaging_metadata.csv")
)

# Upsert via MERGE
final_df.createOrReplaceTempView("incoming_imaging")

try:
    merge_sql = f"""
    MERGE INTO {FACT_TABLE} AS target
    USING incoming_imaging AS source
    ON target.imaging_id = source.imaging_id
    WHEN MATCHED AND target.record_hash != source.record_hash THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
    """
    spark.sql(merge_sql)
    logger.info("MERGE INTO fact_imaging completed.")
except Exception as e:
    logger.warning(f"MERGE failed ({e}), falling back to overwrite.")
    final_df.writeTo(FACT_TABLE).using("iceberg").createOrReplace()

# ---------------------------------------------------------------------------
# 6. POST-WRITE STATS
# ---------------------------------------------------------------------------
stats = spark.sql(f"""
    SELECT
        COUNT(*)                                          AS total_records,
        COUNT(DISTINCT patient_id)                        AS unique_patients,
        COUNT(CASE WHEN abnormal_flag = 'Y' THEN 1 END)  AS abnormal_count,
        COUNT(DISTINCT modality)                          AS modality_types
    FROM {FACT_TABLE}
""").collect()[0]

logger.info(
    f"fact_imaging: total={stats['total_records']}, "
    f"patients={stats['unique_patients']}, "
    f"abnormal={stats['abnormal_count']}, "
    f"modalities={stats['modality_types']}"
)

job.commit()
logger.info("transform_imaging job finished successfully.")
