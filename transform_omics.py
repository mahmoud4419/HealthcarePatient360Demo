"""
===============================================================================
transform_omics.py — Healthcare Patient 360 Lakehouse Demo
===============================================================================
PySpark / AWS Glue script that processes omics_data.csv (genomic variants),
applies variant annotation enrichment (clinical significance classification,
gene-disease mapping), and writes to `healthcare_db.fact_omics` Iceberg table.

This script simulates ClinVar / gnomAD-style annotation enrichment via a
lookup dictionary. In production, this would call AWS HealthOmics or an
external annotation API.

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
    col, trim, upper, when, lit, current_timestamp, to_date,
    year, month, create_map, coalesce, array, split, array_distinct,
    concat_ws, md5, broadcast, udf, struct, count, collect_set
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType,
    ArrayType, MapType
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("transform_omics")
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
FACT_TABLE = f"glue_catalog.{DATABASE_NAME}.fact_omics"
DIM_PATIENTS = f"glue_catalog.{DATABASE_NAME}.dim_patients"

logger.info(f"Starting omics transform from {S3_INPUT_PATH}")

# ---------------------------------------------------------------------------
# VARIANT ANNOTATION LOOKUP (simulated ClinVar/gnomAD)
# ---------------------------------------------------------------------------
# In production, replace with AWS HealthOmics annotation workflow or
# broadcast join against a curated annotation Iceberg table.
GENE_DISEASE_MAP = {
    "BRCA1":  "Breast/Ovarian Cancer",
    "BRCA2":  "Breast/Ovarian Cancer",
    "TP53":   "Li-Fraumeni Syndrome",
    "EGFR":   "Non-Small Cell Lung Cancer",
    "KRAS":   "Colorectal/Lung Cancer",
    "BRAF":   "Melanoma/Colorectal Cancer",
    "APC":    "Familial Adenomatous Polyposis",
    "MLH1":   "Lynch Syndrome",
    "MSH2":   "Lynch Syndrome",
    "CFTR":   "Cystic Fibrosis",
    "HBB":    "Sickle Cell Disease",
    "HTT":    "Huntington Disease",
    "APOE":   "Alzheimer Disease Risk",
    "FMR1":   "Fragile X Syndrome",
    "PTEN":   "Cowden Syndrome",
    "RB1":    "Retinoblastoma",
    "VHL":    "Von Hippel-Lindau Disease",
    "ATM":    "Ataxia-Telangiectasia",
    "PALB2":  "Breast Cancer Risk",
    "CHEK2":  "Breast Cancer Risk",
}

# Pharmacogenomics markers
PHARMACO_GENES = {"CYP2D6", "CYP2C19", "CYP3A4", "DPYD", "TPMT", "UGT1A1", "VKORC1"}

# ---------------------------------------------------------------------------
# 1. INGEST
# ---------------------------------------------------------------------------
OMICS_SCHEMA = StructType([
    StructField("variant_id", StringType(), False),
    StructField("patient_id", StringType(), False),
    StructField("sample_date", StringType(), True),
    StructField("gene_name", StringType(), True),
    StructField("chromosome", StringType(), True),
    StructField("position", IntegerType(), True),
    StructField("ref_allele", StringType(), True),
    StructField("alt_allele", StringType(), True),
    StructField("variant_type", StringType(), True),       # SNV, INDEL, CNV, SV
    StructField("zygosity", StringType(), True),            # Heterozygous / Homozygous
    StructField("allele_frequency", DoubleType(), True),    # Population frequency
    StructField("read_depth", IntegerType(), True),
    StructField("quality_score", DoubleType(), True),
    StructField("clinical_significance", StringType(), True),  # Pathogenic / Benign / VUS
    StructField("sequencing_platform", StringType(), True),
    StructField("panel_name", StringType(), True),
])

try:
    raw_df = spark.read.option("header", "true").schema(OMICS_SCHEMA).csv(S3_INPUT_PATH)
    raw_count = raw_df.count()
    logger.info(f"Raw omics records: {raw_count}")
except Exception as e:
    logger.error(f"Failed to read omics CSV: {e}")
    raise

# ---------------------------------------------------------------------------
# 2. VALIDATE & CLEANSE
# ---------------------------------------------------------------------------
valid_df = raw_df.filter(
    col("variant_id").isNotNull() &
    col("patient_id").isNotNull() &
    col("gene_name").isNotNull()
)

cleansed_df = valid_df.withColumn(
    "sample_date", to_date(col("sample_date"), "yyyy-MM-dd")
).withColumn(
    "gene_name", upper(trim(col("gene_name")))
).withColumn(
    "variant_type", upper(trim(col("variant_type")))
).withColumn(
    "zygosity", upper(trim(col("zygosity")))
).withColumn(
    "clinical_significance", upper(trim(col("clinical_significance")))
).withColumn(
    "sample_year", year(col("sample_date"))
).withColumn(
    "sample_month", month(col("sample_date"))
)

# Quality filter — remove low-confidence calls
cleansed_df = cleansed_df.filter(
    (col("quality_score").isNull()) | (col("quality_score") >= 20)
)

logger.info(f"Quality-filtered omics records: {cleansed_df.count()}")

# ---------------------------------------------------------------------------
# 3. VARIANT ANNOTATION ENRICHMENT
# ---------------------------------------------------------------------------

# 3a. Map gene → associated disease
gene_disease_rows = [(k, v) for k, v in GENE_DISEASE_MAP.items()]
gene_disease_df = spark.createDataFrame(gene_disease_rows, ["gene_name", "associated_disease"])

enriched_df = cleansed_df.join(
    broadcast(gene_disease_df),
    on="gene_name",
    how="left"
).withColumn(
    "associated_disease",
    coalesce(col("associated_disease"), lit("Unknown/Research"))
)

# 3b. Clinical significance standardization
enriched_df = enriched_df.withColumn(
    "clinical_significance_std",
    when(col("clinical_significance").isin("PATHOGENIC", "LIKELY_PATHOGENIC", "LIKELY PATHOGENIC"),
         "PATHOGENIC/LIKELY_PATHOGENIC")
    .when(col("clinical_significance").isin("BENIGN", "LIKELY_BENIGN", "LIKELY BENIGN"),
         "BENIGN/LIKELY_BENIGN")
    .when(col("clinical_significance").isin("VUS", "UNCERTAIN_SIGNIFICANCE", "UNCERTAIN SIGNIFICANCE",
                                             "VARIANT OF UNCERTAIN SIGNIFICANCE"),
         "VUS")
    .otherwise("NOT_CLASSIFIED")
)

# 3c. Actionability score (0-100)
# Pathogenic + known disease = high actionability
enriched_df = enriched_df.withColumn(
    "actionability_score",
    when(
        (col("clinical_significance_std") == "PATHOGENIC/LIKELY_PATHOGENIC") &
        (col("associated_disease") != "Unknown/Research"),
        lit(90)
    ).when(
        col("clinical_significance_std") == "PATHOGENIC/LIKELY_PATHOGENIC",
        lit(70)
    ).when(
        col("clinical_significance_std") == "VUS",
        lit(40)
    ).when(
        col("clinical_significance_std") == "BENIGN/LIKELY_BENIGN",
        lit(10)
    ).otherwise(lit(20))
)

# 3d. Pharmacogenomics flag
enriched_df = enriched_df.withColumn(
    "is_pharmacogenomic",
    when(col("gene_name").isin(*PHARMACO_GENES), lit(True)).otherwise(lit(False))
)

# 3e. Allele frequency classification
enriched_df = enriched_df.withColumn(
    "frequency_class",
    when(col("allele_frequency").isNull(), "UNKNOWN")
    .when(col("allele_frequency") < 0.001, "ULTRA_RARE")
    .when(col("allele_frequency") < 0.01, "RARE")
    .when(col("allele_frequency") < 0.05, "LOW_FREQUENCY")
    .otherwise("COMMON")
)

# ---------------------------------------------------------------------------
# 4. ENRICH WITH PATIENT DEMOGRAPHICS
# ---------------------------------------------------------------------------
try:
    patients_df = spark.sql(f"SELECT patient_id, age_group, gender FROM {DIM_PATIENTS}")
    enriched_df = enriched_df.join(broadcast(patients_df), on="patient_id", how="left")
except Exception as e:
    logger.warning(f"Could not join patients: {e}")
    enriched_df = enriched_df.withColumn("age_group", lit(None).cast(StringType()))
    enriched_df = enriched_df.withColumn("gender", lit(None).cast(StringType()))

# Audit columns
final_df = enriched_df.withColumn(
    "record_hash", md5(concat_ws("||",
        col("variant_id"), col("patient_id"), col("gene_name"),
        col("clinical_significance_std"), col("quality_score").cast(StringType()),
    ))
).withColumn(
    "etl_loaded_at", current_timestamp()
).withColumn(
    "etl_source", lit("omics_data.csv")
)

logger.info(f"Final omics records: {final_df.count()}")

# ---------------------------------------------------------------------------
# 5. WRITE — MERGE INTO fact_omics
# ---------------------------------------------------------------------------
final_df.createOrReplaceTempView("incoming_omics")

try:
    merge_sql = f"""
    MERGE INTO {FACT_TABLE} AS target
    USING incoming_omics AS source
    ON target.variant_id = source.variant_id
       AND target.patient_id = source.patient_id
    WHEN MATCHED AND target.record_hash != source.record_hash THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
    """
    spark.sql(merge_sql)
    logger.info("MERGE INTO fact_omics completed.")
except Exception as e:
    logger.warning(f"MERGE failed ({e}), falling back to overwrite.")
    final_df.writeTo(FACT_TABLE).using("iceberg").createOrReplace()

# ---------------------------------------------------------------------------
# 6. POST-WRITE STATS
# ---------------------------------------------------------------------------
stats = spark.sql(f"""
    SELECT
        COUNT(*)                                                        AS total,
        COUNT(DISTINCT patient_id)                                      AS patients,
        COUNT(CASE WHEN clinical_significance_std = 'PATHOGENIC/LIKELY_PATHOGENIC' THEN 1 END) AS pathogenic,
        COUNT(CASE WHEN is_pharmacogenomic = TRUE THEN 1 END)           AS pharmaco,
        COUNT(DISTINCT gene_name)                                       AS genes
    FROM {FACT_TABLE}
""").collect()[0]

logger.info(
    f"fact_omics: total={stats['total']}, patients={stats['patients']}, "
    f"pathogenic={stats['pathogenic']}, pharmacogenomic={stats['pharmaco']}, "
    f"genes={stats['genes']}"
)

job.commit()
logger.info("transform_omics job finished successfully.")
