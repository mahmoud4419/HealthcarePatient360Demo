-- =============================================================================
-- create_iceberg_tables.sql — Healthcare Patient 360 Lakehouse Demo
-- =============================================================================
-- Athena-compatible DDL to create all Iceberg tables in the healthcare_db
-- database with proper schemas, partitioning, and table properties.
--
-- Prerequisites:
--   1. Create the database:  CREATE DATABASE IF NOT EXISTS healthcare_db;
--   2. Ensure S3 bucket exists: s3://healthcare-patient360-demo/warehouse/
--   3. Run this script via Athena or Spark SQL
--
-- Partitioning strategy:
--   - Time-series tables (admissions, labs, imaging, vitals, omics):
--     partitioned by year/month for time-range pruning
--   - Admissions additionally partitioned by department for department-level queries
--   - dim_patients: unpartitioned (small dimension, ~10K-100K rows)
--
-- Architecture: Athena DDL → Glue Data Catalog → Iceberg on S3
-- =============================================================================

-- ---------------------------------------------------------------------------
-- DATABASE
-- ---------------------------------------------------------------------------
CREATE DATABASE IF NOT EXISTS healthcare_db
COMMENT 'Healthcare Patient 360 Lakehouse Demo — Iceberg tables on S3'
LOCATION 's3://healthcare-patient360-demo/warehouse/';


-- ---------------------------------------------------------------------------
-- 1. dim_patients — Patient dimension (SCD Type 2-ready)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS healthcare_db.dim_patients (
    patient_id            STRING     COMMENT 'Unique patient identifier (PK)',
    first_name            STRING     COMMENT 'Patient first name (InitCap)',
    last_name             STRING     COMMENT 'Patient last name (InitCap)',
    full_name             STRING     COMMENT 'Concatenated first + last name',
    date_of_birth         DATE       COMMENT 'Date of birth (YYYY-MM-DD)',
    age                   INT        COMMENT 'Computed age in years',
    age_group             STRING     COMMENT 'Pediatric / Young Adult / Adult / Senior',
    gender                STRING     COMMENT 'M / F / NB / U',
    blood_type            STRING     COMMENT 'ABO+Rh blood type',
    email                 STRING     COMMENT 'Patient email (lowercase)',
    phone                 STRING     COMMENT 'Phone number (digits only)',
    address               STRING     COMMENT 'Street address',
    city                  STRING     COMMENT 'City (InitCap)',
    state                 STRING     COMMENT 'State code (uppercase)',
    zip_code              STRING     COMMENT 'ZIP / postal code',
    insurance_provider    STRING     COMMENT 'Insurance company name',
    insurance_id          STRING     COMMENT 'Insurance policy ID',
    primary_physician     STRING     COMMENT 'Assigned primary care physician',
    chronic_conditions    ARRAY<STRING>  COMMENT 'List of chronic conditions (parsed from pipe-delimited)',
    registration_date     DATE       COMMENT 'Date patient was registered in the system',
    record_hash           STRING     COMMENT 'MD5 hash for change detection',
    etl_loaded_at         TIMESTAMP  COMMENT 'ETL load timestamp',
    etl_source            STRING     COMMENT 'Source file or system identifier'
)
LOCATION 's3://healthcare-patient360-demo/warehouse/healthcare_db.db/dim_patients/'
TBLPROPERTIES (
    'table_type'          = 'ICEBERG',
    'format'              = 'PARQUET',
    'write_compression'   = 'zstd',
    'optimize_rewrite_delete_file_threshold' = '2'
);


-- ---------------------------------------------------------------------------
-- 2. fact_admissions — Hospital admissions fact table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS healthcare_db.fact_admissions (
    admission_id           STRING     COMMENT 'Unique admission identifier (PK)',
    patient_id             STRING     COMMENT 'FK → dim_patients.patient_id',
    admission_date         DATE       COMMENT 'Date of hospital admission',
    discharge_date         DATE       COMMENT 'Date of discharge (NULL if still admitted)',
    department             STRING     COMMENT 'Hospital department (uppercase)',
    diagnosis_code         STRING     COMMENT 'ICD-10 diagnosis code',
    diagnosis_description  STRING     COMMENT 'Human-readable diagnosis',
    attending_physician    STRING     COMMENT 'Attending physician name',
    room_number            STRING     COMMENT 'Hospital room number',
    bed_number             STRING     COMMENT 'Bed number within room',
    admission_type         STRING     COMMENT 'EMERGENCY / ELECTIVE / URGENT',
    status                 STRING     COMMENT 'ADMITTED / DISCHARGED / TRANSFERRED',
    severity_score         DOUBLE     COMMENT 'Severity score (0.0 – 10.0)',
    severity_category      STRING     COMMENT 'LOW / MEDIUM / HIGH / CRITICAL',
    total_charges          DOUBLE     COMMENT 'Total charges in USD',
    insurance_covered      DOUBLE     COMMENT 'Amount covered by insurance',
    out_of_pocket          DOUBLE     COMMENT 'Patient out-of-pocket amount',
    length_of_stay         INT        COMMENT 'Length of stay in days',
    full_name              STRING     COMMENT 'Patient full name (from dim_patients)',
    age_group              STRING     COMMENT 'Patient age group (from dim_patients)',
    insurance_provider     STRING     COMMENT 'Insurance provider (from dim_patients)',
    patient_found_in_dim   BOOLEAN    COMMENT 'Whether patient exists in dim_patients',
    admission_year         INT        COMMENT 'Year of admission (partition key)',
    admission_month        INT        COMMENT 'Month of admission (partition key)',
    record_hash            STRING     COMMENT 'MD5 hash for CDC merge',
    etl_loaded_at          TIMESTAMP  COMMENT 'ETL load timestamp',
    etl_source             STRING     COMMENT 'Source file identifier'
)
PARTITIONED BY (department, admission_year, admission_month)
LOCATION 's3://healthcare-patient360-demo/warehouse/healthcare_db.db/fact_admissions/'
TBLPROPERTIES (
    'table_type'          = 'ICEBERG',
    'format'              = 'PARQUET',
    'write_compression'   = 'zstd',
    'write.merge.mode'    = 'copy-on-write',
    'optimize_rewrite_delete_file_threshold' = '5'
);


-- ---------------------------------------------------------------------------
-- 3. fact_imaging — Imaging / radiology studies
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS healthcare_db.fact_imaging (
    imaging_id             STRING     COMMENT 'Unique imaging study identifier (PK)',
    patient_id             STRING     COMMENT 'FK → dim_patients.patient_id',
    study_date             DATE       COMMENT 'Date of imaging study',
    modality               STRING     COMMENT 'CT / MRI / X-RAY / ULTRASOUND / PET',
    body_part              STRING     COMMENT 'Anatomical region imaged',
    indication             STRING     COMMENT 'Clinical indication for the study',
    findings               STRING     COMMENT 'Radiological findings',
    impression             STRING     COMMENT 'Radiologist impression / summary',
    radiologist            STRING     COMMENT 'Interpreting radiologist',
    contrast_used          STRING     COMMENT 'YES / NO — contrast agent used',
    num_images             INT        COMMENT 'Number of images in the study',
    radiation_dose_mgy     DOUBLE     COMMENT 'Radiation dose in milligrays',
    study_duration_min     DOUBLE     COMMENT 'Study duration in minutes',
    abnormal_flag          STRING     COMMENT 'Y / N — abnormal findings present',
    priority               STRING     COMMENT 'STAT / ROUTINE / URGENT',
    referring_physician    STRING     COMMENT 'Physician who ordered the study',
    facility               STRING     COMMENT 'Imaging facility name',
    dose_category          STRING     COMMENT 'NONE / LOW / MODERATE / HIGH',
    full_name              STRING     COMMENT 'Patient name (from dim_patients)',
    date_of_birth          DATE       COMMENT 'DOB (from dim_patients)',
    age                    INT        COMMENT 'Age at time of load (from dim_patients)',
    age_group              STRING     COMMENT 'Age group (from dim_patients)',
    gender                 STRING     COMMENT 'Gender (from dim_patients)',
    study_year             INT        COMMENT 'Year of study (partition key)',
    study_month            INT        COMMENT 'Month of study (partition key)',
    record_hash            STRING     COMMENT 'MD5 hash for change detection',
    etl_loaded_at          TIMESTAMP  COMMENT 'ETL load timestamp',
    etl_source             STRING     COMMENT 'Source file identifier'
)
PARTITIONED BY (study_year, study_month)
LOCATION 's3://healthcare-patient360-demo/warehouse/healthcare_db.db/fact_imaging/'
TBLPROPERTIES (
    'table_type'          = 'ICEBERG',
    'format'              = 'PARQUET',
    'write_compression'   = 'zstd'
);


-- ---------------------------------------------------------------------------
-- 4. fact_lab_results — Laboratory test results with trends
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS healthcare_db.fact_lab_results (
    lab_result_id           STRING     COMMENT 'Unique lab result identifier (PK)',
    patient_id              STRING     COMMENT 'FK → dim_patients.patient_id',
    test_date               DATE       COMMENT 'Date the test was performed',
    test_code               STRING     COMMENT 'Standardized test code (e.g., CBC, BMP)',
    test_name               STRING     COMMENT 'Human-readable test name',
    test_category           STRING     COMMENT 'HEMATOLOGY / CHEMISTRY / MICROBIOLOGY / etc.',
    result_value            DOUBLE     COMMENT 'Numeric test result',
    result_unit             STRING     COMMENT 'Unit of measurement',
    reference_range_low     DOUBLE     COMMENT 'Lower bound of normal range',
    reference_range_high    DOUBLE     COMMENT 'Upper bound of normal range',
    ordering_physician      STRING     COMMENT 'Physician who ordered the test',
    specimen_type           STRING     COMMENT 'BLOOD / URINE / TISSUE / CSF',
    lab_facility            STRING     COMMENT 'Laboratory facility name',
    collection_time         TIMESTAMP  COMMENT 'Specimen collection timestamp',
    reporting_time          TIMESTAMP  COMMENT 'Result reporting timestamp',
    -- Derived fields
    abnormal_flag           STRING     COMMENT 'NORMAL / LOW / HIGH / UNKNOWN',
    deviation_pct           DOUBLE     COMMENT 'Percent deviation from reference range',
    critical_flag           STRING     COMMENT 'OK / WARNING / CRITICAL',
    -- Trend fields
    previous_value          DOUBLE     COMMENT 'Previous result value for same patient+test',
    delta_from_previous     DOUBLE     COMMENT 'Difference from previous reading',
    pct_change_from_previous DOUBLE    COMMENT 'Percent change from previous reading',
    moving_avg_3            DOUBLE     COMMENT '3-reading moving average',
    test_sequence_num       INT        COMMENT 'Sequence number for this patient+test',
    trend_direction         STRING     COMMENT 'RISING / FALLING / STABLE / BASELINE',
    turnaround_hours        DOUBLE     COMMENT 'Hours from collection to result reporting',
    -- Patient demographics
    age_group               STRING     COMMENT 'Patient age group (from dim_patients)',
    gender                  STRING     COMMENT 'Patient gender (from dim_patients)',
    -- Partitioning & audit
    test_year               INT        COMMENT 'Year of test (partition key)',
    test_month              INT        COMMENT 'Month of test (partition key)',
    record_hash             STRING     COMMENT 'MD5 hash for change detection',
    etl_loaded_at           TIMESTAMP  COMMENT 'ETL load timestamp',
    etl_source              STRING     COMMENT 'Source file identifier'
)
PARTITIONED BY (test_year, test_month)
LOCATION 's3://healthcare-patient360-demo/warehouse/healthcare_db.db/fact_lab_results/'
TBLPROPERTIES (
    'table_type'          = 'ICEBERG',
    'format'              = 'PARQUET',
    'write_compression'   = 'zstd'
);


-- ---------------------------------------------------------------------------
-- 5. fact_omics — Genomic variants / omics data
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS healthcare_db.fact_omics (
    variant_id              STRING     COMMENT 'Unique variant identifier (PK)',
    patient_id              STRING     COMMENT 'FK → dim_patients.patient_id',
    sample_date             DATE       COMMENT 'Date the genomic sample was collected',
    gene_name               STRING     COMMENT 'Gene symbol (uppercase, e.g., BRCA1)',
    chromosome              STRING     COMMENT 'Chromosome (e.g., chr17)',
    position                INT        COMMENT 'Genomic position',
    ref_allele              STRING     COMMENT 'Reference allele',
    alt_allele              STRING     COMMENT 'Alternate allele',
    variant_type            STRING     COMMENT 'SNV / INDEL / CNV / SV',
    zygosity                STRING     COMMENT 'HETEROZYGOUS / HOMOZYGOUS',
    allele_frequency        DOUBLE     COMMENT 'Population allele frequency',
    read_depth              INT        COMMENT 'Sequencing read depth',
    quality_score           DOUBLE     COMMENT 'Variant quality score (Phred-scaled)',
    clinical_significance   STRING     COMMENT 'Original clinical significance label',
    sequencing_platform     STRING     COMMENT 'Sequencing platform (Illumina, etc.)',
    panel_name              STRING     COMMENT 'Gene panel name',
    -- Enriched fields
    associated_disease      STRING     COMMENT 'Disease associated with gene (ClinVar lookup)',
    clinical_significance_std STRING   COMMENT 'Standardized: PATHOGENIC/VUS/BENIGN/NOT_CLASSIFIED',
    actionability_score     INT        COMMENT 'Clinical actionability score (0-100)',
    is_pharmacogenomic      BOOLEAN    COMMENT 'Whether variant is in a pharmacogene',
    frequency_class         STRING     COMMENT 'ULTRA_RARE / RARE / LOW_FREQUENCY / COMMON',
    -- Patient demographics
    age_group               STRING     COMMENT 'Patient age group (from dim_patients)',
    gender                  STRING     COMMENT 'Patient gender (from dim_patients)',
    -- Partitioning & audit
    sample_year             INT        COMMENT 'Year of sample (partition key)',
    sample_month            INT        COMMENT 'Month of sample (partition key)',
    record_hash             STRING     COMMENT 'MD5 hash for change detection',
    etl_loaded_at           TIMESTAMP  COMMENT 'ETL load timestamp',
    etl_source              STRING     COMMENT 'Source file identifier'
)
PARTITIONED BY (sample_year, sample_month)
LOCATION 's3://healthcare-patient360-demo/warehouse/healthcare_db.db/fact_omics/'
TBLPROPERTIES (
    'table_type'          = 'ICEBERG',
    'format'              = 'PARQUET',
    'write_compression'   = 'zstd'
);


-- ---------------------------------------------------------------------------
-- 6. fact_vitals — Streaming vitals (5-min aggregated windows)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS healthcare_db.fact_vitals (
    patient_id             STRING     COMMENT 'FK → dim_patients.patient_id',
    vital_type             STRING     COMMENT 'HEART_RATE / SYSTOLIC_BP / SPO2 / etc.',
    window_start           STRING     COMMENT 'Window start timestamp (ISO 8601)',
    window_end             STRING     COMMENT 'Window end timestamp (ISO 8601)',
    avg_value              DOUBLE     COMMENT 'Average reading in the 5-min window',
    min_value              DOUBLE     COMMENT 'Minimum reading in the window',
    max_value              DOUBLE     COMMENT 'Maximum reading in the window',
    reading_count          INT        COMMENT 'Number of readings in the window',
    alert_triggered        BOOLEAN    COMMENT 'Whether any reading exceeded thresholds',
    alert_type             STRING     COMMENT 'NONE / LOW / HIGH / BOTH',
    device_id              STRING     COMMENT 'IoT device / sensor identifier',
    unit                   STRING     COMMENT 'Unit of measurement',
    window_year            INT        COMMENT 'Year of window start (partition key)',
    window_month           INT        COMMENT 'Month of window start (partition key)',
    etl_loaded_at          STRING     COMMENT 'ETL load timestamp (ISO 8601)',
    etl_source             STRING     COMMENT 'Source: kinesis_stream'
)
PARTITIONED BY (window_year, window_month)
LOCATION 's3://healthcare-patient360-demo/warehouse/healthcare_db.db/fact_vitals/'
TBLPROPERTIES (
    'table_type'          = 'ICEBERG',
    'format'              = 'PARQUET',
    'write_compression'   = 'snappy',
    'write.metadata.delete-after-commit.enabled' = 'true',
    'write.metadata.previous-versions-max'       = '5'
);


-- ---------------------------------------------------------------------------
-- 7. patient_360_view — Unified Patient 360 (CTAS / refreshed by pipeline)
-- ---------------------------------------------------------------------------
-- NOTE: This table is created/refreshed by the Airflow DAG (Phase 4).
-- The DDL below creates an empty shell; the pipeline populates it via CTAS.
-- See patient_360_view.sql for the full materialization query.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS healthcare_db.patient_360_view (
    patient_id             STRING,
    full_name              STRING,
    age                    INT,
    age_group              STRING,
    gender                 STRING,
    chronic_conditions     ARRAY<STRING>,
    insurance_provider     STRING,
    total_admissions       BIGINT,
    current_status         STRING,
    last_admission_date    DATE,
    avg_length_of_stay     DOUBLE,
    max_severity           STRING,
    total_lab_tests        BIGINT,
    abnormal_lab_count     BIGINT,
    critical_lab_count     BIGINT,
    last_lab_date          DATE,
    total_imaging_studies  BIGINT,
    abnormal_imaging_count BIGINT,
    cumulative_radiation   DOUBLE,
    pathogenic_variants    BIGINT,
    pharmacogenomic_flags  BIGINT,
    composite_risk_level   STRING,
    view_refreshed_at      TIMESTAMP
)
LOCATION 's3://healthcare-patient360-demo/warehouse/healthcare_db.db/patient_360_view/'
TBLPROPERTIES (
    'table_type'          = 'ICEBERG',
    'format'              = 'PARQUET',
    'write_compression'   = 'zstd'
);
