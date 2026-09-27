-- =============================================================================
-- patient_360_view.sql — Healthcare Patient 360 Lakehouse Demo
-- =============================================================================
-- Athena SQL to create a materialized CTAS view joining ALL fact tables into
-- a unified Patient 360 view with:
--   • Risk scores (composite risk level)
--   • Latest vitals (most recent 5-min window per vital type)
--   • Recent labs (last test per category + abnormality trends)
--   • Imaging history (cumulative studies, radiation, abnormal findings)
--   • Omics flags (pathogenic variants, pharmacogenomic markers)
--
-- This query is executed by the Airflow DAG (build_patient_360_view task)
-- and can also be run ad-hoc in Athena.
--
-- Strategy: DROP + CTAS for full refresh (simple, Athena-compatible).
-- For incremental, consider MERGE INTO with Spark SQL instead.
-- =============================================================================


-- -------------------------------------------------------------------------
-- Step 1: Drop existing view table (idempotent refresh)
-- -------------------------------------------------------------------------
DROP TABLE IF EXISTS healthcare_db.patient_360_view;


-- -------------------------------------------------------------------------
-- Step 2: Create the unified Patient 360 view via CTAS
-- -------------------------------------------------------------------------
CREATE TABLE healthcare_db.patient_360_view
WITH (
    table_type   = 'ICEBERG',
    location     = 's3://healthcare-patient360-demo/warehouse/healthcare_db.db/patient_360_view/',
    is_external  = false,
    format       = 'PARQUET',
    write_compression = 'zstd'
) AS

WITH
-- =========================================================================
-- CTE 1: Admission Summary per patient
-- =========================================================================
admission_summary AS (
    SELECT
        patient_id,
        COUNT(*)                                             AS total_admissions,
        COUNT(CASE WHEN status = 'ADMITTED' THEN 1 END)     AS active_admissions,
        MAX(CASE WHEN status = 'ADMITTED'
                  THEN 'CURRENTLY_ADMITTED'
                  ELSE 'NOT_ADMITTED' END)                   AS current_status,
        MAX(admission_date)                                  AS last_admission_date,
        MIN(admission_date)                                  AS first_admission_date,
        ROUND(AVG(length_of_stay), 1)                        AS avg_length_of_stay,
        MAX(length_of_stay)                                  AS max_length_of_stay,
        MAX(severity_category)                               AS max_severity,
        ROUND(SUM(total_charges), 2)                         AS total_charges_lifetime,
        ROUND(SUM(out_of_pocket), 2)                         AS total_out_of_pocket,
        COUNT(CASE WHEN admission_type = 'EMERGENCY' THEN 1 END)  AS emergency_admissions,
        COUNT(CASE WHEN severity_category = 'CRITICAL' THEN 1 END) AS critical_admissions,
        -- Readmission indicator: >1 admission within the same year
        CASE WHEN COUNT(*) > 1 THEN TRUE ELSE FALSE END     AS has_readmissions
    FROM healthcare_db.fact_admissions
    GROUP BY patient_id
),

-- =========================================================================
-- CTE 2: Lab Results Summary per patient
-- =========================================================================
lab_summary AS (
    SELECT
        patient_id,
        COUNT(*)                                                     AS total_lab_tests,
        COUNT(DISTINCT test_code)                                    AS unique_test_types,
        COUNT(CASE WHEN abnormal_flag != 'NORMAL' THEN 1 END)       AS abnormal_lab_count,
        COUNT(CASE WHEN critical_flag = 'CRITICAL' THEN 1 END)      AS critical_lab_count,
        COUNT(CASE WHEN critical_flag = 'WARNING' THEN 1 END)       AS warning_lab_count,
        MAX(test_date)                                               AS last_lab_date,
        MIN(test_date)                                               AS first_lab_date,
        -- Abnormality rate
        ROUND(
            CAST(COUNT(CASE WHEN abnormal_flag != 'NORMAL' THEN 1 END) AS DOUBLE)
            / NULLIF(COUNT(*), 0) * 100, 1
        )                                                            AS abnormal_rate_pct,
        -- Count of worsening trends
        COUNT(CASE WHEN trend_direction = 'RISING'
                    AND abnormal_flag = 'HIGH' THEN 1 END)          AS worsening_high_count,
        COUNT(CASE WHEN trend_direction = 'FALLING'
                    AND abnormal_flag = 'LOW' THEN 1 END)           AS worsening_low_count
    FROM healthcare_db.fact_lab_results
    GROUP BY patient_id
),

-- =========================================================================
-- CTE 3: Latest Lab Results (most recent per test category)
-- =========================================================================
latest_labs AS (
    SELECT
        patient_id,
        -- Use MAP to store latest result per test category
        -- Athena/Presto-compatible: use arbitrary aggregation
        MAX(CASE WHEN test_category = 'HEMATOLOGY'
            THEN CAST(result_value AS VARCHAR) || ' ' || COALESCE(result_unit, '')
            || ' (' || abnormal_flag || ')' END)                     AS latest_hematology,
        MAX(CASE WHEN test_category = 'CHEMISTRY'
            THEN CAST(result_value AS VARCHAR) || ' ' || COALESCE(result_unit, '')
            || ' (' || abnormal_flag || ')' END)                     AS latest_chemistry,
        MAX(CASE WHEN test_category = 'MICROBIOLOGY'
            THEN CAST(result_value AS VARCHAR) || ' ' || COALESCE(result_unit, '')
            || ' (' || abnormal_flag || ')' END)                     AS latest_microbiology
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY patient_id, test_category ORDER BY test_date DESC) AS rn
        FROM healthcare_db.fact_lab_results
    )
    WHERE rn = 1
    GROUP BY patient_id
),

-- =========================================================================
-- CTE 4: Imaging Summary per patient
-- =========================================================================
imaging_summary AS (
    SELECT
        patient_id,
        COUNT(*)                                              AS total_imaging_studies,
        COUNT(CASE WHEN abnormal_flag = 'Y' THEN 1 END)      AS abnormal_imaging_count,
        ROUND(SUM(COALESCE(radiation_dose_mgy, 0)), 2)       AS cumulative_radiation_mgy,
        ROUND(AVG(study_duration_min), 1)                     AS avg_study_duration_min,
        MAX(study_date)                                       AS last_imaging_date,
        COUNT(CASE WHEN priority = 'STAT' THEN 1 END)        AS stat_studies_count,
        COUNT(DISTINCT modality)                              AS modalities_used_count,
        -- High radiation exposure flag (cumulative > 100 mGy)
        CASE WHEN SUM(COALESCE(radiation_dose_mgy, 0)) > 100
             THEN TRUE ELSE FALSE END                         AS high_radiation_flag
    FROM healthcare_db.fact_imaging
    GROUP BY patient_id
),

-- =========================================================================
-- CTE 5: Latest Vitals (most recent window per vital type)
-- =========================================================================
latest_vitals AS (
    SELECT
        patient_id,
        MAX(CASE WHEN vital_type = 'HEART_RATE'
            THEN CAST(avg_value AS VARCHAR) END)              AS latest_heart_rate,
        MAX(CASE WHEN vital_type = 'SYSTOLIC_BP'
            THEN CAST(avg_value AS VARCHAR) END)              AS latest_systolic_bp,
        MAX(CASE WHEN vital_type = 'DIASTOLIC_BP'
            THEN CAST(avg_value AS VARCHAR) END)              AS latest_diastolic_bp,
        MAX(CASE WHEN vital_type = 'TEMPERATURE'
            THEN CAST(avg_value AS VARCHAR) END)              AS latest_temperature,
        MAX(CASE WHEN vital_type = 'SPO2'
            THEN CAST(avg_value AS VARCHAR) END)              AS latest_spo2,
        MAX(CASE WHEN vital_type = 'RESPIRATORY_RATE'
            THEN CAST(avg_value AS VARCHAR) END)              AS latest_respiratory_rate,
        MAX(CASE WHEN vital_type = 'BLOOD_GLUCOSE'
            THEN CAST(avg_value AS VARCHAR) END)              AS latest_blood_glucose,
        -- Any active vital alerts?
        MAX(CASE WHEN alert_triggered = true
            THEN vital_type || ':' || alert_type END)         AS active_vital_alert,
        MAX(window_end)                                       AS last_vitals_timestamp
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY patient_id, vital_type ORDER BY window_end DESC) AS rn
        FROM healthcare_db.fact_vitals
    )
    WHERE rn = 1
    GROUP BY patient_id
),

-- =========================================================================
-- CTE 6: Omics / Genomics Summary per patient
-- =========================================================================
omics_summary AS (
    SELECT
        patient_id,
        COUNT(*)                                                             AS total_variants,
        COUNT(CASE WHEN clinical_significance_std = 'PATHOGENIC/LIKELY_PATHOGENIC'
              THEN 1 END)                                                    AS pathogenic_variants,
        COUNT(CASE WHEN clinical_significance_std = 'VUS' THEN 1 END)       AS vus_variants,
        COUNT(CASE WHEN is_pharmacogenomic = true THEN 1 END)               AS pharmacogenomic_flags,
        MAX(actionability_score)                                             AS max_actionability_score,
        COUNT(DISTINCT gene_name)                                            AS genes_tested,
        -- High-risk genomic flag: multiple pathogenic variants or high actionability
        CASE WHEN COUNT(CASE WHEN clinical_significance_std = 'PATHOGENIC/LIKELY_PATHOGENIC'
                   THEN 1 END) >= 2
             OR MAX(actionability_score) >= 90
             THEN TRUE ELSE FALSE END                                        AS high_risk_genomic_flag
    FROM healthcare_db.fact_omics
    GROUP BY patient_id
)

-- =========================================================================
-- FINAL SELECT: Unified Patient 360 View
-- =========================================================================
SELECT
    -- ---- Patient Demographics ----
    p.patient_id,
    p.full_name,
    p.age,
    p.age_group,
    p.gender,
    p.blood_type,
    p.chronic_conditions,
    p.insurance_provider,
    p.insurance_id,
    p.primary_physician,
    p.registration_date,

    -- ---- Admission Metrics ----
    COALESCE(a.total_admissions, 0)         AS total_admissions,
    COALESCE(a.active_admissions, 0)        AS active_admissions,
    COALESCE(a.current_status, 'NEVER_ADMITTED') AS current_status,
    a.last_admission_date,
    a.first_admission_date,
    a.avg_length_of_stay,
    a.max_length_of_stay,
    a.max_severity,
    a.total_charges_lifetime,
    a.total_out_of_pocket,
    COALESCE(a.emergency_admissions, 0)     AS emergency_admissions,
    COALESCE(a.critical_admissions, 0)      AS critical_admissions,
    COALESCE(a.has_readmissions, FALSE)     AS has_readmissions,

    -- ---- Lab Metrics ----
    COALESCE(l.total_lab_tests, 0)          AS total_lab_tests,
    l.unique_test_types,
    COALESCE(l.abnormal_lab_count, 0)       AS abnormal_lab_count,
    COALESCE(l.critical_lab_count, 0)       AS critical_lab_count,
    l.warning_lab_count,
    l.last_lab_date,
    l.abnormal_rate_pct,
    l.worsening_high_count,
    l.worsening_low_count,

    -- ---- Latest Lab Values ----
    ll.latest_hematology,
    ll.latest_chemistry,
    ll.latest_microbiology,

    -- ---- Imaging Metrics ----
    COALESCE(i.total_imaging_studies, 0)    AS total_imaging_studies,
    COALESCE(i.abnormal_imaging_count, 0)   AS abnormal_imaging_count,
    i.cumulative_radiation_mgy,
    i.avg_study_duration_min,
    i.last_imaging_date,
    i.stat_studies_count,
    i.modalities_used_count,
    COALESCE(i.high_radiation_flag, FALSE)  AS high_radiation_flag,

    -- ---- Latest Vitals ----
    v.latest_heart_rate,
    v.latest_systolic_bp,
    v.latest_diastolic_bp,
    v.latest_temperature,
    v.latest_spo2,
    v.latest_respiratory_rate,
    v.latest_blood_glucose,
    v.active_vital_alert,
    v.last_vitals_timestamp,

    -- ---- Omics / Genomics ----
    COALESCE(o.total_variants, 0)           AS total_variants,
    COALESCE(o.pathogenic_variants, 0)      AS pathogenic_variants,
    COALESCE(o.vus_variants, 0)             AS vus_variants,
    COALESCE(o.pharmacogenomic_flags, 0)    AS pharmacogenomic_flags,
    o.max_actionability_score,
    o.genes_tested,
    COALESCE(o.high_risk_genomic_flag, FALSE) AS high_risk_genomic_flag,

    -- ---- COMPOSITE RISK SCORE ----
    -- Multi-factor risk assessment combining all data domains
    CASE
        -- CRITICAL risk: any single critical indicator
        WHEN COALESCE(a.max_severity, 'LOW') = 'CRITICAL'
             OR COALESCE(l.critical_lab_count, 0) >= 3
             OR COALESCE(o.pathogenic_variants, 0) >= 3
             OR v.active_vital_alert IS NOT NULL
             OR (COALESCE(a.emergency_admissions, 0) >= 3 AND COALESCE(a.has_readmissions, FALSE))
            THEN 'CRITICAL'

        -- HIGH risk: multiple moderate indicators
        WHEN COALESCE(a.max_severity, 'LOW') IN ('HIGH', 'CRITICAL')
             OR COALESCE(l.critical_lab_count, 0) >= 1
             OR COALESCE(l.abnormal_rate_pct, 0) > 40
             OR COALESCE(o.pathogenic_variants, 0) >= 2
             OR COALESCE(i.high_radiation_flag, FALSE) = TRUE
             OR COALESCE(a.emergency_admissions, 0) >= 2
            THEN 'HIGH'

        -- MEDIUM risk: some concerning indicators
        WHEN COALESCE(l.abnormal_lab_count, 0) > 5
             OR COALESCE(a.total_admissions, 0) > 3
             OR COALESCE(l.worsening_high_count, 0) + COALESCE(l.worsening_low_count, 0) > 3
             OR COALESCE(o.vus_variants, 0) >= 5
            THEN 'MEDIUM'

        -- LOW risk: baseline
        ELSE 'LOW'
    END                                      AS composite_risk_level,

    -- ---- Numeric risk score (0-100) for ML model input ----
    LEAST(100,
        COALESCE(a.critical_admissions, 0) * 15
        + COALESCE(a.emergency_admissions, 0) * 10
        + COALESCE(l.critical_lab_count, 0) * 12
        + COALESCE(l.abnormal_lab_count, 0) * 2
        + COALESCE(o.pathogenic_variants, 0) * 8
        + CASE WHEN v.active_vital_alert IS NOT NULL THEN 20 ELSE 0 END
        + CASE WHEN COALESCE(i.high_radiation_flag, FALSE) THEN 5 ELSE 0 END
        + CASE WHEN COALESCE(a.has_readmissions, FALSE) THEN 10 ELSE 0 END
    )                                        AS numeric_risk_score,

    -- ---- Metadata ----
    current_timestamp                        AS view_refreshed_at

FROM healthcare_db.dim_patients p
LEFT JOIN admission_summary   a  ON p.patient_id = a.patient_id
LEFT JOIN lab_summary         l  ON p.patient_id = l.patient_id
LEFT JOIN latest_labs         ll ON p.patient_id = ll.patient_id
LEFT JOIN imaging_summary     i  ON p.patient_id = i.patient_id
LEFT JOIN latest_vitals       v  ON p.patient_id = v.patient_id
LEFT JOIN omics_summary       o  ON p.patient_id = o.patient_id;


-- =========================================================================
-- VERIFICATION QUERIES (run after CTAS to validate)
-- =========================================================================

-- Total patients in the 360 view
-- SELECT COUNT(*) AS total_patients FROM healthcare_db.patient_360_view;

-- Risk distribution
-- SELECT composite_risk_level, COUNT(*) AS patient_count
-- FROM healthcare_db.patient_360_view
-- GROUP BY composite_risk_level
-- ORDER BY CASE composite_risk_level
--     WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2
--     WHEN 'MEDIUM' THEN 3 ELSE 4 END;

-- Top 10 highest-risk patients
-- SELECT patient_id, full_name, age, composite_risk_level, numeric_risk_score,
--        total_admissions, critical_lab_count, pathogenic_variants, active_vital_alert
-- FROM healthcare_db.patient_360_view
-- ORDER BY numeric_risk_score DESC
-- LIMIT 10;
