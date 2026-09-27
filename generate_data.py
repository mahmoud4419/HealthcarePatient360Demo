#!/usr/bin/env python3
"""
Healthcare Patient 360 Demo - Synthetic Data Generator
=======================================================
Generates realistic, internally-consistent healthcare sample data
for 8 interconnected CSV datasets.

Usage:
    python generate_data.py [--output-dir ../data]
"""

import csv
import os
import random
import string
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

try:
    from faker import Faker
except ImportError:
    print("ERROR: 'faker' library required. Install with: pip install faker")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

NUM_PATIENTS = 5000
NUM_PHYSICIANS = 200
NUM_ADMISSIONS = 15000
NUM_IMAGING = 8000
NUM_LAB_RESULTS = 20000
NUM_OMICS = 3000
NUM_VITALS = 50000
NUM_MEDICATIONS = 10000

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Reference / lookup data
# ---------------------------------------------------------------------------

BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
BLOOD_TYPE_WEIGHTS = [0.34, 0.06, 0.09, 0.02, 0.03, 0.01, 0.38, 0.07]

INSURANCE_PROVIDERS = [
    "Blue Cross Blue Shield", "UnitedHealthcare", "Aetna", "Cigna",
    "Humana", "Kaiser Permanente", "Anthem", "Molina Healthcare",
    "Centene", "Medicare", "Medicaid", "Tricare"
]

COUNTRIES = ["United States", "Canada", "United Kingdom"]

CHRONIC_CONDITIONS = [
    "Type 2 Diabetes", "Hypertension", "Asthma", "COPD",
    "Coronary Artery Disease", "Heart Failure", "Atrial Fibrillation",
    "Chronic Kidney Disease", "Osteoarthritis", "Rheumatoid Arthritis",
    "Depression", "Anxiety Disorder", "Hypothyroidism", "Hyperlipidemia",
    "Obesity", "Sleep Apnea", "GERD", "Migraine", "Epilepsy",
    "Parkinson's Disease", "Alzheimer's Disease", "Multiple Sclerosis",
    "Lupus", "Crohn's Disease", "Ulcerative Colitis", "Psoriasis"
]

ALLERGIES = [
    "Penicillin", "Sulfa Drugs", "Aspirin", "NSAIDs", "Codeine",
    "Latex", "Iodine", "Peanuts", "Shellfish", "Eggs", "Soy", "Wheat",
    "Tree Nuts", "Bee Stings", "Morphine", "Erythromycin", "Tetracycline",
    "ACE Inhibitors", "Contrast Dye", "None Known"
]

DEPARTMENTS = [
    "Cardiology", "Oncology", "Neurology", "Orthopedics",
    "Pediatrics", "General Medicine", "ICU", "Emergency"
]

SPECIALTIES = [
    "Cardiology", "Oncology", "Neurology", "Orthopedic Surgery",
    "Pediatrics", "Internal Medicine", "Critical Care", "Emergency Medicine",
    "Radiology", "Pathology", "Pulmonology", "Gastroenterology",
    "Endocrinology", "Nephrology", "Infectious Disease", "Psychiatry",
    "General Surgery", "Urology", "Dermatology", "Ophthalmology"
]

SPECIALTY_TO_DEPT = {
    "Cardiology": "Cardiology", "Oncology": "Oncology",
    "Neurology": "Neurology", "Orthopedic Surgery": "Orthopedics",
    "Pediatrics": "Pediatrics", "Internal Medicine": "General Medicine",
    "Critical Care": "ICU", "Emergency Medicine": "Emergency",
    "Radiology": "General Medicine", "Pathology": "General Medicine",
    "Pulmonology": "General Medicine", "Gastroenterology": "General Medicine",
    "Endocrinology": "General Medicine", "Nephrology": "General Medicine",
    "Infectious Disease": "General Medicine", "Psychiatry": "General Medicine",
    "General Surgery": "General Medicine", "Urology": "General Medicine",
    "Dermatology": "General Medicine", "Ophthalmology": "General Medicine",
}

# Realistic ICD-10 codes grouped by department
ICD10_BY_DEPT = {
    "Cardiology": [
        ("I21.0", "Acute ST elevation myocardial infarction of anterior wall"),
        ("I21.1", "Acute ST elevation myocardial infarction of inferior wall"),
        ("I25.1", "Atherosclerotic heart disease of native coronary artery"),
        ("I48.0", "Paroxysmal atrial fibrillation"),
        ("I48.1", "Persistent atrial fibrillation"),
        ("I50.2", "Systolic congestive heart failure"),
        ("I50.9", "Heart failure, unspecified"),
        ("I10",   "Essential primary hypertension"),
        ("I42.0", "Dilated cardiomyopathy"),
        ("I35.0", "Nonrheumatic aortic valve stenosis"),
        ("I71.3", "Abdominal aortic aneurysm, ruptured"),
        ("I26.9", "Pulmonary embolism without acute cor pulmonale"),
    ],
    "Oncology": [
        ("C34.1", "Malignant neoplasm of upper lobe, bronchus or lung"),
        ("C50.9", "Malignant neoplasm of breast, unspecified"),
        ("C18.9", "Malignant neoplasm of colon, unspecified"),
        ("C61",   "Malignant neoplasm of prostate"),
        ("C25.0", "Malignant neoplasm of head of pancreas"),
        ("C43.9", "Malignant melanoma of skin, unspecified"),
        ("C56.9", "Malignant neoplasm of ovary, unspecified"),
        ("C67.9", "Malignant neoplasm of bladder, unspecified"),
        ("C90.0", "Multiple myeloma"),
        ("C91.0", "Acute lymphoblastic leukemia"),
    ],
    "Neurology": [
        ("G20",   "Parkinson's disease"),
        ("G30.9", "Alzheimer's disease, unspecified"),
        ("G35",   "Multiple sclerosis"),
        ("G40.9", "Epilepsy, unspecified"),
        ("G43.9", "Migraine, unspecified"),
        ("I63.9", "Cerebral infarction, unspecified"),
        ("I61.9", "Nontraumatic intracerebral hemorrhage, unspecified"),
        ("G47.3", "Sleep apnea"),
        ("G71.0", "Muscular dystrophy"),
        ("G12.2", "Motor neuron disease"),
    ],
    "Orthopedics": [
        ("M17.1", "Primary osteoarthritis of knee"),
        ("M16.1", "Primary osteoarthritis of hip"),
        ("S72.0", "Fracture of neck of femur"),
        ("S82.0", "Fracture of patella"),
        ("M54.5", "Low back pain"),
        ("M75.1", "Rotator cuff tear"),
        ("M23.2", "Derangement of meniscus due to old tear"),
        ("S42.0", "Fracture of clavicle"),
        ("M80.0", "Age-related osteoporosis with pathological fracture"),
        ("M51.1", "Lumbar disc degeneration"),
    ],
    "Pediatrics": [
        ("J06.9", "Acute upper respiratory infection, unspecified"),
        ("J18.9", "Pneumonia, unspecified organism"),
        ("J45.2", "Mild intermittent asthma"),
        ("K59.0", "Constipation"),
        ("R50.9", "Fever, unspecified"),
        ("J20.9", "Acute bronchitis, unspecified"),
        ("B34.9", "Viral infection, unspecified"),
        ("L20.9", "Atopic dermatitis, unspecified"),
        ("H66.9", "Otitis media, unspecified"),
        ("K21.0", "Gastro-esophageal reflux disease with esophagitis"),
    ],
    "General Medicine": [
        ("E11.9", "Type 2 diabetes mellitus without complications"),
        ("E11.6", "Type 2 diabetes mellitus with other complications"),
        ("J44.1", "Chronic obstructive pulmonary disease with acute exacerbation"),
        ("N18.3", "Chronic kidney disease, stage 3"),
        ("K74.6", "Cirrhosis of liver, unspecified"),
        ("J15.9", "Bacterial pneumonia, unspecified"),
        ("A41.9", "Sepsis, unspecified organism"),
        ("E78.5", "Hyperlipidemia, unspecified"),
        ("D64.9", "Anemia, unspecified"),
        ("N39.0", "Urinary tract infection, site not specified"),
    ],
    "ICU": [
        ("A41.9", "Sepsis, unspecified organism"),
        ("J96.0", "Acute respiratory failure"),
        ("I46.9", "Cardiac arrest, cause unspecified"),
        ("R57.0", "Cardiogenic shock"),
        ("T78.2", "Anaphylactic shock, unspecified"),
        ("J80",   "Acute respiratory distress syndrome"),
        ("G93.1", "Anoxic brain damage"),
        ("K72.0", "Acute and subacute hepatic failure"),
        ("N17.9", "Acute kidney failure, unspecified"),
        ("I74.9", "Embolism and thrombosis of unspecified artery"),
    ],
    "Emergency": [
        ("S06.0", "Concussion"),
        ("S72.0", "Fracture of neck of femur"),
        ("T78.2", "Anaphylactic shock, unspecified"),
        ("I21.9", "Acute myocardial infarction, unspecified"),
        ("R10.9", "Unspecified abdominal pain"),
        ("R55",   "Syncope and collapse"),
        ("S62.5", "Fracture of thumb"),
        ("T14.9", "Injury, unspecified"),
        ("R07.9", "Chest pain, unspecified"),
        ("J45.9", "Asthma, unspecified"),
    ],
}

IMAGING_MODALITIES = {
    "CT": {
        "body_parts": ["Head", "Chest", "Abdomen", "Pelvis", "Spine", "Extremity"],
        "manufacturers": ["GE Healthcare", "Siemens Healthineers", "Philips", "Canon Medical"],
        "slice_thickness": (0.5, 5.0),
        "pixel_spacing": (0.3, 1.0),
    },
    "MRI": {
        "body_parts": ["Brain", "Spine", "Knee", "Shoulder", "Abdomen", "Pelvis", "Cardiac"],
        "manufacturers": ["Siemens Healthineers", "GE Healthcare", "Philips", "Hitachi"],
        "slice_thickness": (1.0, 5.0),
        "pixel_spacing": (0.3, 1.5),
    },
    "XRAY": {
        "body_parts": ["Chest", "Hand", "Foot", "Knee", "Spine", "Pelvis", "Shoulder"],
        "manufacturers": ["GE Healthcare", "Siemens Healthineers", "Fujifilm", "Carestream"],
        "slice_thickness": (None, None),
        "pixel_spacing": (0.1, 0.2),
    },
    "ULTRASOUND": {
        "body_parts": ["Abdomen", "Pelvis", "Thyroid", "Cardiac", "Carotid", "Obstetric"],
        "manufacturers": ["GE Healthcare", "Philips", "Siemens Healthineers", "Samsung"],
        "slice_thickness": (None, None),
        "pixel_spacing": (0.05, 0.3),
    },
    "PET": {
        "body_parts": ["Whole Body", "Brain", "Chest", "Abdomen"],
        "manufacturers": ["Siemens Healthineers", "GE Healthcare", "Philips"],
        "slice_thickness": (2.0, 5.0),
        "pixel_spacing": (2.0, 4.0),
    },
    "MAMMOGRAPHY": {
        "body_parts": ["Breast"],
        "manufacturers": ["Hologic", "GE Healthcare", "Siemens Healthineers", "Fujifilm"],
        "slice_thickness": (None, None),
        "pixel_spacing": (0.05, 0.1),
    },
}

ABNORMALITY_TYPES = [
    "Mass/Lesion", "Fracture", "Effusion", "Consolidation", "Nodule",
    "Calcification", "Stenosis", "Aneurysm", "Hemorrhage", "Edema",
    "Herniation", "Atrophy", "Inflammation", "Thrombus", "Necrosis",
]

INSTITUTIONS = [
    "Memorial General Hospital", "St. Mary's Medical Center",
    "University Health System", "Metropolitan Hospital",
    "City Medical Center", "Regional Medical Center",
    "Community Health Hospital", "National Medical Institute",
]

# Lab test definitions: (name, category, unit, ref_low, ref_high)
LAB_TESTS = [
    # Hematology
    ("Complete Blood Count (CBC)", "Hematology", "x10^9/L", 4.5, 11.0),
    ("Hemoglobin", "Hematology", "g/dL", 12.0, 17.5),
    ("Hematocrit", "Hematology", "percent", 36.0, 50.0),
    ("Platelet Count", "Hematology", "x10^9/L", 150.0, 400.0),
    ("White Blood Cell Count", "Hematology", "x10^9/L", 4.0, 11.0),
    ("Red Blood Cell Count", "Hematology", "x10^6/uL", 4.0, 5.8),
    ("Mean Corpuscular Volume", "Hematology", "fL", 80.0, 100.0),
    ("Reticulocyte Count", "Hematology", "percent", 0.5, 2.5),
    ("Erythrocyte Sedimentation Rate", "Hematology", "mm/hr", 0.0, 20.0),
    ("INR", "Hematology", "ratio", 0.8, 1.2),
    ("PT", "Hematology", "seconds", 11.0, 13.5),
    ("aPTT", "Hematology", "seconds", 25.0, 35.0),
    ("D-Dimer", "Hematology", "ng/mL", 0.0, 500.0),
    # Chemistry
    ("Blood Glucose (Fasting)", "Chemistry", "mg/dL", 70.0, 100.0),
    ("HbA1c", "Chemistry", "percent", 4.0, 5.6),
    ("BUN", "Chemistry", "mg/dL", 7.0, 20.0),
    ("Creatinine", "Chemistry", "mg/dL", 0.6, 1.2),
    ("eGFR", "Chemistry", "mL/min/1.73m2", 90.0, 120.0),
    ("Sodium", "Chemistry", "mEq/L", 136.0, 145.0),
    ("Potassium", "Chemistry", "mEq/L", 3.5, 5.0),
    ("Chloride", "Chemistry", "mEq/L", 98.0, 106.0),
    ("Calcium", "Chemistry", "mg/dL", 8.5, 10.5),
    ("Magnesium", "Chemistry", "mg/dL", 1.7, 2.2),
    ("Phosphorus", "Chemistry", "mg/dL", 2.5, 4.5),
    ("Total Protein", "Chemistry", "g/dL", 6.0, 8.3),
    ("Albumin", "Chemistry", "g/dL", 3.5, 5.5),
    ("Total Bilirubin", "Chemistry", "mg/dL", 0.1, 1.2),
    ("ALT", "Chemistry", "U/L", 7.0, 56.0),
    ("AST", "Chemistry", "U/L", 10.0, 40.0),
    ("Alkaline Phosphatase", "Chemistry", "U/L", 44.0, 147.0),
    ("LDH", "Chemistry", "U/L", 140.0, 280.0),
    ("Total Cholesterol", "Chemistry", "mg/dL", 0.0, 200.0),
    ("LDL Cholesterol", "Chemistry", "mg/dL", 0.0, 100.0),
    ("HDL Cholesterol", "Chemistry", "mg/dL", 40.0, 60.0),
    ("Triglycerides", "Chemistry", "mg/dL", 0.0, 150.0),
    ("Troponin I", "Chemistry", "ng/mL", 0.0, 0.04),
    ("CRP", "Chemistry", "mg/L", 0.0, 3.0),
    ("Procalcitonin", "Chemistry", "ng/mL", 0.0, 0.1),
    ("TSH", "Chemistry", "mIU/L", 0.4, 4.0),
    ("Free T4", "Chemistry", "ng/dL", 0.8, 1.8),
    ("Lactate", "Chemistry", "mmol/L", 0.5, 2.0),
    # Microbiology
    ("Blood Culture", "Microbiology", "qualitative", 0, 1),
    ("Urine Culture", "Microbiology", "CFU/mL", 0, 100000),
    ("Sputum Culture", "Microbiology", "qualitative", 0, 1),
    ("C. difficile Toxin", "Microbiology", "qualitative", 0, 1),
    ("COVID-19 PCR", "Microbiology", "qualitative", 0, 1),
    ("Influenza A/B", "Microbiology", "qualitative", 0, 1),
    ("RSV Antigen", "Microbiology", "qualitative", 0, 1),
    # Immunology
    ("ANA", "Immunology", "titer", 0, 1),
    ("Rheumatoid Factor", "Immunology", "IU/mL", 0.0, 14.0),
    ("Anti-CCP", "Immunology", "U/mL", 0.0, 20.0),
    ("Complement C3", "Immunology", "mg/dL", 90.0, 180.0),
    ("Complement C4", "Immunology", "mg/dL", 10.0, 40.0),
    ("IgG", "Immunology", "mg/dL", 700.0, 1600.0),
    ("IgA", "Immunology", "mg/dL", 70.0, 400.0),
    ("IgM", "Immunology", "mg/dL", 40.0, 230.0),
    # Genetics
    ("BRCA1/BRCA2 Panel", "Genetics", "qualitative", 0, 1),
    ("Karyotype", "Genetics", "qualitative", 0, 1),
    ("HER2", "Genetics", "qualitative", 0, 1),
    ("EGFR Mutation", "Genetics", "qualitative", 0, 1),
    ("BCR-ABL", "Genetics", "qualitative", 0, 1),
]

GENE_NAMES = [
    "BRCA1", "BRCA2", "TP53", "EGFR", "KRAS", "BRAF", "PIK3CA",
    "ALK", "MET", "ROS1", "HER2", "PTEN", "APC", "MLH1", "MSH2",
    "RB1", "VHL", "NF1", "NF2", "CFTR", "APOE", "LRRK2", "GBA",
    "FMR1", "HTT", "DMD", "SMN1", "FGFR3", "RET", "JAK2",
    "NOTCH1", "IDH1", "IDH2", "FLT3", "NPM1", "DNMT3A", "TET2",
    "SF3B1", "CALR", "MPL", "ABL1", "BCR", "MYC", "BCL2", "CDKN2A",
]

GENE_CONDITIONS = {
    "BRCA1": "Hereditary Breast and Ovarian Cancer",
    "BRCA2": "Hereditary Breast and Ovarian Cancer",
    "TP53": "Li-Fraumeni Syndrome",
    "EGFR": "Non-Small Cell Lung Cancer",
    "KRAS": "Colorectal Cancer / Pancreatic Cancer",
    "BRAF": "Melanoma / Colorectal Cancer",
    "PIK3CA": "Breast Cancer",
    "ALK": "Non-Small Cell Lung Cancer",
    "MET": "Hepatocellular Carcinoma",
    "ROS1": "Non-Small Cell Lung Cancer",
    "HER2": "Breast Cancer",
    "PTEN": "Cowden Syndrome",
    "APC": "Familial Adenomatous Polyposis",
    "MLH1": "Lynch Syndrome",
    "MSH2": "Lynch Syndrome",
    "RB1": "Retinoblastoma",
    "VHL": "Von Hippel-Lindau Disease",
    "CFTR": "Cystic Fibrosis",
    "APOE": "Alzheimer's Disease Risk",
    "LRRK2": "Parkinson's Disease",
    "GBA": "Gaucher Disease / Parkinson's Risk",
    "HTT": "Huntington's Disease",
    "DMD": "Duchenne Muscular Dystrophy",
    "FGFR3": "Achondroplasia",
    "RET": "Multiple Endocrine Neoplasia",
    "JAK2": "Myeloproliferative Neoplasm",
    "FLT3": "Acute Myeloid Leukemia",
    "NPM1": "Acute Myeloid Leukemia",
    "IDH1": "Glioma / Acute Myeloid Leukemia",
    "IDH2": "Glioma / Acute Myeloid Leukemia",
}

CHROMOSOMES = [str(i) for i in range(1, 23)] + ["X", "Y"]

MEDICATION_DATA = [
    # (name, drug_class, route, dosages, frequencies)
    ("Metformin", "Antidiabetic", "Oral", ["500mg", "850mg", "1000mg"], ["BID", "TID"]),
    ("Lisinopril", "ACE Inhibitor", "Oral", ["5mg", "10mg", "20mg", "40mg"], ["QD"]),
    ("Amlodipine", "Calcium Channel Blocker", "Oral", ["2.5mg", "5mg", "10mg"], ["QD"]),
    ("Metoprolol", "Beta Blocker", "Oral", ["25mg", "50mg", "100mg", "200mg"], ["BID"]),
    ("Atorvastatin", "Statin", "Oral", ["10mg", "20mg", "40mg", "80mg"], ["QD"]),
    ("Omeprazole", "Proton Pump Inhibitor", "Oral", ["20mg", "40mg"], ["QD", "BID"]),
    ("Levothyroxine", "Thyroid Hormone", "Oral", ["25mcg", "50mcg", "75mcg", "100mcg"], ["QD"]),
    ("Albuterol", "Bronchodilator", "Inhaled", ["90mcg/actuation"], ["Q4H PRN"]),
    ("Warfarin", "Anticoagulant", "Oral", ["1mg", "2mg", "2.5mg", "5mg", "7.5mg", "10mg"], ["QD"]),
    ("Insulin Glargine", "Insulin", "Subcutaneous", ["10 units", "20 units", "30 units", "40 units"], ["QD"]),
    ("Furosemide", "Loop Diuretic", "Oral", ["20mg", "40mg", "80mg"], ["QD", "BID"]),
    ("Hydrochlorothiazide", "Thiazide Diuretic", "Oral", ["12.5mg", "25mg", "50mg"], ["QD"]),
    ("Prednisone", "Corticosteroid", "Oral", ["5mg", "10mg", "20mg", "40mg", "60mg"], ["QD", "BID"]),
    ("Ciprofloxacin", "Fluoroquinolone", "Oral", ["250mg", "500mg", "750mg"], ["BID"]),
    ("Amoxicillin", "Penicillin Antibiotic", "Oral", ["250mg", "500mg", "875mg"], ["BID", "TID"]),
    ("Azithromycin", "Macrolide Antibiotic", "Oral", ["250mg", "500mg"], ["QD"]),
    ("Ceftriaxone", "Cephalosporin", "IV", ["1g", "2g"], ["QD", "BID"]),
    ("Vancomycin", "Glycopeptide Antibiotic", "IV", ["500mg", "1g", "1.5g"], ["Q12H"]),
    ("Piperacillin-Tazobactam", "Beta-Lactam Combination", "IV", ["3.375g", "4.5g"], ["Q6H", "Q8H"]),
    ("Morphine", "Opioid Analgesic", "IV", ["2mg", "4mg", "8mg"], ["Q4H PRN"]),
    ("Hydromorphone", "Opioid Analgesic", "IV", ["0.5mg", "1mg", "2mg"], ["Q3H PRN"]),
    ("Acetaminophen", "Analgesic/Antipyretic", "Oral", ["325mg", "500mg", "650mg", "1000mg"], ["Q6H PRN"]),
    ("Ibuprofen", "NSAID", "Oral", ["200mg", "400mg", "600mg", "800mg"], ["Q6H PRN", "TID"]),
    ("Gabapentin", "Anticonvulsant", "Oral", ["100mg", "300mg", "600mg", "900mg"], ["TID"]),
    ("Sertraline", "SSRI", "Oral", ["25mg", "50mg", "100mg", "200mg"], ["QD"]),
    ("Fluoxetine", "SSRI", "Oral", ["10mg", "20mg", "40mg", "60mg"], ["QD"]),
    ("Lorazepam", "Benzodiazepine", "Oral", ["0.5mg", "1mg", "2mg"], ["BID", "TID", "PRN"]),
    ("Heparin", "Anticoagulant", "IV", ["5000 units", "10000 units", "25000 units"], ["continuous"]),
    ("Enoxaparin", "LMWH Anticoagulant", "Subcutaneous", ["30mg", "40mg", "60mg", "80mg"], ["QD", "BID"]),
    ("Aspirin", "Antiplatelet", "Oral", ["81mg", "325mg"], ["QD"]),
    ("Clopidogrel", "Antiplatelet", "Oral", ["75mg"], ["QD"]),
    ("Pantoprazole", "Proton Pump Inhibitor", "IV", ["40mg"], ["QD", "BID"]),
    ("Ondansetron", "Antiemetic", "IV", ["4mg", "8mg"], ["Q6H PRN", "Q8H PRN"]),
    ("Dexamethasone", "Corticosteroid", "IV", ["4mg", "8mg", "10mg"], ["QD", "Q6H"]),
    ("Norepinephrine", "Vasopressor", "IV", ["0.1mcg/kg/min", "0.2mcg/kg/min"], ["continuous"]),
    ("Propofol", "Sedative", "IV", ["10mg/mL"], ["continuous"]),
    ("Fentanyl", "Opioid Analgesic", "IV", ["25mcg", "50mcg", "100mcg"], ["Q1H PRN", "continuous"]),
    ("Docetaxel", "Chemotherapy - Taxane", "IV", ["75mg/m2", "100mg/m2"], ["Q3W"]),
    ("Carboplatin", "Chemotherapy - Platinum", "IV", ["AUC 5", "AUC 6"], ["Q3W"]),
    ("Pembrolizumab", "Immunotherapy - PD-1 Inhibitor", "IV", ["200mg"], ["Q3W"]),
    ("Tamoxifen", "Hormonal Therapy", "Oral", ["20mg"], ["QD"]),
    ("Letrozole", "Aromatase Inhibitor", "Oral", ["2.5mg"], ["QD"]),
]

INTERACTION_FLAGS = [
    "None", "None", "None", "None", "None",  # most have none
    "CYP3A4 interaction", "QT prolongation risk", "Bleeding risk increased",
    "Hypoglycemia risk", "Nephrotoxicity risk", "Hepatotoxicity risk",
    "Serotonin syndrome risk", "CNS depression", "Hyperkalemia risk",
]

PHARMACY_STATUSES = ["ACTIVE", "DISPENSED", "COMPLETED", "DISCONTINUED", "ON_HOLD"]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def gen_uuid():
    return str(uuid.uuid4())

def gen_phone():
    return f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"

def gen_email(first, last):
    domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "protonmail.com"]
    return f"{first.lower()}.{last.lower()}{random.randint(1,99)}@{random.choice(domains)}"

def gen_insurance_id():
    prefix = random.choice(["INS", "POL", "HMO", "PPO"])
    return f"{prefix}-{random.randint(100000000, 999999999)}"

def gen_license():
    state = random.choice(["MD", "DO"])
    return f"{state}-{random.randint(100000, 999999)}"

def gen_dicom_uid():
    parts = [str(random.randint(1, 999)) for _ in range(8)]
    return "1.2.840.10008." + ".".join(parts)

def random_date(start, end):
    """Random datetime between start and end."""
    delta = end - start
    seconds = int(delta.total_seconds())
    if seconds <= 0:
        return start
    return start + timedelta(seconds=random.randint(0, seconds))

def write_csv(filepath, rows, fieldnames):
    """Write list of dicts to CSV."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓ {filepath.name}: {len(rows):,} rows")

# ---------------------------------------------------------------------------
# 1. Physicians
# ---------------------------------------------------------------------------
def generate_physicians():
    physicians = []
    for i in range(1, NUM_PHYSICIANS + 1):
        spec = random.choice(SPECIALTIES)
        first = fake.first_name()
        last = fake.last_name()
        physicians.append({
            "physician_id": f"PHY-{i:04d}",
            "first_name": first,
            "last_name": last,
            "specialty": spec,
            "department": SPECIALTY_TO_DEPT.get(spec, "General Medicine"),
            "license_number": gen_license(),
            "years_experience": random.randint(1, 35),
            "email": f"{first.lower()}.{last.lower()}@hospital.org",
        })
    return physicians

# ---------------------------------------------------------------------------
# 2. Patients
# ---------------------------------------------------------------------------
def generate_patients():
    patients = []
    for i in range(1, NUM_PATIENTS + 1):
        gender = random.choice(["Male", "Female"])
        first = fake.first_name_male() if gender == "Male" else fake.first_name_female()
        last = fake.last_name()
        dob = fake.date_of_birth(minimum_age=1, maximum_age=95)

        num_conditions = random.choices([0, 1, 2, 3, 4], weights=[0.25, 0.30, 0.25, 0.15, 0.05])[0]
        conditions = random.sample(CHRONIC_CONDITIONS, min(num_conditions, len(CHRONIC_CONDITIONS)))

        num_allergies = random.choices([0, 1, 2, 3], weights=[0.35, 0.35, 0.20, 0.10])[0]
        if num_allergies == 0:
            allergy_list = ["None Known"]
        else:
            allergy_list = random.sample([a for a in ALLERGIES if a != "None Known"], num_allergies)

        patients.append({
            "patient_id": f"PAT-{i:05d}",
            "first_name": first,
            "last_name": last,
            "dob": dob.isoformat(),
            "gender": gender,
            "blood_type": random.choices(BLOOD_TYPES, weights=BLOOD_TYPE_WEIGHTS)[0],
            "insurance_id": gen_insurance_id(),
            "insurance_provider": random.choice(INSURANCE_PROVIDERS),
            "phone": gen_phone(),
            "email": gen_email(first, last),
            "address": fake.street_address(),
            "city": fake.city(),
            "country": random.choices(COUNTRIES, weights=[0.85, 0.10, 0.05])[0],
            "emergency_contact": fake.name(),
            "chronic_conditions": "; ".join(conditions) if conditions else "None",
            "allergies": "; ".join(allergy_list),
        })
    return patients

# ---------------------------------------------------------------------------
# 3. Admissions
# ---------------------------------------------------------------------------
def generate_admissions(patient_ids, physician_ids_by_dept):
    admissions = []
    now = datetime(2026, 9, 25)
    start_window = datetime(2023, 1, 1)

    for i in range(1, NUM_ADMISSIONS + 1):
        pid = random.choice(patient_ids)
        dept = random.choice(DEPARTMENTS)
        adm_type = random.choices(
            ["INPATIENT", "OUTPATIENT", "EMERGENCY"],
            weights=[0.45, 0.35, 0.20]
        )[0]

        adm_date = random_date(start_window, now - timedelta(days=1))

        # Determine status and discharge
        is_current = random.random() < 0.05  # 5% still admitted
        if is_current:
            discharge_date = ""
            status = random.choice(["ADMITTED", "CRITICAL"])
        else:
            los_days = max(1, int(random.gauss(5, 4)))
            if adm_type == "OUTPATIENT":
                los_days = random.choice([0, 1])
            discharge_date = (adm_date + timedelta(days=los_days)).isoformat()[:10]
            status = random.choices(
                ["DISCHARGED", "TRANSFERRED"],
                weights=[0.90, 0.10]
            )[0]

        diag_code, diag_desc = random.choice(ICD10_BY_DEPT[dept])
        severity = random.randint(1, 10)
        if dept == "ICU":
            severity = max(severity, random.randint(6, 10))
        if dept == "Emergency":
            severity = max(severity, random.randint(4, 10))

        # Pick physician from matching department (or random)
        dept_phys = physician_ids_by_dept.get(dept, [])
        if dept_phys:
            physician_id = random.choice(dept_phys)
        else:
            physician_id = random.choice(list(physician_ids_by_dept.values())[0])

        floor = random.randint(1, 8)
        room = f"{floor}{random.randint(1, 50):02d}"
        bed = random.choice(["A", "B", "C", "D"])

        admissions.append({
            "admission_id": f"ADM-{i:06d}",
            "patient_id": pid,
            "admission_date": adm_date.isoformat()[:10],
            "discharge_date": discharge_date,
            "admission_type": adm_type,
            "department": dept,
            "attending_physician_id": physician_id,
            "diagnosis_code": diag_code,
            "diagnosis_description": diag_desc,
            "severity_score": severity,
            "bed_number": f"{room}-{bed}",
            "room_number": room,
            "status": status,
        })
    return admissions

# ---------------------------------------------------------------------------
# 4. Imaging Metadata
# ---------------------------------------------------------------------------
def generate_imaging(admissions, physician_ids):
    imaging = []
    radiologist_ids = [p for p in physician_ids]  # simplify: any physician can be radiologist
    adm_subset = random.sample(admissions, min(NUM_IMAGING, len(admissions)))

    for i, adm in enumerate(adm_subset, 1):
        modality = random.choice(list(IMAGING_MODALITIES.keys()))
        mod_info = IMAGING_MODALITIES[modality]
        body_part = random.choice(mod_info["body_parts"])
        manufacturer = random.choice(mod_info["manufacturers"])

        adm_date = datetime.fromisoformat(adm["admission_date"])
        study_date = adm_date + timedelta(days=random.randint(0, 5))

        abnormal = random.random() < 0.35
        findings = []
        if abnormal:
            abn_type = random.choice(ABNORMALITY_TYPES)
            confidence = round(random.uniform(0.60, 0.99), 2)
            findings.append(f"{abn_type} detected in {body_part} region")
        else:
            abn_type = ""
            confidence = round(random.uniform(0.85, 0.99), 2)
            findings.append(f"No significant abnormality detected in {body_part}")

        st = mod_info["slice_thickness"]
        ps = mod_info["pixel_spacing"]

        imaging.append({
            "imaging_id": f"IMG-{i:06d}",
            "patient_id": adm["patient_id"],
            "admission_id": adm["admission_id"],
            "study_date": study_date.isoformat()[:10],
            "modality": modality,
            "body_part": body_part,
            "study_description": f"{modality} {body_part} {'with contrast' if random.random() < 0.3 else 'without contrast'}",
            "dicom_series_uid": gen_dicom_uid(),
            "num_images": random.randint(1, 500) if modality in ("CT", "MRI") else random.randint(1, 20),
            "findings_summary": findings[0],
            "abnormality_detected": abnormal,
            "abnormality_type": abn_type,
            "confidence_score": confidence,
            "radiologist_id": random.choice(radiologist_ids),
            "report_status": random.choices(["PENDING", "PRELIMINARY", "FINAL"], weights=[0.10, 0.15, 0.75])[0],
            "pixel_spacing": round(random.uniform(ps[0], ps[1]), 3) if ps[0] else "",
            "slice_thickness": round(random.uniform(st[0], st[1]), 1) if st[0] else "",
            "manufacturer": manufacturer,
            "institution_name": random.choice(INSTITUTIONS),
        })
    return imaging

# ---------------------------------------------------------------------------
# 5. Lab Results
# ---------------------------------------------------------------------------
def generate_lab_results(admissions, physician_ids):
    labs = []
    for i in range(1, NUM_LAB_RESULTS + 1):
        adm = random.choice(admissions)
        test = random.choice(LAB_TESTS)
        test_name, category, unit, ref_low, ref_high = test

        adm_date = datetime.fromisoformat(adm["admission_date"])
        test_date = adm_date + timedelta(hours=random.randint(0, 120))

        # Generate realistic values - most normal, some abnormal
        is_qualitative = unit == "qualitative"
        if is_qualitative:
            result_value = random.choice(["Positive", "Negative", "Negative", "Negative"])
            is_abnormal = result_value == "Positive"
            critical = is_abnormal and random.random() < 0.3
            ref_range_low = ""
            ref_range_high = ""
        else:
            # 75% normal, 25% abnormal
            if random.random() < 0.75:
                result_value = round(random.uniform(ref_low, ref_high), 2)
                is_abnormal = False
            else:
                # Abnormal: some high, some low
                if random.random() < 0.5:
                    result_value = round(ref_high * random.uniform(1.05, 2.5), 2)
                else:
                    result_value = round(ref_low * random.uniform(0.2, 0.95), 2)
                    if result_value < 0:
                        result_value = 0.0
                is_abnormal = True
            critical = is_abnormal and random.random() < 0.15
            ref_range_low = ref_low
            ref_range_high = ref_high

        labs.append({
            "lab_id": f"LAB-{i:06d}",
            "patient_id": adm["patient_id"],
            "admission_id": adm["admission_id"],
            "test_date": test_date.isoformat()[:19],
            "test_name": test_name,
            "test_category": category,
            "result_value": result_value,
            "result_unit": unit,
            "reference_range_low": ref_range_low,
            "reference_range_high": ref_range_high,
            "is_abnormal": is_abnormal,
            "critical_flag": critical,
            "ordering_physician_id": random.choice(physician_ids),
        })
    return labs

# ---------------------------------------------------------------------------
# 6. Omics Data
# ---------------------------------------------------------------------------
def generate_omics(patient_ids):
    omics = []
    bases = ["A", "T", "C", "G"]
    platforms = [
        "Illumina NovaSeq 6000", "Illumina HiSeq X", "Illumina MiSeq",
        "Oxford Nanopore PromethION", "PacBio Sequel IIe",
        "Ion Torrent Genexus", "BGI DNBSEQ-G400"
    ]
    significances = ["Pathogenic", "Likely_Pathogenic", "VUS", "Likely_Benign", "Benign"]
    sig_weights = [0.08, 0.12, 0.30, 0.20, 0.30]

    for i in range(1, NUM_OMICS + 1):
        gene = random.choice(GENE_NAMES)
        chrom = random.choice(CHROMOSOMES)
        variant = random.choices(
            ["SNP", "INDEL", "CNV", "SV"],
            weights=[0.60, 0.25, 0.10, 0.05]
        )[0]

        ref = random.choice(bases)
        if variant == "SNP":
            alt = random.choice([b for b in bases if b != ref])
        elif variant == "INDEL":
            alt = "".join(random.choices(bases, k=random.randint(1, 5)))
        else:
            alt = f"<{variant}>"

        significance = random.choices(significances, weights=sig_weights)[0]
        condition = GENE_CONDITIONS.get(gene, "Unknown association")

        omics.append({
            "sample_id": f"OMC-{i:05d}",
            "patient_id": random.choice(patient_ids),
            "collection_date": fake.date_between(start_date="-3y", end_date="today").isoformat(),
            "sample_type": random.choices(
                ["Blood", "Tissue", "Saliva", "Urine"],
                weights=[0.50, 0.25, 0.15, 0.10]
            )[0],
            "sequencing_platform": random.choice(platforms),
            "gene_name": gene,
            "variant_type": variant,
            "chromosome": chrom,
            "position": random.randint(10000, 250000000),
            "ref_allele": ref,
            "alt_allele": alt,
            "allele_frequency": round(random.uniform(0.001, 0.999), 4),
            "clinical_significance": significance,
            "associated_condition": condition,
            "zygosity": random.choice(["Heterozygous", "Homozygous", "Hemizygous"]),
            "read_depth": random.randint(10, 500),
            "quality_score": round(random.uniform(10.0, 99.9), 1),
        })
    return omics

# ---------------------------------------------------------------------------
# 7. Vitals Streaming (with deterioration patterns)
# ---------------------------------------------------------------------------
def generate_vitals(patient_ids, admissions):
    vitals = []
    device_ids = [f"DEV-{random.randint(1000,9999)}" for _ in range(50)]
    ward_ids = [f"W-{d[0]}{random.randint(1,5)}" for d in DEPARTMENTS for _ in range(2)]

    # Select ~50 patients for detailed deterioration patterns
    deteriorating_patients = set(random.sample(patient_ids, min(50, len(patient_ids))))

    # Build a map of patient -> admitted admission for realism
    current_admissions = [a for a in admissions if a["status"] in ("ADMITTED", "CRITICAL")]
    current_pids = {a["patient_id"] for a in current_admissions}

    record_id = 0
    generated = 0
    target = NUM_VITALS

    # Strategy: generate vitals bursts per patient
    # Each "burst" is a window of vitals for one patient during one stay
    while generated < target:
        pid = random.choice(patient_ids)
        is_deteriorating = pid in deteriorating_patients
        is_current = pid in current_pids

        # Burst length: number of readings for this window
        burst_len = random.randint(3, 30)
        if is_deteriorating:
            burst_len = random.randint(15, 40)

        # Base timestamp
        if is_current:
            base_ts = datetime(2026, 9, 25) - timedelta(hours=random.randint(1, 72))
        else:
            base_ts = random_date(datetime(2024, 1, 1), datetime(2026, 9, 24))

        # Base vitals (normal ranges)
        base_hr = random.gauss(75, 12)
        base_sys = random.gauss(120, 15)
        base_dia = random.gauss(80, 10)
        base_temp = random.gauss(98.4, 0.4)
        base_rr = random.gauss(16, 3)
        base_spo2 = random.gauss(97, 1.5)
        base_pain = random.randint(0, 3)

        for j in range(burst_len):
            if generated >= target:
                break
            record_id += 1
            ts = base_ts + timedelta(minutes=j * random.randint(3, 15))

            # Deterioration: progressive worsening over the burst
            if is_deteriorating and j > burst_len * 0.4:
                progress = (j - burst_len * 0.4) / (burst_len * 0.6)
                hr = base_hr + progress * random.uniform(20, 50) + random.gauss(0, 3)
                sys_bp = base_sys - progress * random.uniform(10, 30) + random.gauss(0, 5)
                dia_bp = base_dia - progress * random.uniform(5, 15) + random.gauss(0, 3)
                temp = base_temp + progress * random.uniform(1.0, 3.5) + random.gauss(0, 0.2)
                rr = base_rr + progress * random.uniform(5, 15) + random.gauss(0, 2)
                spo2 = base_spo2 - progress * random.uniform(3, 10) + random.gauss(0, 0.5)
                pain = min(10, base_pain + int(progress * 5))
                consciousness = random.choices(
                    ["Alert", "Verbal", "Pain", "Unresponsive"],
                    weights=[max(0.01, 0.7 - progress), 0.2, 0.05 + progress * 0.3, progress * 0.2]
                )[0]
            else:
                hr = base_hr + random.gauss(0, 5)
                sys_bp = base_sys + random.gauss(0, 8)
                dia_bp = base_dia + random.gauss(0, 5)
                temp = base_temp + random.gauss(0, 0.3)
                rr = base_rr + random.gauss(0, 2)
                spo2 = base_spo2 + random.gauss(0, 1)
                pain = max(0, min(10, base_pain + random.randint(-1, 1)))
                consciousness = random.choices(
                    ["Alert", "Verbal", "Pain", "Unresponsive"],
                    weights=[0.85, 0.10, 0.04, 0.01]
                )[0]

            vitals.append({
                "record_id": f"VIT-{record_id:07d}",
                "patient_id": pid,
                "timestamp": ts.isoformat(),
                "heart_rate": max(30, min(220, round(hr))),
                "systolic_bp": max(60, min(250, round(sys_bp))),
                "diastolic_bp": max(30, min(150, round(dia_bp))),
                "temperature": max(95.0, min(106.0, round(temp, 1))),
                "respiratory_rate": max(6, min(45, round(rr))),
                "spo2": max(60, min(100, round(spo2, 1))),
                "pain_score": pain,
                "consciousness_level": consciousness,
                "device_id": random.choice(device_ids),
                "ward_id": random.choice(ward_ids),
            })
            generated += 1

    return vitals

# ---------------------------------------------------------------------------
# 8. Medications
# ---------------------------------------------------------------------------
def generate_medications(admissions, physician_ids):
    medications = []
    for i in range(1, NUM_MEDICATIONS + 1):
        adm = random.choice(admissions)
        med = random.choice(MEDICATION_DATA)
        med_name, drug_class, route, dosages, frequencies = med

        adm_date = datetime.fromisoformat(adm["admission_date"])
        start = adm_date + timedelta(days=random.randint(0, 3))
        duration = random.randint(1, 90)
        end = start + timedelta(days=duration)

        status = random.choices(
            PHARMACY_STATUSES,
            weights=[0.25, 0.30, 0.25, 0.10, 0.10]
        )[0]

        medications.append({
            "prescription_id": f"RX-{i:06d}",
            "patient_id": adm["patient_id"],
            "admission_id": adm["admission_id"],
            "medication_name": med_name,
            "dosage": random.choice(dosages),
            "route": route,
            "frequency": random.choice(frequencies),
            "start_date": start.isoformat()[:10],
            "end_date": end.isoformat()[:10] if status != "ACTIVE" else "",
            "prescribing_physician_id": random.choice(physician_ids),
            "pharmacy_status": status,
            "drug_class": drug_class,
            "interaction_flags": random.choice(INTERACTION_FLAGS),
        })
    return medications

# ===========================================================================
# MAIN
# ===========================================================================
def main():
    output_dir = OUTPUT_DIR
    if len(sys.argv) > 2 and sys.argv[1] == "--output-dir":
        output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Healthcare Patient 360 - Synthetic Data Generator")
    print("=" * 65)
    print(f"  Output: {output_dir}\n")

    # -- Physicians --
    print("[1/8] Generating physicians...")
    physicians = generate_physicians()
    physician_ids = [p["physician_id"] for p in physicians]
    physician_ids_by_dept = {}
    for p in physicians:
        physician_ids_by_dept.setdefault(p["department"], []).append(p["physician_id"])
    write_csv(output_dir / "physicians.csv", physicians,
              ["physician_id", "first_name", "last_name", "specialty",
               "department", "license_number", "years_experience", "email"])

    # -- Patients --
    print("[2/8] Generating patients...")
    patients = generate_patients()
    patient_ids = [p["patient_id"] for p in patients]
    write_csv(output_dir / "patients.csv", patients,
              ["patient_id", "first_name", "last_name", "dob", "gender",
               "blood_type", "insurance_id", "insurance_provider", "phone",
               "email", "address", "city", "country", "emergency_contact",
               "chronic_conditions", "allergies"])

    # -- Admissions --
    print("[3/8] Generating admissions...")
    admissions = generate_admissions(patient_ids, physician_ids_by_dept)
    write_csv(output_dir / "admissions.csv", admissions,
              ["admission_id", "patient_id", "admission_date", "discharge_date",
               "admission_type", "department", "attending_physician_id",
               "diagnosis_code", "diagnosis_description", "severity_score",
               "bed_number", "room_number", "status"])

    # -- Imaging --
    print("[4/8] Generating imaging metadata...")
    imaging = generate_imaging(admissions, physician_ids)
    write_csv(output_dir / "imaging_metadata.csv", imaging,
              ["imaging_id", "patient_id", "admission_id", "study_date",
               "modality", "body_part", "study_description", "dicom_series_uid",
               "num_images", "findings_summary", "abnormality_detected",
               "abnormality_type", "confidence_score", "radiologist_id",
               "report_status", "pixel_spacing", "slice_thickness",
               "manufacturer", "institution_name"])

    # -- Lab Results --
    print("[5/8] Generating lab results...")
    lab_results = generate_lab_results(admissions, physician_ids)
    write_csv(output_dir / "lab_results.csv", lab_results,
              ["lab_id", "patient_id", "admission_id", "test_date",
               "test_name", "test_category", "result_value", "result_unit",
               "reference_range_low", "reference_range_high", "is_abnormal",
               "critical_flag", "ordering_physician_id"])

    # -- Omics --
    print("[6/8] Generating omics data...")
    omics = generate_omics(patient_ids)
    write_csv(output_dir / "omics_data.csv", omics,
              ["sample_id", "patient_id", "collection_date", "sample_type",
               "sequencing_platform", "gene_name", "variant_type", "chromosome",
               "position", "ref_allele", "alt_allele", "allele_frequency",
               "clinical_significance", "associated_condition", "zygosity",
               "read_depth", "quality_score"])

    # -- Vitals --
    print("[7/8] Generating vitals streaming data...")
    vitals = generate_vitals(patient_ids, admissions)
    write_csv(output_dir / "vitals_streaming.csv", vitals,
              ["record_id", "patient_id", "timestamp", "heart_rate",
               "systolic_bp", "diastolic_bp", "temperature", "respiratory_rate",
               "spo2", "pain_score", "consciousness_level", "device_id", "ward_id"])

    # -- Medications --
    print("[8/8] Generating medications...")
    medications = generate_medications(admissions, physician_ids)
    write_csv(output_dir / "medications.csv", medications,
              ["prescription_id", "patient_id", "admission_id", "medication_name",
               "dosage", "route", "frequency", "start_date", "end_date",
               "prescribing_physician_id", "pharmacy_status", "drug_class",
               "interaction_flags"])

    print(f"\n{'=' * 65}")
    print("  Generation complete!")
    print(f"  Total records: {sum(len(x) for x in [physicians, patients, admissions, imaging, lab_results, omics, vitals, medications]):,}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
