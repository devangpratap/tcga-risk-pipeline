# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Ingest Raw Data into Bronze Tables
# MAGIC
# MAGIC Downloads three TCGA data sources and loads them into **Delta tables** in `workspace.tcga_bronze`.
# MAGIC
# MAGIC | Table | Source | Description |
# MAGIC |---|---|---|
# MAGIC | `image_metadata` | HuggingFace (tcga-ut) | ~271K rows — tile image paths, cancer types, patient barcodes, splits |
# MAGIC | `pathology_reports` | Mendeley (hyg5xkznpx) | ~9.5K pathology reports with patient barcodes and text |
# MAGIC | `clinical_outcomes` | GDC / Liu et al. 2018 | ~11K patients — demographics, staging, survival endpoints (OS, PFI, DSS) |
# MAGIC
# MAGIC **Note:** Mendeley and GDC are not reachable from serverless compute. Those files are
# MAGIC pre-staged in the Volume via `databricks fs cp` from a local machine.
# MAGIC Only HuggingFace is downloaded at runtime.
# MAGIC
# MAGIC **Idempotent:** safe to re-run — all writes use overwrite mode.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Create Staging Volume

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS workspace.tcga_bronze.raw_files;

# COMMAND ----------

print("Staging volume ready: /Volumes/workspace/tcga_bronze/raw_files/")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 1. Image Metadata (HuggingFace — tcga-ut)
# MAGIC
# MAGIC **Source:** `dakomura/tcga-ut` on HuggingFace (reachable from serverless)
# MAGIC
# MAGIC **Columns:** `path`, `case` (cancer type), `patient` (TCGA barcode), `split_internal`, `facility`, `split_external`

# COMMAND ----------

import requests, os

url = "https://huggingface.co/datasets/dakomura/tcga-ut/resolve/main/train_val_test_split.csv"
dest = "/Volumes/workspace/tcga_bronze/raw_files/train_val_test_split.csv"

print("Downloading image metadata from HuggingFace...")
resp = requests.get(url, allow_redirects=True, timeout=120)
resp.raise_for_status()

with open(dest, "wb") as f:
    f.write(resp.content)

print(f"Downloaded: {os.path.getsize(dest):,} bytes -> {dest}")

# COMMAND ----------

df_image = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("/Volumes/workspace/tcga_bronze/raw_files/train_val_test_split.csv")
)

print(f"Image metadata rows: {df_image.count()}")
df_image.printSchema()
df_image.show(5, truncate=False)

# COMMAND ----------

(
    df_image.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.tcga_bronze.image_metadata")
)

spark.sql("COMMENT ON TABLE workspace.tcga_bronze.image_metadata IS 'Raw image tile metadata from HuggingFace dakomura/tcga-ut — cancer type, patient barcode, train/val/test splits'")

print("workspace.tcga_bronze.image_metadata written")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 2. Pathology Reports (Mendeley — TCGA_Reports)
# MAGIC
# MAGIC **Source:** Mendeley dataset `hyg5xkznpx` — pre-staged to Volume (Mendeley unreachable from serverless)
# MAGIC
# MAGIC **Columns:** `patient_filename` (renamed to `patient_barcode`), `text` (renamed to `report_text`)

# COMMAND ----------

import pandas as pd

reports_path = "/Volumes/workspace/tcga_bronze/raw_files/TCGA_Reports.csv"

pdf_reports = pd.read_csv(reports_path, low_memory=False)
print(f"Shape: {pdf_reports.shape}")
print(f"Original columns: {list(pdf_reports.columns)}")

# Rename to standard names
pdf_reports = pdf_reports.rename(columns={
    "patient_filename": "patient_barcode",
    "text": "report_text"
})

df_reports = spark.createDataFrame(pdf_reports)

print(f"\nPathology reports rows: {df_reports.count()}")
df_reports.printSchema()
df_reports.show(3, truncate=80)

# COMMAND ----------

(
    df_reports.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.tcga_bronze.pathology_reports")
)

spark.sql("COMMENT ON TABLE workspace.tcga_bronze.pathology_reports IS 'Raw pathology reports from Mendeley (hyg5xkznpx) — patient barcode, report text'")

print("workspace.tcga_bronze.pathology_reports written")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 3. Clinical Outcomes (GDC — Liu et al. 2018 TCGA-CDR)
# MAGIC
# MAGIC **Source:** TCGA-CDR supplemental table from Liu et al. 2018 — pre-staged to Volume (GDC unreachable from serverless)
# MAGIC
# MAGIC **Format:** Excel (.xlsx), sheet "TCGA-CDR" — 33 columns including survival endpoints.

# COMMAND ----------

# MAGIC %pip install openpyxl -q

# COMMAND ----------

import pandas as pd

pdf_clinical = pd.read_excel(
    "/Volumes/workspace/tcga_bronze/raw_files/TCGA-CDR-SupplementalTableS1.xlsx",
    sheet_name="TCGA-CDR",
    engine="openpyxl"
)

print(f"Shape: {pdf_clinical.shape}")
print(f"Columns: {list(pdf_clinical.columns)}")
pdf_clinical.head(3)

# COMMAND ----------

# Drop the unnamed index column from the Excel file
pdf_clinical = pdf_clinical.drop(columns=[c for c in pdf_clinical.columns if c.startswith("Unnamed")])

# Replace TCGA sentinel values with None
pdf_clinical = pdf_clinical.replace({
    "[Not Available]": None,
    "[Not Applicable]": None,
    "[Discrepancy]": None,
    "[Unknown]": None
})

df_clinical = spark.createDataFrame(pdf_clinical)

print(f"Clinical outcomes rows: {df_clinical.count()}")
df_clinical.printSchema()
df_clinical.show(5, truncate=40)

# COMMAND ----------

(
    df_clinical.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.tcga_bronze.clinical_outcomes")
)

spark.sql("COMMENT ON TABLE workspace.tcga_bronze.clinical_outcomes IS 'TCGA-CDR clinical outcomes from Liu et al. 2018 — demographics, staging, OS/PFI/DSS/DFI survival endpoints'")

print("workspace.tcga_bronze.clinical_outcomes written")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 4. Validation Summary

# COMMAND ----------

print("=" * 70)
print("BRONZE LAYER INGESTION SUMMARY")
print("=" * 70)

tables = [
    ("workspace.tcga_bronze.image_metadata", "patient"),
    ("workspace.tcga_bronze.pathology_reports", "patient_barcode"),
    ("workspace.tcga_bronze.clinical_outcomes", "bcr_patient_barcode"),
]

for table_name, barcode_col in tables:
    df = spark.table(table_name)
    row_count = df.count()
    unique_patients = df.select(barcode_col).distinct().count()
    sample = [row[0] for row in df.select(barcode_col).distinct().limit(5).collect()]

    print(f"\n  {table_name}")
    print(f"   Rows:            {row_count:,}")
    print(f"   Unique patients: {unique_patients:,}  (column: {barcode_col})")
    print(f"   Sample barcodes: {sample}")

print("\n" + "=" * 70)
print("All three bronze tables loaded successfully.")
print("=" * 70)