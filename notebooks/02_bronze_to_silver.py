# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Bronze to Silver: Clean, Join, Standardize
# MAGIC
# MAGIC Transforms three bronze tables into joined, cleaned silver tables.
# MAGIC
# MAGIC **Input (Bronze):**
# MAGIC | Table | Rows | Patients |
# MAGIC |---|---|---|
# MAGIC | `image_metadata` | 271,710 | 7,175 |
# MAGIC | `pathology_reports` | 9,523 | 9,523 |
# MAGIC | `clinical_outcomes` | 11,160 | 11,160 |
# MAGIC
# MAGIC **Output (Silver):**
# MAGIC | Table | Description |
# MAGIC |---|---|
# MAGIC | `cancer_type_map` | Abbreviation ↔ full name lookup |
# MAGIC | `patients` | One row per patient, inner-joined across all 3 sources |
# MAGIC | `patches` | One row per image patch, FK to patient |
# MAGIC | `data_quality` | Join metrics and null rates |
# MAGIC
# MAGIC **Idempotent:** safe to re-run — all writes use overwrite mode.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 1. Cancer Type Mapping Table

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

CANCER_TYPE_MAP = {
    "ACC": "Adrenocortical_carcinoma",
    "BLCA": "Bladder_Urothelial_Carcinoma",
    "LGG": "Brain_Lower_Grade_Glioma",
    "BRCA": "Breast_invasive_carcinoma",
    "CESC": "Cervical_squamous_cell_carcinoma_and_endocervical_adenocarcinoma",
    "CHOL": "Cholangiocarcinoma",
    "COAD": "Colon_Rectum_adenocarcinoma",
    "READ": "Colon_Rectum_adenocarcinoma",
    "ESCA": "Esophageal_carcinoma",
    "GBM": "Glioblastoma_multiforme",
    "HNSC": "Head_and_Neck_squamous_cell_carcinoma",
    "KICH": "Kidney_Chromophobe",
    "KIRC": "Kidney_renal_clear_cell_carcinoma",
    "KIRP": "Kidney_renal_papillary_cell_carcinoma",
    "LIHC": "Liver_hepatocellular_carcinoma",
    "LUAD": "Lung_adenocarcinoma",
    "LUSC": "Lung_squamous_cell_carcinoma",
    "DLBC": "Lymphoid_Neoplasm_Diffuse_Large_B-cell_Lymphoma",
    "MESO": "Mesothelioma",
    "OV": "Ovarian_serous_cystadenocarcinoma",
    "PAAD": "Pancreatic_adenocarcinoma",
    "PCPG": "Pheochromocytoma_and_Paraganglioma",
    "PRAD": "Prostate_adenocarcinoma",
    "SARC": "Sarcoma",
    "SKCM": "Skin_Cutaneous_Melanoma",
    "STAD": "Stomach_adenocarcinoma",
    "TGCT": "Testicular_Germ_Cell_Tumors",
    "THYM": "Thymoma",
    "THCA": "Thyroid_carcinoma",
    "UCS": "Uterine_Carcinosarcoma",
    "UCEC": "Uterine_Corpus_Endometrial_Carcinoma",
    "UVM": "Uveal_Melanoma",
}

schema = StructType([
    StructField("cancer_type_abbrev", StringType(), False),
    StructField("cancer_type", StringType(), False),
])

df_map = spark.createDataFrame(
    [(k, v) for k, v in CANCER_TYPE_MAP.items()],
    schema=schema
)

(
    df_map.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.tcga_silver.cancer_type_map")
)

print(f"cancer_type_map: {df_map.count()} rows")
df_map.show(35, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 2. Silver Patients Table
# MAGIC
# MAGIC Inner join across all 3 sources — only patients present in images, reports, AND clinical.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2a. Prepare image metadata (distinct patients)

# COMMAND ----------

df_img_raw = spark.table("workspace.tcga_bronze.image_metadata")

# One row per patient: cancer type, facility, splits, patch count
df_img_patients = (
    df_img_raw
    .groupBy("patient", "case", "facility", "split_internal", "split_external")
    .agg(F.count("*").alias("patch_count"))
)

print(f"Image patients (distinct): {df_img_patients.count()}")
df_img_patients.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b. Prepare pathology reports

# COMMAND ----------

df_reports_raw = spark.table("workspace.tcga_bronze.pathology_reports")

# Check barcode format — truncate to 12 chars if longer (TCGA-XX-XXXX format)
df_reports = (
    df_reports_raw
    .withColumn("patient_barcode_clean", F.substring(F.col("patient_barcode"), 1, 12))
    .withColumn("report_word_count", F.size(F.split(F.col("report_text"), "\\s+")))
    .select(
        F.col("patient_barcode_clean").alias("patient_barcode"),
        "report_text",
        "report_word_count"
    )
)

# Show barcode lengths to verify
df_reports_raw.select(
    F.length("patient_barcode").alias("barcode_len"),
    "patient_barcode"
).groupBy("barcode_len").count().show()

print(f"Reports patients: {df_reports.count()}")
df_reports.show(3, truncate=80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2c. Prepare clinical outcomes

# COMMAND ----------

df_clinical_raw = spark.table("workspace.tcga_bronze.clinical_outcomes")

# Rename dot-columns and select relevant fields
df_clinical = (
    df_clinical_raw
    .select(
        F.col("bcr_patient_barcode").alias("patient_barcode"),
        F.col("type").alias("cancer_type_abbrev"),
        F.col("age_at_initial_pathologic_diagnosis").alias("age"),
        "gender",
        "race",
        F.col("ajcc_pathologic_tumor_stage").alias("ajcc_stage"),
        "histological_grade",
        "vital_status",
        F.col("OS").alias("os_event"),
        F.col("`OS.time`").alias("os_days"),
        F.col("PFI").alias("pfi_event"),
        F.col("`PFI.time`").alias("pfi_days"),
    )
)

print(f"Clinical patients: {df_clinical.count()}")
df_clinical.printSchema()
df_clinical.show(5, truncate=40)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2d. Join clinical to cancer type map

# COMMAND ----------

df_type_map = spark.table("workspace.tcga_silver.cancer_type_map")

# Join clinical with type map to get full cancer type name
df_clinical_typed = (
    df_clinical
    .join(df_type_map, on="cancer_type_abbrev", how="left")
)

# Check for any unmatched types
unmatched = df_clinical_typed.filter(F.col("cancer_type").isNull()).select("cancer_type_abbrev").distinct()
if unmatched.count() > 0:
    print("WARNING — unmatched cancer types in clinical data:")
    unmatched.show()
else:
    print("All clinical cancer types matched to the mapping table.")

print(f"Clinical with type names: {df_clinical_typed.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2e. Inner join all three sources

# COMMAND ----------

# Join: image patients <-> clinical (via cancer_type name from map)
# Then join to reports on patient barcode

df_silver_patients = (
    df_img_patients
    .join(
        df_clinical_typed,
        (df_img_patients["patient"] == df_clinical_typed["patient_barcode"])
        & (df_img_patients["case"] == df_clinical_typed["cancer_type"]),
        how="inner"
    )
    .join(
        df_reports,
        df_img_patients["patient"] == df_reports["patient_barcode"],
        how="inner"
    )
    .select(
        df_img_patients["patient"].alias("patient_barcode"),
        df_clinical_typed["cancer_type"],
        df_clinical_typed["cancer_type_abbrev"],
        df_img_patients["facility"],
        F.col("age").cast("int"),
        "gender",
        "race",
        "vital_status",
        "ajcc_stage",
        "histological_grade",
        F.col("os_event").cast("int"),
        F.col("os_days").cast("double"),
        F.col("pfi_event").cast("int"),
        F.col("pfi_days").cast("double"),
        df_reports["report_text"],
        df_reports["report_word_count"].cast("int"),
        "patch_count",
        df_img_patients["split_internal"],
        df_img_patients["split_external"],
    )
)

patient_count = df_silver_patients.count()
print(f"Silver patients (inner join of all 3 sources): {patient_count}")
df_silver_patients.printSchema()
df_silver_patients.show(5, truncate=40)

# COMMAND ----------

(
    df_silver_patients.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.tcga_silver.patients")
)

spark.sql("COMMENT ON TABLE workspace.tcga_silver.patients IS 'One row per TCGA patient — inner join of image metadata, pathology reports, and clinical outcomes'")

print("workspace.tcga_silver.patients written")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 3. Silver Patches Table
# MAGIC
# MAGIC One row per image patch, only for patients in silver_patients.
# MAGIC Parses tile coordinates from the filename: `tile_x`, `tile_y`, `tile_size` from `1_2_506.jpg`.

# COMMAND ----------

# Get the set of patients in silver
silver_patient_barcodes = spark.table("workspace.tcga_silver.patients").select("patient_barcode")

df_patches = (
    df_img_raw
    .join(silver_patient_barcodes, df_img_raw["patient"] == silver_patient_barcodes["patient_barcode"], how="inner")
    .select(
        F.col("patient").alias("patient_barcode"),
        F.col("case").alias("cancer_type"),
        F.col("path").alias("patch_path"),
        "split_internal",
        "split_external",
        "facility",
        # Parse tile coordinates from filename: path ends with /tile_x_tile_y_tile_size.jpg
        F.regexp_extract("path", r"(\d+)_(\d+)_(\d+)\.jpg$", 1).cast("int").alias("tile_x"),
        F.regexp_extract("path", r"(\d+)_(\d+)_(\d+)\.jpg$", 2).cast("int").alias("tile_y"),
        F.regexp_extract("path", r"(\d+)_(\d+)_(\d+)\.jpg$", 3).cast("int").alias("tile_size"),
    )
)

patch_count = df_patches.count()
print(f"Silver patches: {patch_count:,}")
df_patches.show(5, truncate=False)

# COMMAND ----------

(
    df_patches.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.tcga_silver.patches")
)

spark.sql("COMMENT ON TABLE workspace.tcga_silver.patches IS 'One row per image patch — FK to silver.patients, includes parsed tile coordinates'")

print("workspace.tcga_silver.patches written")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 4. Data Quality Table

# COMMAND ----------

from pyspark.sql import Row

# Source patient counts
img_patients = spark.table("workspace.tcga_bronze.image_metadata").select("patient").distinct().count()
rpt_patients = spark.table("workspace.tcga_bronze.pathology_reports").select(
    F.substring("patient_barcode", 1, 12).alias("pb")
).distinct().count()
cln_patients = spark.table("workspace.tcga_bronze.clinical_outcomes").select("bcr_patient_barcode").distinct().count()

# Silver (inner join) count
silver_count = spark.table("workspace.tcga_silver.patients").count()

# Set operations for coverage
img_set = spark.table("workspace.tcga_bronze.image_metadata").select(F.col("patient").alias("pb")).distinct()
rpt_set = spark.table("workspace.tcga_bronze.pathology_reports").select(
    F.substring("patient_barcode", 1, 12).alias("pb")
).distinct()
cln_set = spark.table("workspace.tcga_bronze.clinical_outcomes").select(
    F.col("bcr_patient_barcode").alias("pb")
).distinct()

img_not_rpt = img_set.subtract(rpt_set).count()
img_not_cln = img_set.subtract(cln_set).count()

# Overall null rates in silver_patients
df_sp = spark.table("workspace.tcga_silver.patients")
total = df_sp.count()

null_metrics = []
for col_name in ["ajcc_stage", "histological_grade", "os_days", "pfi_days", "report_text", "age", "race"]:
    null_count = df_sp.filter(F.col(col_name).isNull()).count()
    null_metrics.append(Row(
        metric_type="overall_null_rate",
        dimension=col_name,
        value=round(null_count / total, 4) if total > 0 else 0.0,
        count=null_count
    ))

# Source counts as rows
source_metrics = [
    Row(metric_type="source_patients", dimension="image_metadata", value=float(img_patients), count=img_patients),
    Row(metric_type="source_patients", dimension="pathology_reports", value=float(rpt_patients), count=rpt_patients),
    Row(metric_type="source_patients", dimension="clinical_outcomes", value=float(cln_patients), count=cln_patients),
    Row(metric_type="join_result", dimension="inner_join_all_3", value=float(silver_count), count=silver_count),
    Row(metric_type="coverage_gap", dimension="in_images_not_reports", value=float(img_not_rpt), count=img_not_rpt),
    Row(metric_type="coverage_gap", dimension="in_images_not_clinical", value=float(img_not_cln), count=img_not_cln),
]

# Per cancer type stats
df_by_type = (
    df_sp.groupBy("cancer_type_abbrev")
    .agg(
        F.count("*").alias("n"),
        F.round(F.sum(F.when(F.col("ajcc_stage").isNull(), 1).otherwise(0)) / F.count("*"), 4).alias("stage_null_rate"),
        F.round(F.sum(F.when(F.col("histological_grade").isNull(), 1).otherwise(0)) / F.count("*"), 4).alias("grade_null_rate"),
        F.round(F.sum(F.when(F.col("os_days").isNull(), 1).otherwise(0)) / F.count("*"), 4).alias("os_days_null_rate"),
    )
    .collect()
)

type_metrics = []
for row in df_by_type:
    type_metrics.append(Row(
        metric_type="per_type_count", dimension=row["cancer_type_abbrev"],
        value=float(row["n"]), count=row["n"]
    ))
    type_metrics.append(Row(
        metric_type="per_type_stage_null_rate", dimension=row["cancer_type_abbrev"],
        value=float(row["stage_null_rate"]), count=int(row["stage_null_rate"] * row["n"])
    ))
    type_metrics.append(Row(
        metric_type="per_type_grade_null_rate", dimension=row["cancer_type_abbrev"],
        value=float(row["grade_null_rate"]), count=int(row["grade_null_rate"] * row["n"])
    ))

all_metrics = source_metrics + null_metrics + type_metrics
df_quality = spark.createDataFrame(all_metrics)

(
    df_quality.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.tcga_silver.data_quality")
)

spark.sql("COMMENT ON TABLE workspace.tcga_silver.data_quality IS 'Data quality metrics — join coverage, null rates, per-cancer-type stats'")

print(f"data_quality: {df_quality.count()} metric rows written")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 5. Validation Summary

# COMMAND ----------

df_sp = spark.table("workspace.tcga_silver.patients")
df_pa = spark.table("workspace.tcga_silver.patches")

sp_count = df_sp.count()
pa_count = df_pa.count()
type_count = df_sp.select("cancer_type_abbrev").distinct().count()

print("=" * 70)
print("SILVER LAYER SUMMARY")
print("=" * 70)

print(f"\n  silver.patients:  {sp_count:,} rows")
print(f"  silver.patches:   {pa_count:,} rows")
print(f"  Cancer types:     {type_count}")

print("\n  Top 5 cancer types by patient count:")
(
    df_sp.groupBy("cancer_type_abbrev", "cancer_type")
    .count()
    .orderBy(F.desc("count"))
    .show(5, truncate=False)
)

print("  Null rates for key columns:")
for col_name in ["ajcc_stage", "histological_grade", "os_days", "race"]:
    null_pct = df_sp.filter(F.col(col_name).isNull()).count() / sp_count * 100
    print(f"    {col_name:25s} {null_pct:.1f}%")

print("\n  Sample patients:")
(
    df_sp.select("patient_barcode", "cancer_type_abbrev", "os_days", "patch_count", "report_word_count")
    .limit(5)
    .show(truncate=False)
)

print("=" * 70)
print("Silver layer complete.")
print("=" * 70)