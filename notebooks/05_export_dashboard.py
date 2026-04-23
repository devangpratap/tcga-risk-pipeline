# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Export Dashboard Tables
# MAGIC
# MAGIC Materializes five Gold-layer summary tables for BI consumption (PowerBI / Databricks SQL).
# MAGIC
# MAGIC **Input (Gold / Silver):**
# MAGIC | Table | Description |
# MAGIC |---|---|
# MAGIC | `workspace.tcga_silver.patients` | 6,242 patients with demographics and clinical data |
# MAGIC | `workspace.tcga_gold.risk_labels` | Risk label per patient (high_risk / low_risk / censored) |
# MAGIC | `workspace.tcga_gold.model_predictions` | Per-patient predicted label and confidence |
# MAGIC | `workspace.tcga_gold.model_metrics` | Model performance metrics (train / test splits) |
# MAGIC | `workspace.tcga_silver.data_quality` | Join and null-rate metrics |
# MAGIC
# MAGIC **Output (Gold — Dashboard):**
# MAGIC | Table | Description |
# MAGIC |---|---|
# MAGIC | `workspace.tcga_gold.dash_risk_by_cancer` | Risk distribution aggregated by cancer type |
# MAGIC | `workspace.tcga_gold.dash_model_performance` | One row per model with pivoted test-set metrics |
# MAGIC | `workspace.tcga_gold.dash_data_quality` | Join rates and null rates per data source |
# MAGIC | `workspace.tcga_gold.dash_demographics` | Patient counts and risk rates by cancer type, gender, race |
# MAGIC | `workspace.tcga_gold.dash_predictions` | Flat prediction detail table for patient-level drill-down |
# MAGIC
# MAGIC **Idempotent:** safe to re-run — all writes use overwrite mode.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 0. Imports and Source Table Reads

# COMMAND ----------

from pyspark.sql import functions as F

# --- Source reads ---
patients      = spark.table("workspace.tcga_silver.patients")
risk_labels   = spark.table("workspace.tcga_gold.risk_labels")
predictions   = spark.table("workspace.tcga_gold.model_predictions")
model_metrics = spark.table("workspace.tcga_gold.model_metrics")
data_quality  = spark.table("workspace.tcga_silver.data_quality")

print("Source tables loaded.")
print(f"  patients:       {patients.count():,} rows")
print(f"  risk_labels:    {risk_labels.count():,} rows")
print(f"  predictions:    {predictions.count():,} rows")
print(f"  model_metrics:  {model_metrics.count():,} rows")
print(f"  data_quality:   {data_quality.count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 1: Create Dashboard Summary Tables

# COMMAND ----------

# MAGIC %md
# MAGIC ### Table 1: `dash_risk_by_cancer`
# MAGIC Risk distribution (high / low / censored) aggregated by cancer type, with median OS and mean age.

# COMMAND ----------

# Join risk labels with patient demographics
# Use risk_labels for cancer_type/cancer_type_abbrev to avoid ambiguity
risk_patients = (
    risk_labels
    .join(
        patients.select("patient_barcode", "age", "gender"),
        on="patient_barcode",
        how="inner"
    )
)

dash_risk_by_cancer = (
    risk_patients
    .groupBy("cancer_type", "cancer_type_abbrev")
    .agg(
        F.count("patient_barcode").alias("total_patients"),
        F.count(
            F.when(F.col("risk_label") == "high_risk", F.col("patient_barcode"))
        ).alias("high_risk_count"),
        F.count(
            F.when(F.col("risk_label") == "low_risk", F.col("patient_barcode"))
        ).alias("low_risk_count"),
        F.percentile_approx("os_days", 0.5).alias("median_os_days"),
        F.round(F.avg("age"), 2).alias("mean_age"),
    )
    .withColumn(
        # Exclude censored from denominator: denominator = high_risk + low_risk
        "high_risk_pct",
        F.round(
            F.col("high_risk_count")
            / F.nullif(
                F.col("high_risk_count") + F.col("low_risk_count"),
                F.lit(0)
            ) * 100,
            2
        )
    )
    .orderBy("cancer_type")
)

(
    dash_risk_by_cancer
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.tcga_gold.dash_risk_by_cancer")
)

spark.sql("""
    COMMENT ON TABLE workspace.tcga_gold.dash_risk_by_cancer IS
    'Dashboard: risk label distribution (high/low) by cancer type, with median OS and mean age. Censored patients excluded from high_risk_pct denominator.'
""")

print("dash_risk_by_cancer written.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Table 2: `dash_model_performance`
# MAGIC One row per model — test-set metrics pivoted into wide format for easy BI consumption.

# COMMAND ----------

# Filter to test split only, then pivot metric_name -> columns
test_metrics = model_metrics.filter(F.col("split") == "test")

dash_model_performance = (
    test_metrics
    .groupBy("model_name")
    .agg(
        F.round(
            F.first(F.when(F.col("metric_name") == "accuracy",          F.col("metric_value"))), 4
        ).alias("accuracy"),
        F.round(
            F.first(F.when(F.col("metric_name") == "balanced_accuracy", F.col("metric_value"))), 4
        ).alias("balanced_accuracy"),
        F.round(
            F.first(F.when(F.col("metric_name") == "precision_high",    F.col("metric_value"))), 4
        ).alias("precision_high"),
        F.round(
            F.first(F.when(F.col("metric_name") == "recall_high",       F.col("metric_value"))), 4
        ).alias("recall_high"),
        F.round(
            F.first(F.when(F.col("metric_name") == "f1_high",           F.col("metric_value"))), 4
        ).alias("f1_high"),
        F.round(
            F.first(F.when(F.col("metric_name") == "precision_low",     F.col("metric_value"))), 4
        ).alias("precision_low"),
        F.round(
            F.first(F.when(F.col("metric_name") == "recall_low",        F.col("metric_value"))), 4
        ).alias("recall_low"),
        F.round(
            F.first(F.when(F.col("metric_name") == "f1_low",            F.col("metric_value"))), 4
        ).alias("f1_low"),
    )
    .orderBy("model_name")
)

(
    dash_model_performance
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.tcga_gold.dash_model_performance")
)

spark.sql("""
    COMMENT ON TABLE workspace.tcga_gold.dash_model_performance IS
    'Dashboard: test-set model performance metrics pivoted to wide format — one row per model.'
""")

print("dash_model_performance written.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Table 3: `dash_data_quality`
# MAGIC Join rates and null rates per data source, sourced from `tcga_silver.data_quality`.

# COMMAND ----------

# data_quality table has columns: metric_type, dimension, value, count
# metric_type values: source_patients, join_result, overall_null_rate, per_type_*
# dimension values: image_metadata, pathology_reports, clinical_outcomes, inner_join_all_3, ajcc_stage, etc.

# Extract source patient counts
source_counts = (
    data_quality
    .filter(F.col("metric_type") == "source_patients")
    .select(
        F.col("dimension").alias("source_name"),
        F.col("count").cast("long").alias("total_patients")
    )
)

# Get the inner join count (single value)
joined_count = (
    data_quality
    .filter(F.col("metric_type") == "join_result")
    .select(F.col("count").cast("long"))
    .first()[0]
)

# Get null rates
null_rates = (
    data_quality
    .filter(F.col("metric_type") == "overall_null_rate")
    .filter(F.col("dimension").isin("ajcc_stage", "histological_grade", "os_days"))
    .select("dimension", "value")
    .toPandas()
    .set_index("dimension")["value"]
    .to_dict()
)

dash_data_quality = (
    source_counts
    .withColumn("joined_patients", F.lit(joined_count))
    .withColumn("join_rate", F.round(F.lit(joined_count) / F.col("total_patients"), 4))
    .withColumn("null_rate_stage", F.lit(round(null_rates.get("ajcc_stage", 0.0), 4)))
    .withColumn("null_rate_grade", F.lit(round(null_rates.get("histological_grade", 0.0), 4)))
    .withColumn("null_rate_os", F.lit(round(null_rates.get("os_days", 0.0), 4)))
    .orderBy("source_name")
)

(
    dash_data_quality
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.tcga_gold.dash_data_quality")
)

spark.sql("""
    COMMENT ON TABLE workspace.tcga_gold.dash_data_quality IS
    'Dashboard: per-source join rates and null rates derived from tcga_silver.data_quality.'
""")

print("dash_data_quality written.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Table 4: `dash_demographics`
# MAGIC Patient counts and high-risk rates grouped by cancer type, gender, and race.

# COMMAND ----------

# Use risk_labels for cancer_type to avoid ambiguity
demo_base = (
    risk_labels
    .join(
        patients.select("patient_barcode", "gender", "race", "age"),
        on="patient_barcode",
        how="inner"
    )
)

dash_demographics = (
    demo_base
    .groupBy("cancer_type", "gender", "race")
    .agg(
        F.count("patient_barcode").alias("patient_count"),
        F.round(F.avg("age"), 2).alias("avg_age"),
        # high_risk_pct among non-censored patients only
        F.round(
            F.count(
                F.when(F.col("risk_label") == "high_risk", F.col("patient_barcode"))
            )
            / F.nullif(
                F.count(
                    F.when(F.col("risk_label").isin("high_risk", "low_risk"), F.col("patient_barcode"))
                ),
                F.lit(0)
            ) * 100,
            2
        ).alias("high_risk_pct"),
    )
    .orderBy("cancer_type", "gender", "race")
)

(
    dash_demographics
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.tcga_gold.dash_demographics")
)

spark.sql("""
    COMMENT ON TABLE workspace.tcga_gold.dash_demographics IS
    'Dashboard: patient counts, average age, and high-risk rate by cancer type, gender, and race. Censored patients excluded from high_risk_pct denominator.'
""")

print("dash_demographics written.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Table 5: `dash_predictions`
# MAGIC Flat patient-level prediction detail table for drill-down visuals in BI tools.

# COMMAND ----------

# predictions already has: patient_barcode, cancer_type, true_label, predicted_label,
#   prediction_confidence, model_name, split
# Pull age/gender/vital_status from patients, os_days from risk_labels
dash_predictions = (
    predictions
    .join(
        patients.select("patient_barcode", "age", "gender", "vital_status"),
        on="patient_barcode",
        how="inner"
    )
    .join(
        risk_labels.select("patient_barcode", "os_days"),
        on="patient_barcode",
        how="left"
    )
    .select(
        "patient_barcode",
        predictions["cancer_type"],
        "age",
        "gender",
        predictions["true_label"].alias("risk_label"),
        "predicted_label",
        "prediction_confidence",
        risk_labels["os_days"],
        "vital_status",
    )
    .orderBy("patient_barcode")
)

(
    dash_predictions
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.tcga_gold.dash_predictions")
)

spark.sql("""
    COMMENT ON TABLE workspace.tcga_gold.dash_predictions IS
    'Dashboard: patient-level prediction detail — true label, predicted label, confidence, and clinical covariates for drill-down analysis.'
""")

print("dash_predictions written.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 2: Validation

# COMMAND ----------

dashboard_tables = {
    "dash_risk_by_cancer":    "workspace.tcga_gold.dash_risk_by_cancer",
    "dash_model_performance": "workspace.tcga_gold.dash_model_performance",
    "dash_data_quality":      "workspace.tcga_gold.dash_data_quality",
    "dash_demographics":      "workspace.tcga_gold.dash_demographics",
    "dash_predictions":       "workspace.tcga_gold.dash_predictions",
}

print("=" * 55)
print("Dashboard Table Row Counts")
print("=" * 55)

for label, full_name in dashboard_tables.items():
    count = spark.table(full_name).count()
    print(f"  {label:<30} {count:>8,} rows")

print("=" * 55)
print()
print("Dashboard tables ready.")
print("Connect PowerBI/Databricks SQL to the SQL Warehouse to visualize.")