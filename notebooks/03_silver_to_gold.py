# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Silver to Gold
# MAGIC
# MAGIC Transforms `workspace.tcga_silver.patients` into gold-layer tables:
# MAGIC
# MAGIC | Gold Table | Description |
# MAGIC |---|---|
# MAGIC | `workspace.tcga_gold.risk_labels` | Per-patient risk label derived from cancer-type-specific median OS |
# MAGIC | `workspace.tcga_gold.patient_features` | Full feature matrix (numerical + categorical + TF-IDF) |
# MAGIC | `workspace.tcga_gold.feature_metadata` | Feature registry with name, type, and description |
# MAGIC
# MAGIC **Compute:** Serverless (no GPU). All sklearn work runs on the driver — 6,242 rows fits comfortably in memory.

# COMMAND ----------

# MAGIC %pip install scikit-learn -q

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window
import pandas as pd
import numpy as np

print("Imports loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 0 — Load Image Embeddings into Bronze
# MAGIC
# MAGIC Pre-computed 512-dim image embeddings per patient, uploaded to the Volume.

# COMMAND ----------

# Read embeddings CSV from the Volume and persist as a bronze table
df_emb = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("/Volumes/workspace/tcga_bronze/raw_files/patient_image_embeddings.csv")
)

(
    df_emb.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.tcga_bronze.image_embeddings")
)

print(f"workspace.tcga_bronze.image_embeddings: {df_emb.count()} rows, {len(df_emb.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 1 — Risk Label Derivation
# MAGIC
# MAGIC For each cancer type we compute the **33rd and 67th percentile of `os_days`** using only deceased patients.
# MAGIC Tertile split gives cleaner class separation than a median cut — patients near the boundary are dropped.
# MAGIC
# MAGIC Labels:
# MAGIC - `high_risk`  — `os_event = 1` AND `os_days < p33` (bottom third, died quickly)
# MAGIC - `low_risk`   — `os_days >= p67` (top third, long survivors)
# MAGIC - `ambiguous`  — everyone in the middle third (dropped from training)

# COMMAND ----------

silver = spark.table("workspace.tcga_silver.patients")
print(f"Silver row count: {silver.count():,}")
silver.printSchema()

# COMMAND ----------

# Compute per-cancer-type tertile thresholds using only deceased patients (os_event = 1)
deceased = silver.filter(F.col("os_event") == 1)

tertiles = (
    deceased
    .groupBy("cancer_type")
    .agg(
        F.percentile_approx("os_days", 0.33).alias("p33_os"),
        F.percentile_approx("os_days", 0.67).alias("p67_os"),
    )
)

print("OS tertile thresholds per cancer type (deceased patients only):")
tertiles.orderBy("cancer_type").show(40, truncate=False)

# COMMAND ----------

# Join tertiles back to full patient table
patients_with_tertiles = silver.join(tertiles, on="cancer_type", how="left")

# Derive risk label using tertile split
risk_labels = patients_with_tertiles.withColumn(
    "risk_label",
    F.when(
        (F.col("os_event") == 1) & (F.col("os_days") < F.col("p33_os")),
        F.lit("high_risk")
    ).when(
        F.col("os_days") >= F.col("p67_os"),
        F.lit("low_risk")
    ).otherwise(
        F.lit("ambiguous")
    )
).select(
    "patient_barcode",
    "cancer_type",
    "cancer_type_abbrev",
    "os_days",
    "os_event",
    "p33_os",
    "p67_os",
    "risk_label"
)

print("Risk label distribution:")
risk_labels.groupBy("risk_label").count().orderBy("risk_label").show()

# COMMAND ----------

# Persist risk_labels gold table
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.tcga_gold")

(
    risk_labels
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.tcga_gold.risk_labels")
)

print("workspace.tcga_gold.risk_labels written successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2 — Feature Engineering
# MAGIC
# MAGIC All feature construction runs in **pandas / sklearn on the driver**.
# MAGIC
# MAGIC Feature groups:
# MAGIC
# MAGIC | Group | Features | Transform |
# MAGIC |---|---|---|
# MAGIC | Numerical | `age`, `os_days`, `report_word_count`, `patch_count` | Median impute → StandardScaler |
# MAGIC | Categorical | `gender`, `cancer_type_abbrev` (top-10), `ajcc_stage_simple` | One-hot encode (drop_first=False) |
# MAGIC | Text | `report_text` | TF-IDF, max_features=200 |
# MAGIC | Image | 512 `img_emb_*` columns | Pre-computed embeddings, zero-filled for missing patients |

# COMMAND ----------

# Pull joined data to the driver
cols_needed = [
    "patient_barcode", "cancer_type", "cancer_type_abbrev",
    "age", "os_days", "report_word_count", "patch_count",
    "gender", "ajcc_stage", "report_text"
]

gold_rl = spark.table("workspace.tcga_gold.risk_labels").select(
    "patient_barcode", "risk_label"
)

silver_subset = silver.select(cols_needed)

joined_spark = silver_subset.join(gold_rl, on="patient_barcode", how="inner")

df = joined_spark.toPandas()
print(f"Pulled {len(df):,} rows to driver.")

# COMMAND ----------

# INNER JOIN with image embeddings — only keep patients with real embeddings
df_emb_pd = spark.table("workspace.tcga_bronze.image_embeddings").toPandas()
emb_cols = [c for c in df_emb_pd.columns if c.startswith("img_emb_")]

before = len(df)
df = df.merge(df_emb_pd, on="patient_barcode", how="inner").reset_index(drop=True)
print(f"After inner join with image embeddings: {len(df):,} rows (dropped {before - len(df)} without embeddings)")

# COMMAND ----------

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# ------------------------------------------------------------------
# 2a. Numerical features
# ------------------------------------------------------------------
num_cols = ["age", "report_word_count", "patch_count"]

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

df_num_imputed = imputer.fit_transform(df[num_cols])
df_num_scaled = scaler.fit_transform(df_num_imputed)

df_numerical = pd.DataFrame(
    df_num_scaled,
    columns=[f"num_{c}" for c in num_cols],
    index=df.index
)

print(f"Numerical features shape: {df_numerical.shape}")

# ------------------------------------------------------------------
# 2b. Categorical features
# ------------------------------------------------------------------

# gender — fill nulls
df["gender_clean"] = df["gender"].fillna("Unknown")

# cancer_type_abbrev — top 10 by count, rest → "OTHER"
top10_abbrev = df["cancer_type_abbrev"].value_counts().head(10).index.tolist()
df["cancer_type_abbrev_grouped"] = df["cancer_type_abbrev"].apply(
    lambda x: x if x in top10_abbrev else "OTHER"
).fillna("OTHER")

# ajcc_stage simplification
def simplify_stage(raw):
    if pd.isna(raw):
        return "Unknown"
    s = str(raw).strip()
    if s.startswith("Stage IV"):
        return "Stage_IV"
    if s.startswith("Stage III"):
        return "Stage_III"
    if s.startswith("Stage II"):
        return "Stage_II"
    if s.startswith("Stage I"):
        return "Stage_I"
    return "Unknown"

df["ajcc_stage_simple"] = df["ajcc_stage"].apply(simplify_stage)

cat_cols = {
    "gender_clean": None,
    "cancer_type_abbrev_grouped": None,
    "ajcc_stage_simple": None,
}

# sparse_output was renamed from sparse in sklearn 1.2
import sklearn
if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 2):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
else:
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
cat_array = ohe.fit_transform(df[list(cat_cols.keys())])
cat_feature_names = ohe.get_feature_names_out(list(cat_cols.keys()))

df_categorical = pd.DataFrame(
    cat_array,
    columns=cat_feature_names,
    index=df.index
)

print(f"Categorical features shape: {df_categorical.shape}")
print(f"Categorical feature names ({len(cat_feature_names)}): {list(cat_feature_names)}")

# ------------------------------------------------------------------
# 2c. TF-IDF on report_text
# ------------------------------------------------------------------
report_text_filled = df["report_text"].fillna("")

tfidf = TfidfVectorizer(max_features=200, sublinear_tf=True, min_df=2)
tfidf_array = tfidf.fit_transform(report_text_filled).toarray()

# PCA: 200 TF-IDF features → 30 components
pca_tfidf = PCA(n_components=30, random_state=42)
tfidf_pca = pca_tfidf.fit_transform(tfidf_array)
tfidf_feature_names = [f"tfidf_pc_{i}" for i in range(30)]
print(f"TF-IDF PCA: {tfidf_array.shape[1]} → {tfidf_pca.shape[1]} (variance retained: {pca_tfidf.explained_variance_ratio_.sum():.2%})")

df_tfidf = pd.DataFrame(
    tfidf_pca,
    columns=tfidf_feature_names,
    index=df.index
)

print(f"TF-IDF features shape: {df_tfidf.shape}")

# ------------------------------------------------------------------
# 2d. Image embeddings — PCA 512 → 50
# ------------------------------------------------------------------
emb_array = df[emb_cols].values

pca_img = PCA(n_components=50, random_state=42)
emb_pca = pca_img.fit_transform(emb_array)
emb_pca_names = [f"img_pc_{i}" for i in range(50)]
print(f"Image PCA: {emb_array.shape[1]} → {emb_pca.shape[1]} (variance retained: {pca_img.explained_variance_ratio_.sum():.2%})")

df_image_emb = pd.DataFrame(emb_pca, columns=emb_pca_names, index=df.index).reset_index(drop=True)
print(f"Image embedding features shape: {df_image_emb.shape}")

# COMMAND ----------

# Combine into final feature matrix
df_features = pd.concat(
    [
        df[["patient_barcode", "risk_label"]].reset_index(drop=True),
        df_numerical.reset_index(drop=True),
        df_categorical.reset_index(drop=True),
        df_tfidf.reset_index(drop=True),
        df_image_emb,
    ],
    axis=1
)

print(f"Combined feature matrix shape: {df_features.shape}")
print(f"Columns: patient_barcode, risk_label + {df_features.shape[1] - 2} feature columns")

# COMMAND ----------

# Write patient_features gold table
spark_features = spark.createDataFrame(df_features)

(
    spark_features
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.tcga_gold.patient_features")
)

print("workspace.tcga_gold.patient_features written successfully.")

# COMMAND ----------

# Build feature_metadata table
metadata_rows = []

for c in num_cols:
    metadata_rows.append({
        "feature_name": f"num_{c}",
        "feature_type": "numerical",
        "description": f"Median-imputed and standard-scaled '{c}'"
    })

for c in cat_feature_names:
    metadata_rows.append({
        "feature_name": c,
        "feature_type": "categorical",
        "description": f"One-hot encoded categorical feature: {c}"
    })

for t in tfidf_feature_names:
    metadata_rows.append({
        "feature_name": t,
        "feature_type": "tfidf_pca",
        "description": f"PCA component {t.replace('tfidf_pc_', '')} of TF-IDF (200 → 30)"
    })

for c in emb_pca_names:
    metadata_rows.append({
        "feature_name": c,
        "feature_type": "image_pca",
        "description": f"PCA component {c.replace('img_pc_', '')} of ABMIL attention-pooled embeddings (512 → 50)"
    })

df_meta = pd.DataFrame(metadata_rows)
spark_meta = spark.createDataFrame(df_meta)

(
    spark_meta
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.tcga_gold.feature_metadata")
)

print(f"workspace.tcga_gold.feature_metadata written — {len(metadata_rows)} features registered.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3 — Validation

# COMMAND ----------

# 3a. Feature matrix shape
print("=" * 60)
print("VALIDATION REPORT")
print("=" * 60)
print(f"\n[1] Feature matrix shape: {df_features.shape}")
print(f"    Rows    : {df_features.shape[0]:,}")
print(f"    Columns : {df_features.shape[1]:,}  (2 id + {df_features.shape[1]-2} features)")

# COMMAND ----------

# 3b. Risk label distribution (exclude censored)
labeled = df_features[df_features["risk_label"].isin(["high_risk", "low_risk"])]
dist = labeled["risk_label"].value_counts()
print("\n[2] Risk label distribution (censored excluded):")
for label, cnt in dist.items():
    pct = cnt / len(labeled) * 100
    print(f"    {label:<12} : {cnt:>5,}  ({pct:.1f}%)")
print(f"    {'ambiguous':<12} : {(df_features['risk_label'] == 'ambiguous').sum():>5,}  (middle tertile, excluded from model)")

# COMMAND ----------

# 3c. Sample of 5 rows — patient_barcode, risk_label, first 5 feature columns
first5_feat_cols = [c for c in df_features.columns if c not in ("patient_barcode", "risk_label")][:5]
sample_cols = ["patient_barcode", "risk_label"] + first5_feat_cols

print("\n[3] Sample 5 rows (patient_barcode, risk_label, first 5 feature columns):")
print(df_features[sample_cols].head(5).to_string(index=False))

# COMMAND ----------

# 3d. Null count across all feature columns
feat_cols_only = [c for c in df_features.columns if c not in ("patient_barcode", "risk_label")]
null_counts = df_features[feat_cols_only].isnull().sum()
total_nulls = null_counts.sum()

print(f"\n[4] Null count across all {len(feat_cols_only)} feature columns: {total_nulls}")
if total_nulls == 0:
    print("    PASS — zero nulls in feature matrix.")
else:
    print("    FAIL — nulls detected:")
    print(null_counts[null_counts > 0])

print("\n" + "=" * 60)
print("Gold layer pipeline complete.")
print("=" * 60)