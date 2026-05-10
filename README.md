# TCGA Multimodal Cancer Risk Prediction Pipeline

An end-to-end data engineering pipeline that ingests cancer patient data from three separate sources — histopathology images, free-text pathology reports, and clinical survival records — cleans and joins them using medallion architecture on Databricks, extracts features from each modality, and trains a multimodal model to predict patient risk. The pipeline runs nightly on a schedule, and results feed a live PowerBI dashboard. Built as a demo for the Applied AI & Data Engineering lab at Florida Institute of Technology.

## Data Sources

Three sources, each in a different format, joined on the TCGA patient barcode (e.g. `TCGA-OR-A5JK`):

| Source | Format | Records | What it contains |
|---|---|---|---|
| TCGA-UT (HuggingFace) | WebDataset .tar + CSV | 271,710 tiles / 7,175 patients | H&E stained histopathology image patches, 256x256 px, 31 cancer types |
| TCGA-Reports (Mendeley) | Zipped CSV | 9,523 reports | OCR'd and cleaned pathology reports — free text describing tumor characteristics |
| TCGA-CDR (GDC) | Excel (.xlsx) | 11,160 patients | Clinical outcomes — survival time, vital status, stage, grade, demographics |

After inner-joining all three on patient barcode, 6,242 patients remain. The image embeddings from the Colab step cover all 7,175 TCGA-UT patients, so every joined patient has embeddings.

## Pipeline Architecture

The pipeline follows medallion architecture — bronze holds raw data exactly as ingested, silver is where the cleaning and joining happens, and gold contains the feature-engineered tables ready for the model and dashboard. Each layer is stored as Delta tables in Databricks Unity Catalog.

```
Raw Sources --> Bronze (untouched) --> Silver (cleaned, joined) --> Gold (features, risk labels) --> Model --> Dashboard
```

## What Each Notebook Does

**00_setup.py** creates the three medallion schemas (`tcga_bronze`, `tcga_silver`, `tcga_gold`) in Unity Catalog. Run once.

**01_ingest_bronze.py** downloads all three data sources into raw Delta tables. HuggingFace is the only URL reachable from Databricks serverless compute — Mendeley and GDC are blocked by the serverless networking restrictions, so those files get downloaded locally and uploaded to a Unity Catalog Volume via `databricks fs cp`. The notebook reads from the Volume and writes bronze tables.

**02_bronze_to_silver.py** is the heaviest notebook. It builds a cancer type mapping table (e.g. BRCA maps to Breast_invasive_carcinoma — COAD and READ both map to Colon_Rectum_adenocarcinoma since they were merged in the TCGA-UT dataset). It inner-joins all three sources on patient barcode, standardizes column names, handles TCGA sentinel values like `[Not Available]`, computes per-patient patch counts and report word counts. It also produces a data quality table tracking join coverage and null rates per cancer type.

**03_silver_to_gold.py** derives risk labels by computing per-cancer-type median overall survival (using only deceased patients so the survival time is fully observed) and labeling patients above/below the median. Censored patients with short follow-up get flagged separately. For feature engineering, it runs TF-IDF on pathology report text (200 features), one-hot encodes clinical categoricals (gender, top-10 cancer types, simplified AJCC stage), scales numericals (age, report word count, patch count), and joins in 512-dim ABMIL attention-pooled image embeddings from the Colab step. The output is a feature matrix with zero nulls.

**04_model_inference.py** trains three models on the fused feature vector: logistic regression, random forest, and a 3-layer PyTorch MLP (734 to 256 to 128 to 2). It evaluates each on a held-out test set, picks the winner by balanced accuracy, and saves predictions and per-model metrics to gold tables. The PyTorch install on serverless required upgrading `typing_extensions` and using the CPU-only torch wheel — one of those runtime compatibility issues you just have to work through.

**05_export_dashboard.py** pre-aggregates gold tables into five dashboard-ready tables — risk distribution by cancer type, model performance metrics, data quality summary, demographic breakdowns, and a patient-level prediction explorer. PowerBI reads these via the SQL Warehouse connector.

## Image Embeddings (Google Colab)

Serverless Databricks has no GPU, so image feature extraction and aggregation run in a separate Colab notebook with a free T4 GPU. It downloads all 51 tar shards (39 train + 6 valid + 6 test) from HuggingFace containing ~250K histopathology tiles across 7,175 patients, runs each tile through a pretrained ResNet-18 (with the classification head removed), and saves per-patient patch-level embeddings to disk.

Instead of naive mean pooling, the script trains a **Gated Attention MIL** (Multiple Instance Learning) network that learns which patches are most informative for each patient. The attention mechanism assigns per-patch weights — so a patient with 90% benign tissue and 10% aggressive tumor gets a representation dominated by the tumor patches, not washed out by the background. The trained model outputs a single 512-dim attention-pooled vector per patient.

Risk labels are derived directly in Colab from the TCGA-CDR clinical data (per-cancer-type median OS among deceased patients) so the script is self-contained. The exported CSV gets uploaded to Databricks and joined into the gold feature table. The script is in `colab/image_embeddings.py`.

## The Model

The approach is late fusion — ABMIL attention-pooled image embeddings, TF-IDF vectors from pathology text, and one-hot encoded clinical fields all get concatenated into a single feature vector per patient. Three classifiers are compared on a held-out test set:

| Model | Accuracy | Balanced Accuracy |
|---|---|---|
| Logistic Regression | TBD | TBD |
| MLP (PyTorch, 3-layer) | TBD | TBD |
| Random Forest | TBD | TBD |

*Results pending rerun after fixing data leakage (survival time was previously included as a feature) and upgrading from mean pooling to attention-based MIL.*

## Automation

The pipeline is orchestrated as a Databricks Workflow — a DAG of five notebook tasks that run in sequence. Scheduled for 2 AM EST nightly, though it ships paused by default so it doesn't burn credits. If any step fails, downstream tasks don't execute. The workflow config is in `workflow/workflow_config.json` and was deployed via `databricks jobs create`.

## Dashboard

PowerBI connects to the Databricks SQL Warehouse via the native connector. Five gold tables feed the dashboard: risk distribution by cancer type, model performance metrics, data quality summary, demographic breakdowns, and a patient-level prediction explorer. The connection is live — when the pipeline runs, the dashboard updates automatically.

Databricks Genie is also set up for natural language queries against the gold tables.

## Tech Stack

- **Databricks** — serverless compute, Unity Catalog, Delta Lake, Workflows
- **Python** — pandas, PySpark, scikit-learn, PyTorch
- **Google Colab** — ResNet-18 feature extraction + ABMIL training on T4 GPU
- **PowerBI** — live dashboard via SQL Warehouse connector
- **Databricks CLI** — notebook deployment, file uploads to Volumes

## Repo Structure

```
tcga-risk-pipeline/
├── README.md
├── notebooks/
│   ├── 00_setup.py
│   ├── 01_ingest_bronze.py
│   ├── 02_bronze_to_silver.py
│   ├── 03_silver_to_gold.py
│   ├── 04_model_inference.py
│   └── 05_export_dashboard.py
├── colab/
│   └── image_embeddings.py
└── workflow/
    └── workflow_config.json
```

## Data Sources & References

- Komura et al. (2022). "Universal encoding of pan-cancer histology by deep texture representations." *Cell Reports*. — the TCGA-UT image dataset on HuggingFace.
- Kefeli & Tatonetti (2024). "TCGA-Reports: A machine-readable pathology report resource for benchmarking text-based AI models." *Patterns*. — the pathology reports dataset on Mendeley.
- Liu et al. (2018). "An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive High-Quality Survival Outcome Analytics." *Cell*. — the TCGA-CDR clinical outcomes table from GDC.
