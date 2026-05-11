# TCGA Multimodal Cancer Risk Prediction Pipeline

An end-to-end data engineering pipeline that ingests cancer patient data from three separate sources — histopathology images, free-text pathology reports, and clinical survival records — cleans and joins them using medallion architecture on Databricks, extracts features from each modality, and trains a multimodal model to predict patient risk. Results are visualized in an interactive Streamlit dashboard. Built as a demo for the Applied AI & Data Engineering lab at Florida Institute of Technology.

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
Raw Sources --> Bronze (untouched) --> Silver (cleaned, joined) --> Gold (features, risk labels) --> Model --> Streamlit Dashboard
```

## What Each Notebook Does

**00_setup.py** creates the three medallion schemas (`tcga_bronze`, `tcga_silver`, `tcga_gold`) in Unity Catalog. Run once.

**01_ingest_bronze.py** downloads all three data sources into raw Delta tables. HuggingFace is the only URL reachable from Databricks serverless compute — Mendeley and GDC are blocked by the serverless networking restrictions, so those files get downloaded locally and uploaded to a Unity Catalog Volume via `databricks fs cp`. The notebook reads from the Volume and writes bronze tables.

**02_bronze_to_silver.py** is the heaviest notebook. It builds a cancer type mapping table (e.g. BRCA maps to Breast_invasive_carcinoma — COAD and READ both map to Colon_Rectum_adenocarcinoma since they were merged in the TCGA-UT dataset). It inner-joins all three sources on patient barcode, standardizes column names, handles TCGA sentinel values like `[Not Available]`, computes per-patient patch counts and report word counts. It also produces a data quality table tracking join coverage and null rates per cancer type.

**03_silver_to_gold.py** derives risk labels using a tertile split — for each cancer type, it computes the 33rd and 67th percentile of overall survival among deceased patients. Patients who died below p33 are `high_risk`, those above p67 are `low_risk`, and the ambiguous middle third is dropped from training. This gives cleaner class separation than a simple median cut. For feature engineering, it runs TF-IDF on pathology report text (200 features → 30 via PCA), one-hot encodes clinical categoricals (gender, top-10 cancer types, simplified AJCC stage), scales numericals (age, report word count, patch count), and joins in ABMIL attention-pooled image embeddings (512-dim → 50 via PCA) from the Colab step. PCA reduces noise given the ~4K training samples. The output is a feature matrix with zero nulls.

**04_model_inference.py** trains three global models on the fused feature vector: logistic regression, LightGBM, and a 3-layer PyTorch MLP (input → 64 → 32 → 2 with BatchNorm and dropout). It also trains per-cancer-type LightGBM models for the six largest cancer types (GBM, LUSC, HNSC, KIRC, LUAD, BLCA) to capture cancer-specific risk patterns. All models are evaluated on stratified held-out test sets using balanced accuracy as the primary metric. Predictions and per-model metrics are saved to gold tables.

**05_export_dashboard.py** pre-aggregates gold tables into five dashboard-ready tables — risk distribution by cancer type, model performance metrics, data quality summary, demographic breakdowns, and a patient-level prediction explorer. These tables are exported as CSVs to feed the Streamlit dashboard.

## Streamlit Dashboard

The dashboard (`app/streamlit_app.py`) reads six exported CSVs from the pipeline and provides three interactive views:

- **Patient Explorer** — select any patient barcode to see their risk label, clinical profile, per-model predictions with confidence scores, and a radar chart of their feature profile (age, report length, patch count, image/text PCA components)
- **Cohort Analytics** — risk distribution by cancer type, overall survival histograms, AJCC stage breakdown, age distribution by risk group, model performance heatmap across all metrics, and a 3D scatter of image + text embedding space
- **Pipeline** — visual overview of the medallion architecture, feature composition pie chart, and data quality summary

Run with:
```bash
cd app && pip install -r requirements.txt && streamlit run streamlit_app.py
```

## Image Embeddings (Google Colab)

Serverless Databricks has no GPU, so image feature extraction and aggregation run in a separate Colab notebook with a free T4 GPU. It downloads all 51 tar shards (39 train + 6 valid + 6 test) from HuggingFace containing ~250K histopathology tiles across 7,175 patients, runs each tile through a pretrained ResNet-18 (with the classification head removed), and saves per-patient patch-level embeddings to disk.

Instead of naive mean pooling, the script trains a **Gated Attention MIL** (Multiple Instance Learning) network that learns which patches are most informative for each patient. The attention mechanism assigns per-patch weights — so a patient with 90% benign tissue and 10% aggressive tumor gets a representation dominated by the tumor patches, not washed out by the background. The trained model outputs a single 512-dim attention-pooled vector per patient.

Risk labels are derived directly in Colab from the TCGA-CDR clinical data (per-cancer-type median OS among deceased patients) so the script is self-contained. The exported CSV gets uploaded to Databricks and joined into the gold feature table. The script is in `colab/image_embeddings.py`.

## The Model

The approach is late fusion — ABMIL attention-pooled image embeddings (PCA'd to 50 dims), TF-IDF vectors from pathology text (PCA'd to 30 dims), and one-hot encoded clinical fields all get concatenated into a single feature vector per patient.

### Global Models

Three classifiers are compared on a held-out test set:

| Model | Accuracy | Balanced Accuracy |
|---|---|---|
| Logistic Regression | 71.0% | 69.5% |
| MLP (PyTorch, 3-layer) | 72.5% | 65.1% |
| LightGBM | 79.6% | 58.7% |

Logistic regression wins on balanced accuracy — LightGBM has higher raw accuracy but leans toward the majority class. The MLP (64→32→2 with BatchNorm) lands in between.

### Per-Cancer-Type Models

Dedicated LightGBM models trained on individual cancer types capture disease-specific risk patterns:

| Cancer Type | Accuracy | Balanced Accuracy |
|---|---|---|
| GBM (Glioblastoma) | 74.1% | **72.5%** |
| LUSC (Lung Squamous) | 77.5% | 67.6% |
| KIRC (Kidney Clear Cell) | 80.9% | 63.6% |
| HNSC (Head & Neck) | 73.2% | 58.5% |
| BLCA (Bladder) | 68.6% | 48.8% |
| LUAD (Lung Adeno) | 59.5% | 43.9% |

The GBM-specific model hits 72.5% balanced accuracy — the best result in the pipeline — because GBM has the most balanced class split and distinctive histopathology. Cancer types with severe class imbalance (BLCA, LUAD) don't benefit from per-type training.

These numbers are honest: no survival time leakage (os_days excluded from features), tertile labels derived only from deceased patients, no test set contamination. The model predicts risk from pathology images (ABMIL attention-pooled), report text (TF-IDF), and clinical staging alone.

## Automation

The pipeline is orchestrated as a Databricks Workflow — a DAG of five notebook tasks that run in sequence. Scheduled for 7 AM EST daily, though it ships paused by default so it doesn't burn credits. If any step fails, downstream tasks don't execute. The workflow config is in `workflow/workflow_config.json` and was deployed via `databricks jobs create`.

## Tech Stack

- **Databricks** — serverless compute, Unity Catalog, Delta Lake, Workflows
- **Python** — pandas, PySpark, scikit-learn, PyTorch, LightGBM
- **Google Colab** — ResNet-18 feature extraction + ABMIL training on T4 GPU
- **Streamlit** — interactive dashboard with Plotly visualizations
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
├── app/
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── data/
│       ├── patient_features.csv
│       ├── risk_labels.csv
│       ├── model_metrics.csv
│       ├── model_predictions.csv
│       ├── feature_metadata.csv
│       └── data_quality.csv
└── workflow/
    └── workflow_config.json
```

## Data Sources & References

- Komura et al. (2022). "Universal encoding of pan-cancer histology by deep texture representations." *Cell Reports*. — the TCGA-UT image dataset on HuggingFace.
- Kefeli & Tatonetti (2024). "TCGA-Reports: A machine-readable pathology report resource for benchmarking text-based AI models." *Patterns*. — the pathology reports dataset on Mendeley.
- Liu et al. (2018). "An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive High-Quality Survival Outcome Analytics." *Cell*. — the TCGA-CDR clinical outcomes table from GDC.
