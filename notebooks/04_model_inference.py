# Databricks notebook source
# MAGIC %md
# MAGIC # ## 04 — Model Inference
# MAGIC # Train Logistic Regression, Random Forest, and a PyTorch MLP on TCGA patient features,
# MAGIC # evaluate per-model metrics, and persist predictions + metrics to Gold tables.

# COMMAND ----------

# MAGIC %pip install typing_extensions --upgrade -q

# COMMAND ----------

# MAGIC %pip install torch --index-url https://download.pytorch.org/whl/cpu -q

# COMMAND ----------

# MAGIC %pip install scikit-learn -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC # ### Section 1 — Load Data

# COMMAND ----------

print("Loading patient_features from workspace.tcga_gold.patient_features ...")
features_df = spark.table("workspace.tcga_gold.patient_features").toPandas()
print(f"  Raw rows: {len(features_df):,}")

# Drop censored patients — they carry no usable risk label
features_df = features_df[features_df["risk_label"] != "censored"].reset_index(drop=True)
print(f"  After removing censored: {len(features_df):,}")

# COMMAND ----------

print("Loading split and cancer type info from workspace.tcga_silver.patients ...")
silver_df = (
    spark.table("workspace.tcga_silver.patients")
    .select("patient_barcode", "split_internal", "cancer_type_abbrev")
    .toPandas()
)

# Join on patient_barcode
merged_df = features_df.merge(silver_df, on="patient_barcode", how="inner")
print(f"  Rows after join: {len(merged_df):,}")

# Try split_internal first; if test set is empty, fall back to random 80/20
train_mask = merged_df["split_internal"] == "train"
test_mask  = merged_df["split_internal"].isin(["valid", "test"])

if test_mask.sum() == 0:
    print("  WARNING: split_internal produced empty test set — falling back to random 80/20 split")
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        merged_df.index, test_size=0.2, random_state=42,
        stratify=merged_df["risk_label"]
    )
    merged_df["split_used"] = "train"
    merged_df.loc[test_idx, "split_used"] = "test"
    train_mask = merged_df["split_used"] == "train"
    test_mask  = merged_df["split_used"] == "test"
else:
    merged_df["split_used"] = "test"
    merged_df.loc[train_mask, "split_used"] = "train"

print(f"  Split distribution: {merged_df['split_used'].value_counts().to_dict()}")

# COMMAND ----------

train_df = merged_df[train_mask].reset_index(drop=True)
test_df  = merged_df[test_mask].reset_index(drop=True)

# Feature matrix — drop non-feature columns
DROP_COLS = ["patient_barcode", "risk_label", "split_internal", "cancer_type_abbrev", "split_used"]
feature_cols = [c for c in merged_df.columns if c not in DROP_COLS]

X_train = train_df[feature_cols].astype(float)
X_test  = test_df[feature_cols].astype(float)

# Label encoding: high_risk → 1, low_risk → 0
label_map = {"high_risk": 1, "low_risk": 0}
y_train = train_df["risk_label"].map(label_map)
y_test  = test_df["risk_label"].map(label_map)

print(f"Train size : {len(X_train):,}  (high_risk={y_train.sum():,}, low_risk={(y_train==0).sum():,})")
print(f"Test  size : {len(X_test):,}  (high_risk={y_test.sum():,}, low_risk={(y_test==0).sum():,})")
print(f"Features   : {len(feature_cols)}")

# COMMAND ----------

# MAGIC %md
# MAGIC # ### Section 2 — Train Models

# COMMAND ----------

print("Training LogisticRegression ...")
lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
lr_model.fit(X_train, y_train)
print("  LogisticRegression training complete.")

# COMMAND ----------

print("Training RandomForestClassifier ...")
rf_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)
print("  RandomForestClassifier training complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC # #### PyTorch MLP (3-layer, CPU)

# COMMAND ----------

# Define the MLP
class RiskMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)

# Prepare tensors
device = torch.device("cpu")
input_dim = X_train.shape[1]

X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.long)
X_test_t  = torch.tensor(X_test.values, dtype=torch.float32)
y_test_t  = torch.tensor(y_test.values, dtype=torch.long)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Handle class imbalance with weighted loss
class_counts = np.bincount(y_train.values)
class_weights = torch.tensor(
    [1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32
)
class_weights = class_weights / class_weights.sum() * 2  # normalize

mlp_model = RiskMLP(input_dim).to(device)
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# Train
print(f"Training PyTorch MLP (input={input_dim} → 256 → 128 → 2) for 50 epochs ...")
for epoch in range(50):
    mlp_model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = mlp_model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(y_batch)
    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / len(train_dataset)
        print(f"  Epoch {epoch+1:>2}/50  loss={avg_loss:.4f}")

mlp_model.eval()
print("  MLP training complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC # ### Section 3 — Evaluate

# COMMAND ----------

def evaluate_model(model, model_name, X, y, split_label, cancer_type_series=None):
    """
    Evaluate a model. Works for both sklearn models and the string "MLP".
    Returns a dict of scalar metrics.
    """
    preds = model.predict(X)
    acc    = accuracy_score(y, preds)
    bal_acc = balanced_accuracy_score(y, preds)

    print(f"\n{'='*60}")
    print(f"Model : {model_name}  |  Split : {split_label}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Balanced Accuracy : {bal_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, preds, target_names=["low_risk", "high_risk"]))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(y, preds))

    # Per-cancer-type accuracy for top-5 types
    if cancer_type_series is not None:
        results_df = pd.DataFrame({
            "cancer_type": cancer_type_series.values,
            "true":  y.values,
            "pred":  preds,
        })
        top5_types = results_df["cancer_type"].value_counts().head(5).index.tolist()
        print(f"\nPer-cancer-type accuracy (top 5 by count):")
        for ct in top5_types:
            sub = results_df[results_df["cancer_type"] == ct]
            ct_acc = accuracy_score(sub["true"], sub["pred"])
            print(f"  {ct:<12s} n={len(sub):>4,}  acc={ct_acc:.4f}")

    # Extract per-class metrics from classification_report dict
    report = classification_report(y, preds, target_names=["low_risk", "high_risk"], output_dict=True)
    return {
        "accuracy":           acc,
        "balanced_accuracy":  bal_acc,
        "precision_high_risk": report["high_risk"]["precision"],
        "recall_high_risk":    report["high_risk"]["recall"],
        "f1_high_risk":        report["high_risk"]["f1-score"],
        "precision_low_risk":  report["low_risk"]["precision"],
        "recall_low_risk":     report["low_risk"]["recall"],
        "f1_low_risk":         report["low_risk"]["f1-score"],
    }

# COMMAND ----------

# Wrap MLP in an sklearn-like interface for evaluate_model and build_predictions_df
class MLPWrapper:
    """Thin wrapper so the MLP can be used like an sklearn model."""
    def __init__(self, torch_model, device):
        self.torch_model = torch_model
        self.device = device
        self.classes_ = np.array([0, 1])

    def predict(self, X):
        self.torch_model.eval()
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.torch_model(X_t)
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        self.torch_model.eval()
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.torch_model(X_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            return proba

mlp_wrapper = MLPWrapper(mlp_model, device)

# COMMAND ----------

print("Evaluating LogisticRegression on train set ...")
lr_train_metrics = evaluate_model(
    lr_model, "LogisticRegression", X_train, y_train, "train"
)

print("\nEvaluating LogisticRegression on test set ...")
lr_test_metrics = evaluate_model(
    lr_model, "LogisticRegression", X_test, y_test, "test",
    cancer_type_series=test_df["cancer_type_abbrev"]
)

# COMMAND ----------

print("Evaluating RandomForestClassifier on train set ...")
rf_train_metrics = evaluate_model(
    rf_model, "RandomForestClassifier", X_train, y_train, "train"
)

print("\nEvaluating RandomForestClassifier on test set ...")
rf_test_metrics = evaluate_model(
    rf_model, "RandomForestClassifier", X_test, y_test, "test",
    cancer_type_series=test_df["cancer_type_abbrev"]
)

# COMMAND ----------

print("Evaluating MLP on train set ...")
mlp_train_metrics = evaluate_model(
    mlp_wrapper, "MLP_3Layer", X_train, y_train, "train"
)

print("\nEvaluating MLP on test set ...")
mlp_test_metrics = evaluate_model(
    mlp_wrapper, "MLP_3Layer", X_test, y_test, "test",
    cancer_type_series=test_df["cancer_type_abbrev"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC # ### Section 4 — Save Predictions

# COMMAND ----------

# Pick better model by balanced_accuracy on test set
all_test_metrics = {
    "LogisticRegression": lr_test_metrics,
    "RandomForestClassifier": rf_test_metrics,
    "MLP_3Layer": mlp_test_metrics,
}
all_models = {
    "LogisticRegression": lr_model,
    "RandomForestClassifier": rf_model,
    "MLP_3Layer": mlp_wrapper,
}

best_model_name = max(all_test_metrics, key=lambda k: all_test_metrics[k]["balanced_accuracy"])
best_model = all_models[best_model_name]

print(f"Best model (by balanced_accuracy on test): {best_model_name}")

# COMMAND ----------

def build_predictions_df(model, model_name, X, barcode_series, cancer_type_series, true_y, split_label):
    """
    Returns a pandas DataFrame with prediction columns for one split.
    """
    preds      = model.predict(X)
    proba      = model.predict_proba(X)          # shape (n, 2); classes: [0, 1]
    inv_label  = {1: "high_risk", 0: "low_risk"}

    # Confidence = probability of the predicted class
    class_idx  = {cls: i for i, cls in enumerate(model.classes_)}
    confidence = np.array([
        proba[i, class_idx[pred]] for i, pred in enumerate(preds)
    ])

    return pd.DataFrame({
        "patient_barcode":       barcode_series.values,
        "cancer_type":           cancer_type_series.values,
        "true_label":            true_y.map(inv_label).values,
        "predicted_label":       [inv_label[p] for p in preds],
        "prediction_confidence": confidence,
        "model_name":            model_name,
        "split":                 split_label,
    })

# COMMAND ----------

print("Building prediction DataFrames for all patients ...")

train_preds_pd = build_predictions_df(
    best_model, best_model_name,
    X_train,
    train_df["patient_barcode"],
    train_df["cancer_type_abbrev"],
    y_train,
    "train",
)

test_preds_pd = build_predictions_df(
    best_model, best_model_name,
    X_test,
    test_df["patient_barcode"],
    test_df["cancer_type_abbrev"],
    y_test,
    "test",
)

all_preds_pd = pd.concat([train_preds_pd, test_preds_pd], ignore_index=True)
print(f"  Total prediction rows: {len(all_preds_pd):,}")

# COMMAND ----------

print("Writing workspace.tcga_gold.model_predictions (Delta, overwrite) ...")
all_preds_spark = spark.createDataFrame(all_preds_pd)
(
    all_preds_spark.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.tcga_gold.model_predictions")
)
print("  Done.")

# COMMAND ----------

# MAGIC %md
# MAGIC # ### Section 5 — Save Model Metrics

# COMMAND ----------

run_ts = datetime.utcnow().isoformat(timespec="seconds")

def metrics_to_rows(metrics_dict, model_name, split_label, run_timestamp):
    """Convert a scalar metrics dict into a list of row dicts."""
    return [
        {
            "model_name":    model_name,
            "metric_name":   metric_name,
            "metric_value":  float(metric_value),
            "split":         split_label,
            "run_timestamp": run_timestamp,
        }
        for metric_name, metric_value in metrics_dict.items()
    ]

all_metrics_rows = (
    metrics_to_rows(lr_train_metrics,   "LogisticRegression",     "train", run_ts) +
    metrics_to_rows(lr_test_metrics,    "LogisticRegression",     "test",  run_ts) +
    metrics_to_rows(rf_train_metrics,   "RandomForestClassifier", "train", run_ts) +
    metrics_to_rows(rf_test_metrics,    "RandomForestClassifier", "test",  run_ts) +
    metrics_to_rows(mlp_train_metrics,  "MLP_3Layer",             "train", run_ts) +
    metrics_to_rows(mlp_test_metrics,   "MLP_3Layer",             "test",  run_ts)
)

all_metrics_pd = pd.DataFrame(all_metrics_rows)
print(f"Metrics rows to write: {len(all_metrics_pd):,}")

# COMMAND ----------

print("Writing workspace.tcga_gold.model_metrics (Delta, overwrite) ...")
all_metrics_spark = spark.createDataFrame(all_metrics_pd)
(
    all_metrics_spark.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.tcga_gold.model_metrics")
)
print("  Done.")

# COMMAND ----------

# MAGIC %md
# MAGIC # ### Section 6 — Validation Summary

# COMMAND ----------

print("\n" + "="*60)
print("FINAL VALIDATION SUMMARY")
print("="*60)

print(f"\nWinning model : {best_model_name}")

for name, metrics in all_test_metrics.items():
    print(f"\n{name} — test set:")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")

print(f"\nConfusion Matrix for winning model ({best_model_name}) on test set:")
print(confusion_matrix(y_test, best_model.predict(X_test)))

print(f"\nrun_timestamp : {run_ts}")
print("\nGold tables written:")
print("  workspace.tcga_gold.model_predictions")
print("  workspace.tcga_gold.model_metrics")
print("\nNotebook 04 complete.")