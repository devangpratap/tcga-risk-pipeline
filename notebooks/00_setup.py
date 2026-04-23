# Databricks notebook source
# MAGIC %md
# MAGIC # TCGA Pipeline — Schema Setup
# MAGIC
# MAGIC This notebook creates the **medallion architecture** schemas for the TCGA pipeline.
# MAGIC
# MAGIC | Schema | Purpose |
# MAGIC |---|---|
# MAGIC | `tcga_bronze` | **Raw data** — ingested as-is from GDC/TCGA sources (clinical TSVs, manifest files, metadata JSON) |
# MAGIC | `tcga_silver` | **Cleaned & joined** — deduplicated, typed, joined across clinical tables, null-handled |
# MAGIC | `tcga_gold` | **Business-ready / model-ready** — feature tables, aggregations, ML-ready datasets |

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.tcga_bronze;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.tcga_silver;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.tcga_gold;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW SCHEMAS IN workspace LIKE 'tcga_*';

# COMMAND ----------

print("✅ Medallion schemas created: tcga_bronze, tcga_silver, tcga_gold")