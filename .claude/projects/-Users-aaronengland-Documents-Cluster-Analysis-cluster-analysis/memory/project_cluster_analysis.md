---
name: Cluster Analysis Project
description: Cluster analysis project for portfolio site - S3 bucket cluster-analysis-demo, parquet files, SageMaker notebooks
type: project
---

Building a cluster analysis for Aaron's portfolio website.

- S3 bucket: `cluster-analysis-demo`
- Data format: parquet (not CSV)
- Data must never be saved locally — only read/write from S3
- Small tables, models, and graphs are stored locally in `./output/` per notebook
- Notebooks run on AWS SageMaker
- Follow the same numbered folder structure as the credit risk model repo at /Users/aaronengland/Documents/Credit_Risk_Model_Braviant/credit_risk_model

**Why:** Portfolio showcase — code quality and structure matter.
**How to apply:** Mirror the credit risk model patterns (numbered folders, output/ subdirs, preprocessing classes, joblib serialization, S3 data flow, comprehensive visualizations).
