---
name: Credit Risk Model Reference Repo
description: Reference repo at /Users/aaronengland/Documents/Credit_Risk_Model_Braviant/credit_risk_model for coding style and structure patterns
type: reference
---

The credit risk model repo serves as the template for all portfolio projects. Located at `/Users/aaronengland/Documents/Credit_Risk_Model_Braviant/credit_risk_model`.

Key patterns to replicate:
- Numbered folders (01_eda/, 02_split_data/, etc.) each with notebook.ipynb + output/
- Constants defined at top of notebook (bucket, task name, output dir)
- Custom preprocessing classes with .fit()/.transform() pattern
- Data read from S3, artifacts saved locally to ./output/
- Joblib for model serialization
- Optuna for hyperparameter tuning
- Comprehensive visualizations (matplotlib/seaborn)
- requirements.txt at root
