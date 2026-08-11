# Homework 3: SVM Classification

Support Vector Machine classifiers applied to two tabular datasets, with a focus on preprocessing choices and the effect of feature scaling.

## What it covers

- **Diabetes diagnosis (SVM)**
  - Data preprocessing: median imputation for invalid zero-valued medical readings (robust to outliers and skewed distributions), and feature scaling
  - Linear SVM trained both without and with scaling, comparing accuracy and confusion matrices
  - Discussion of *why* SVMs are scale-sensitive (margins and dot products are distorted when feature magnitudes differ, e.g. glucose ~150 vs. BMI ~30)
- **Social network ad purchase prediction** — classification on the `Social_Network_Ads.csv` dataset
- Additional experiments on the `jeeves_tennis.csv` toy dataset

## Contents

- `HW3.ipynb` — main notebook with preprocessing, model training, evaluation, and written analysis
- `HW3-810801065-paria-khalili.pdf` — exported write-up of the solution
- `Machine Learning Basics-HW3 (1).pdf` — the assignment prompt
- `diabetes.csv`, `Social_Network_Ads.csv`, `jeeves_tennis.csv` — datasets used across the notebook
