# [cite_start]Predicting customer churn in SaaS platforms using machine learning techniques [cite: 1]

## Academic Context
* [cite_start]**University:** University of Wolverhampton [cite: 111]
* **Programme:** Master of Science (MSc)
* **Route:** Computer Science
* **Author:** IONUT BOSINCIANU
* **Year:** 2026

## Description
[cite_start]This repository contains a reproducible churn-prediction artefact developed for an MSc Dissertation[cite: 196]. The project implements an end-to-end machine learning pipeline specifically designed for SaaS-like tabular data.

### Key Features
* [cite_start]**Data Sources:** Utilizes the IBM Telco Customer Churn benchmark [cite: 203] [cite_start]and a custom synthetic SaaS-like dataset generator[cite: 210].
* [cite_start]**Leakage Control:** Implements strict, leakage-aware preprocessing via `sklearn` Pipelines and `ColumnTransformer` (median imputation, one-hot encoding, and scaling)[cite: 214, 219, 222].
* [cite_start]**Algorithms:** Evaluates regularised Logistic Regression [cite: 229][cite_start], Random Forest [cite: 236][cite_start], and XGBoost[cite: 242].
* [cite_start]**Evaluation:** Advanced metrics for imbalanced classes, including PR-AUC (with bootstrap CIs) [cite: 104, 264][cite_start], ROC-AUC [cite: 264][cite_start], Brier score calibration [cite: 265][cite_start], and F1-optimised threshold policies[cite: 260].
* [cite_start]**Interpretability:** Global (permutation importance) and local (SHAP beeswarm) attributions[cite: 267].

## Repository Structure
* [cite_start]`data/raw/Telco-Customer-Churn.csv`: Primary public dataset from IBM[cite: 203].
* [cite_start]`data/synthetic/saas_synth.csv`: Generated synthetic SaaS dataset[cite: 210].
* `src/churn_artefact/`: Core Python modules for the pipeline, calibration, and explainability.
* `run_experiments.py`: Main orchestration script for running experiments.
* `configs/default.json`: Global configuration for seeds, tuning iterations (40), and model settings.
* `outputs/`: Directory for saved metrics, calibration plots, and SHAP visualizations.

## Quickstart
```bash
# Setup virtual environment
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Install as editable package
pip install -e .

# Run experiments
python -m churn_artefact.run_experiments --config configs/default.json