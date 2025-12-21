# AI-Integrated Intrusion Detection System 🚨🧠

A production-oriented, local-first intrusion detection prototype combining a supervised gradient-boosting classifier (CatBoost) with an unsupervised anomaly detector (Local Outlier Factor). The repository contains a Streamlit inference app, a full local training pipeline, and supporting artifacts for evaluation and explainability.

---

## Table of Contents
- 🚀 Project Overview
- 🧾 Recruiter-Friendly Project Summary (what you should know)
- ✨ Key Technical Features
- 🛠️ Tech Stack & Dependencies
- 📁 Project Structure (detailed)
- 🧩 Installation & Setup (train & run)
- ▶️ How to Train Locally (exact scripts & artifacts produced)
- ▶️ How to Run the Streamlit App (inference)
- 📦 Saved Artifacts & File Formats
- 🔍 Files used for Local Training — Thorough File-by-File Analysis
- ✅ Notes & Next Steps

---

## 🚀 Project Overview

This repository implements a hybrid intrusion detection workflow that combines:
- CatBoost (supervised binary classifier) to predict attacks vs. benign traffic, and
- Local Outlier Factor (LOF) as an anomaly detector trained only on benign traffic.

The hybrid voting strategy marks a sample as malicious if either detector raises a flag (OR logic), prioritizing detection (high recall) while preserving explainability through feature statistics and saved metrics.

---

## 🧾 Project Summary (what you should know)

This project demonstrates end-to-end competencies valuable for production ML roles:

- Practical model selection for imbalanced classification: replaces XGBoost + OCSVM with CatBoost + LOF for speed and better handling of categorical/imbalanced data.
- Data-leakage-aware design: strict temporal split (no shuffling), scaler fit on training only, and automated leakage warnings for suspicious metric values.
- Imbalance handling & resampling expertise: fallback simple resampling and an aggressive pipeline using imbalanced-learn's RandomUnderSampler, SMOTE, and ADASYN.
- Model engineering & deployment readiness: training script serializes models, scaler, feature names, and produces JSON artifacts for metrics and sample predictions — ready for Streamlit deployment.
- Performance & scaling considerations: sampling benign data for LOF to drastically reduce training time, multi-threaded CatBoost configuration, and measured training-time reporting.
- Reproducibility & artifacts: saved Pickle and JSON outputs for reproducible inference and dashboarding.

---

## ✨ Key Technical Features

- Hybrid ensemble: CatBoost classifier + LOF anomaly detector with OR voting to maximize recall.
- Leakage-aware pipeline: strict chronological train/test split and scaler fit on training only.
- Aggressive balancing strategy: undersampling + SMOTE + ADASYN (when imbalanced-learn available), with robust fallback to simple oversampling.
- LOF optimization: benign-only training with configurable sampling ratio to reduce LOF training time on large datasets.
- Comprehensive artifact generation: trained model pickle, metrics JSON, sample predictions JSON, and feature statistics JSON for Streamlit consumption.

---

## 🛠️ Tech Stack & Dependencies

Badges (primary):

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.1-orange)](https://streamlit.io/)  
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.2-red)](https://catboost.ai/)  
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-green)](https://scikit-learn.org/)  
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.11.0-lightgrey)](https://imbalanced-learn.org/)  

Primary packages (from repository files):
- streamlit
- numpy
- pandas
- scikit-learn
- catboost
- imbalanced-learn (optional; used in training)
- matplotlib, seaborn (visualization)
- gdown (suggests model download helper present in other flows)

Two sets of requirements are provided:
- requirements_catboost_lof_training.txt — for training (includes imbalanced-learn)
- requirements_catboost_lof_inference.txt — inference-only (no imbalanced-learn)
- requirements.txt — Streamlit inference general dependencies

---

## 📁 Project Structure (detailed)

Root (main files and purpose)

- .devcontainer/                 — optional devcontainer config (repro environment)
- .streamlit/                    — Streamlit config (UI settings)
- app.py                         — Main Streamlit inference application (UI + model loading)
- pipeline_catboost_lof.py       — Core training pipeline implementing CatBoost + LOF hybrid
- train_catboost_lof.py          — Convenience training script that runs pipeline and writes artifacts
- requirements.txt               — Inference requirements (streamlit + viz libs)
- requirements_catboost_lof_inference.txt — Inference-only reqs
- requirements_catboost_lof_training.txt  — Training-only reqs (includes imbalanced-learn)
- runtime.txt                    — runtime hint (platform)
- feature_stats.json             — example feature statistics for Streamlit dashboard
- sample_predictions.json        — example predictions for Streamlit demo
- metrics.json                   — example metrics for Streamlit demo
- sample_predictions_catboost_lof.json — (produced by training script) example predictions for CatBoost+LOF
- trained_models_catboost_lof.pkl        — (produced by training script) pickled models & metadata
- feature_stats_catboost_lof.json        — (produced by training script) training feature stats
- metrics_catboost_lof.json               — (produced by training script) metrics from training
- catboost_lof_results.csv                  — (produced by pipeline when run directly)

Note: Some of the files above (with _catboost_lof suffix) are created when you run the included local training script. If not present, run training to generate them.

---

## 🔍 Files used for Local Training — Thorough File-by-File Analysis

Below are the training-focused files you provided, with detailed technical notes (useful for interview talking points).

1) pipeline_catboost_lof.py
- Purpose: Encapsulates the complete CatBoost + LOF training, evaluation, and prediction pipeline.
- Key design decisions:
  - Temporal split: Implemented in temporal_split() — uses chronological split with no shuffling to avoid leakage across time.
  - Robust cleaning: Replaces +/-inf with NaN for "Flow Bytes/s" and "Flow Packets/s" if present, and converts 'Label' to binary (BENIGN -> 0, others -> 1).
  - Preprocessing: numeric-only features selected; missing values imputed using training medians (applied to test using training stats).
  - Scaling: StandardScaler fit on training only (scale_features).
  - Class imbalance handling:
    - When imbalanced-learn is available: aggressive_balance() runs a three-step balancing approach — optional light RandomUnderSampler, then SMOTE to 1:1, then ADASYN to refine. This is a strong pipeline for class imbalance.
    - If imbalanced-learn is not installed: _simple_balance() falls back to sklearn.utils.resample to oversample minority until parity.
  - CatBoost model:
    - CatBoostClassifier configured with iterations=500, depth=10, learning_rate=0.05, Logloss, eval_metric=AUC.
    - class_weights set to balance with ratio n_neg / n_pos.
    - thread_count=-1 to use all CPU cores; task_type='CPU' by default (GPU switchable).
  - LOF model:
    - LOF trained only on benign samples (novelty=True), contamination configurable, and uses sampling of benign set if dataset is large to reduce training time.
    - Predicts novelty at inference time and converts LOF outputs to a pseudo-probability via logistic transform on score_samples().
  - Voting strategy:
    - OR logic: sample classified as attack (1) if either CatBoost or LOF predicts attack — maximizes recall.
  - Evaluation:
    - evaluate_model() computes accuracy, precision, recall, f1, roc_auc (if proba available), confusion matrix, false positive rate, false negative rate, and prints human-friendly summary with leakage warnings when metrics are suspiciously high (recall & precision > 0.98).
- Runtime & performance:
  - Reports training time for CatBoost and LOF separately and aggregates them.
  - LOF sample ratio default 0.3 (30% of benign samples) to drastically reduce LOF training time for very large datasets.

2) train_catboost_lof.py
- Purpose: A convenience script that runs the pipeline and writes model and artifact files for use by Streamlit.
- Behavior:
  - Creates pipeline with default hyperparameters (threshold 0.15, LOF contamination 0.1, neighbors 20, LOF sample ratio 0.3).
  - Calls pipeline.run_pipeline(), then packages:
    - trained models + scaler + feature names into trained_models_catboost_lof.pkl using pickle.
    - metrics into metrics_catboost_lof.json.
    - sample predictions into sample_predictions_catboost_lof.json (samples up to 1000 records).
    - feature statistics (means/stds/mins/maxs) into feature_stats_catboost_lof.json.
- Output is intended for immediate consumption by the Streamlit app — minimal additional glue code necessary.
- Safety checks:
  - Exits with error if `merged_output.csv` not found.

3) requirements_catboost_lof_training.txt
- Purpose: Training-specific dependency list.
- Notable packages:
  - imbalanced-learn==0.11.0 — used heavily in aggressive balancing (RandomUnderSampler, SMOTE, ADASYN).

4) requirements_catboost_lof_inference.txt
- Purpose: Inference-only dependencies (Streamlit, CatBoost, scikit-learn, etc.) — smaller surface area than training.

Practical implications & interview notes:
- The repo demonstrates realistic operational concerns: class imbalance, temporal leakage, training-time constraints (sampling), multi-model hybridization, artifact serialization.
- The code shows familiarity with both supervised and unsupervised detection techniques and with standard ML libraries.

---
## 🧩 Installation & Setup

1. Clone repository
```bash
git clone https://github.com/aliuzair1/AI-Integrated-Intrusion-Detection-System.git
cd AI-Integrated-Intrusion-Detection-System
```

2. Create & activate a virtual environment
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

3. Install dependencies

- For local training (recommended):
```bash
pip install -r requirements_catboost_lof_training.txt
```

- For inference/Streamlit only:
```bash
pip install -r requirements_catboost_lof_inference.txt
# or use the general requirements.txt for the Streamlit app
pip install -r requirements.txt
```

4. Place training dataset
- The training scripts expect a CSV named `merged_output.csv` in the project root. If your filename differs, pass the path to the training script.

---

## ▶️ How to Train Locally (exact scripts & artifacts produced)

Train using the provided training script (this runs the entire pipeline and saves artifacts):

```bash
python train_catboost_lof.py
```

What train_catboost_lof.py does (summary):
- Instantiates CatBoostLOFPipeline from pipeline_catboost_lof.py with tuned defaults.
- Runs entire pipeline (load -> temporal split -> preprocessing -> balancing -> train CatBoost -> train LOF -> evaluate).
- Serializes the following files (saved to project root):
  - trained_models_catboost_lof.pkl (pickle containing CatBoost model, LOF model, scaler, feature names, thresholds, config)
  - metrics_catboost_lof.json (training metrics + metadata)
  - sample_predictions_catboost_lof.json (sample of predictions for dashboard)
  - feature_stats_catboost_lof.json (means, stds, mins, maxs for features)
- Estimated training time: ~4–6 minutes on a typical multi-core CPU (the pipeline reports timing).

Important: the script will error-out if `merged_output.csv` is not present — place your dataset in the root or update the path.

---

## ▶️ How to Run the Streamlit App (inference)

1. Ensure inference requirements installed:
```bash
pip install -r requirements_catboost_lof_inference.txt
```

2. Place model artifacts produced by training (or download them into repo root). Expected artifacts (names used by training script; app.py may look for slightly different names — update accordingly):
- trained_models_catboost_lof.pkl  (or trained models expected by app.py)
- sample_predictions_catboost_lof.json (for demo)
- feature_stats_catboost_lof.json
- metrics_catboost_lof.json

3. Run Streamlit
```bash
streamlit run app.py
```

Streamlit will open at http://localhost:8501 by default.

If the app expects a model at a different path or expects to download from Google Drive, check `app.py` and update model paths or set environment placeholders like [MODEL_DRIVE_ID].

---

## 📦 Saved Artifacts & File Formats (what to expect after training)

- trained_models_catboost_lof.pkl — Pickle file with keys:
  - 'catboost_model' : CatBoostClassifier object
  - 'lof_model' : LocalOutlierFactor object
  - 'scaler' : StandardScaler object
  - 'feature_names' : list of feature columns
  - 'catboost_threshold' : float
  - 'config' : metadata dict (sampling, neighbors, test_size, training_date)
- metrics_catboost_lof.json — JSON with evaluation metrics and metadata (accuracy, precision, recall, f1, roc_auc, confusion matrix counts, training times)
- sample_predictions_catboost_lof.json — JSON with arrays: y_true, y_pred_catboost, y_pred_lof, y_pred_voting, proba arrays
- feature_stats_catboost_lof.json — JSON with 'means', 'stds', 'mins', 'maxs' per feature (computed from scaled X_train)

These artifacts are designed to be consumed directly by the Streamlit app for visualization and demo.

---

## ✅ Notes

- The repository has the training and inference pieces required to reproduce results locally. To fully run training, place a CSV named `merged_output.csv` in the project root or change the path in `train_catboost_lof.py`.

---