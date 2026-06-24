# Internship Submission — NLP Customer Support Project

**Project:** NLP-Based Customer Support Ticket Classification and Auto Reply System  
**Domain:** Machine Learning — Internship Project 3  
**Submitted by:** Sukhxdx  
**Date:** June 2025

---

## Quick Links

| Resource | Link |
|----------|------|
| **Live Streamlit App** | https://sukhxdx-customer-support-nlp-app-fvlufn.streamlit.app |
| **GitHub Repository** | https://github.com/Sukhxdx/customer-support-nlp |
| **Project Report** | `reports/project_report.md` |
| **Submission ZIP** | `customer-support-nlp-submission.zip` (upload this to Google Drive) |

### Upload to Google Drive (for internship submission)

1. Go to [drive.google.com](https://drive.google.com) and sign in.
2. Click **New** → **File upload**.
3. Select `customer-support-nlp-submission.zip` from this project folder.
4. After upload, right-click the file → **Share** → **General access** → **Anyone with the link** → **Viewer**.
5. Click **Copy link** — that is your Google Drive submission link.

---

## What Is Included

### 1. Project Report
- Full internship report: `reports/project_report.md`
- Covers abstract, problem statement, methodology, results, and conclusion

### 2. Source Code
- `src/` — Modular Python package (preprocessing, training, evaluation, prediction)
- `app.py` — Streamlit web application
- `train.py` — Model training script
- `requirements.txt` — All dependencies

### 3. Dataset
- `data/customer_support_tickets.csv` — 500 balanced support tickets (5 categories)

### 4. Jupyter Notebook
- `notebooks/analysis_and_model_training.ipynb` — Exploratory analysis and training

### 5. Model Artifacts
- `models/support_ticket_classifier.pkl` — Trained best model pipeline
- `models/category_labels.pkl` — Category labels
- `models/training_metrics.json` — Accuracy, precision, recall, F1 scores

### 6. Evaluation & Screenshots
- `reports/figures/` — Confusion matrix, model comparison, top features charts
- `screenshots/` — App UI and training output screenshots

### 7. README
- `README.md` — Installation, usage, deployment, and screenshots

---

## Key Results

| Metric | Value |
|--------|-------|
| Best Model | Logistic Regression |
| Test Accuracy | 100% |
| Target (≥ 85%) | Met |
| Categories | 5 |
| Dataset Size | 500 tickets |

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or train manually:

```bash
python train.py
```

The Streamlit app auto-trains the model on first run if the `.pkl` file is missing.

---

## Submission Checklist

- [x] Problem statement and objectives
- [x] Dataset description and sample data
- [x] NLP preprocessing and TF-IDF feature extraction
- [x] Multi-model comparison (Logistic Regression, Naive Bayes, Linear SVM)
- [x] Model evaluation with confusion matrix and metrics
- [x] ≥ 85% classification accuracy achieved
- [x] Auto-reply generation per category
- [x] Streamlit deployment (live demo link above)
- [x] GitHub repository with full source code
- [x] Project report and screenshots

---

*Machine Learning Internship — Project 3: NLP for Customer Support*
