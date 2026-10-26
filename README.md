# Patient Readmission Risk Prediction

Predicting 30-day hospital readmission for diabetic patients, with a calibrated risk score served through a Streamlit app.

**Live demo:** [patient-readmission-prediction-wachirapat.streamlit.app](https://patient-readmission-prediction-wachirapat.streamlit.app/)

## Problem
About 11% of diabetic patient encounters in this dataset are readmitted within 30 days. Flagging likely readmissions at discharge can help hospitals prioritize follow-up care.

## Data
- [Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), preprocessed version with 101,763 encounters
- Target: readmitted within 30 days (positive rate 11.2%)
- Stratified 80/20 train/test split

## Approach
1. Cleaned missing values, grouped rare categories, one-hot encoded categorical features, and scaled numeric features (fit on train only).
2. Compared ways to handle class imbalance. In this experiment, SMOTE lowered ROC-AUC (0.586) compared with class weighting (0.640).
3. Selected the model with 5-fold cross-validation on the training set only.
4. Calibrated probabilities (isotonic) and chose the decision threshold on a held-out validation set.
5. Evaluated once on the test set.

## Results

| Model (5-fold CV on train) | ROC-AUC |
|---|---|
| Logistic Regression (baseline) | 0.627 ± 0.003 |
| Random Forest | 0.629 ± 0.002 |
| HistGradientBoosting | 0.628 ± 0.004 |

All three models performed similarly. HistGradientBoosting was selected and calibrated.

**Final model on the test set (threshold 0.12, chosen on validation):**
- ROC-AUC 0.64, PR-AUC 0.177 (base rate 0.112)
- Recall 0.60, Precision 0.158 (about 1.4× the base rate)
- Mean predicted probability 0.112, matching the observed rate after calibration

**Top predictors (permutation importance, drop in ROC-AUC):** number of previous visits (0.074), discharge disposition other than home (0.026), and the interaction between length of stay and number of medications.

## Limitations
- Moderate discriminative power (ROC-AUC ~0.64), consistent with the difficulty of this problem.
- At this threshold about 42% of patients are flagged, so the model suits screening, not individual decisions.
- This dataset version has no patient ID, so the split could not be grouped by patient.
- Discharge disposition is pre-grouped into "Home" and "Other", so expired and hospice encounters (excluded in the original study) could not be removed.
- The demo app uses five inputs; all other features are set to median values.
- Educational project. Not intended for clinical use.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack
Python, pandas, scikit-learn, imbalanced-learn, Streamlit


