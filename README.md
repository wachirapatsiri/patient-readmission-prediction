# 🏥 Patient Readmission Prediction & Risk Assessment

## 📌 Project Overview
An end-to-end machine learning pipeline and web application designed to predict the likelihood of hospital readmission for diabetes patients within 30 days.

## 🎯 Business Problem
High hospital readmission rates strain healthcare resources and impact the quality of patient care. Identifying high-risk patients early allows healthcare providers to allocate resources effectively, optimize operational costs, and provide targeted follow-up care to those who need it most.

## 🛠️ Solution & Action (Tech Stack: Python, Scikit-Learn, HistGradientBoosting, Streamlit)
* **Data Engineering:** Processed the complex Diabetes-130-Hospitals dataset (**81K** records) by handling missing values and engineering new variables (Log transformations, Polynomial features).
* **Rigorous Model Selection:** Instead of blindly relying on a single algorithm, I established a baseline using Logistic Regression and compared it against Random Forest (with and without SMOTE) and HistGradientBoosting.
* **Tackling Class Imbalance:** My experiments revealed that synthetic oversampling (SMOTE) actually degraded performance on this highly categorical dataset due to overlapping noise. I successfully addressed the imbalance by using native `class_weight="balanced"` and tuning tree complexity.
* **Deployment:** Selected the best-performing model (HistGradientBoosting) and deployed it as an interactive Streamlit web application for real-time clinical risk assessment.

## 📈 Key Results
* **HistGradientBoosting** emerged as the top performer, achieving an **ROC-AUC of 0.641** and a **PR-AUC of 0.177**, significantly outperforming the initial SMOTE approach (ROC-AUC 0.586).
* Gained a crucial technical insight: Advanced gradient boosting combined with strategic hyperparameter tuning is often more robust than data-level resampling (SMOTE) for complex healthcare datasets.
* Deployed a functional, real-time prototype that translates complex predictive algorithms into an actionable, user-friendly tool for healthcare professionals.

## Web : https://patient-readmission-prediction-wachirapat.streamlit.app/

## 💻 Live Demo & UI
<img width="1443" height="767" alt="image" src="https://github.com/user-attachments/assets/28fb1bcb-68db-4630-9811-6034ee106a3b" />


