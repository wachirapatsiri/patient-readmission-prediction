# 🏥 Patient Readmission Prediction & Risk Assessment

## 📌 Project Overview
An end-to-end machine learning pipeline and web application designed to predict the likelihood of hospital readmission for diabetes patients within 30 days.

## 🎯 Business Problem
High hospital readmission rates strain healthcare resources and impact the quality of patient care. Identifying high-risk patients early allows healthcare providers to allocate resources effectively, optimize operational costs, and provide targeted follow-up care to those who need it most.

## 🛠️ Solution & Action (Tech Stack: Python, Scikit-Learn, SMOTE, Streamlit)
* **Data Engineering & Imbalanced Data:** Processed the complex Diabetes-130-Hospitals dataset. Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance **81K** patient records, ensuring the model can effectively detect minority high-risk cases.
* **Feature Engineering:** Extracted clinical insights by applying advanced techniques like Log transformations and Polynomial features on medical variables (e.g., number of medications, length of stay).
* **Predictive Modeling:** Trained a Random Forest Classifier, shifting the evaluation focus away from standard accuracy to prioritize Recall—a critical metric in healthcare to minimize missed high-risk patients.
* **Deployment:** Developed an interactive web interface using Streamlit, allowing users to input clinical data and instantly receive a readmission risk assessment.

## 📈 Key Results
* Achieved a **Recall of 0.87** and an **ROC-AUC of 0.556**, successfully aligning machine learning metrics with real-world healthcare constraints (minimizing False Negatives).
* Deployed a functional, real-time prototype that translates complex predictive algorithms into an actionable, user-friendly tool for healthcare professionals.

## Web : https://patient-readmission-prediction-wachirapat.streamlit.app/

## 💻 Live Demo & UI
