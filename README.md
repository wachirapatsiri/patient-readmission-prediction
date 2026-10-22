# 🏥 Patient Readmission Prediction & Risk Assessment

## 📌 Project Overview
An end-to-end machine learning pipeline and web application designed to predict the likelihood of hospital readmission for diabetes patients within 30 days.

## 🎯 Business Problem
High hospital readmission rates strain healthcare resources and impact the quality of patient care. Identifying high-risk patients early allows healthcare providers to allocate resources effectively, optimize operational costs, and provide targeted follow-up care.

## 🛠️ Solution & Action (Tech Stack: Python, Scikit-Learn, SMOTE, Streamlit)
* **Feature Engineering:** Processed the complex Diabetes-130-Hospitals dataset by handling missing values, standardizing clinical features, and engineering new variables (Log transformations, Polynomial features).
* **Imbalanced Data Handling:** Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset, significantly improving the model's ability to detect high-risk patients.
* **Predictive Modeling:** Trained and evaluated a Random Forest Classifier, focusing on metrics like Recall and ROC-AUC that matter most in healthcare contexts.
* **Model Deployment:** Developed an interactive web interface using Streamlit, allowing medical staff to input clinical data and instantly receive a readmission risk assessment.

## 📈 Key Results
* Successfully identified key clinical factors driving readmissions, such as length of stay and the number of prescribed medications.
* Deployed a functional prototype that translates complex predictive algorithms into an actionable, user-friendly tool for healthcare professionals.
