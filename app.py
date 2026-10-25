import streamlit as st
import pandas as pd
import joblib

# 1. โหลดโมเดล
@st.cache_resource
def load_resources():
    model = joblib.load('diabetes_hgb_model.joblib')
    return model

model = load_resources()

st.markdown("### 🩺 เครื่องมือประเมินความเสี่ยงการกลับมาแอดมิทซ้ำ (ภายใน 30 วัน)")

# 2. เอา Dictionary 42 คอลัมน์ที่รันจาก Colab มาวางตรงนี้ทั้งหมด!
# ⚠️ สำคัญมาก: ต้องเอาผลลัพธ์จาก Colab มาวางทับตรงนี้ให้ครบ 42 บรรทัด โมเดลถึงจะไม่ Error
default_patient = {
    "time_in_hospital": -0.13147904236707508,
    "num_medications": -0.12497919115819858,
    "number_diagnoses": 0.29782065870337604,
    "change": 0.0,
    "diabetesMed": 1.0,
    "medicare": 0.0,
    "medicaid": 0.0,
    "log_num_medications": 0.10333281913613272,
    "time_x_meds_poly": -0.34295721853973404,
    "avg_procedures_per_visit": 0.02959047130684174,
    "total_procedure": 0.02959047130684174,
    "care_intensity": 0.006168613861198635,
    "previous_visits": -0.7826468657452593,
    "race_Asian": 0.0,
    "race_Caucasian": 1.0,
    "race_Hispanic": 0.0,
    "race_Other": 0.0,
    "race_Unknown": 0.0,
    "gender_Male": 0.0,
    "age_'30-60 years'": 0.0,
    "age_'Over 60 years'": 1.0,
    "discharge_disposition_id_Other": 0.0,
    "admission_source_id_Other": 0.0,
    "admission_source_id_Referral": 0.0,
    "medical_specialty_Emergency/Trauma": 0.0,
    "medical_specialty_Family/GeneralPractice": 0.0,
    "medical_specialty_InternalMedicine": 0.0,
    "medical_specialty_Other": 0.0,
    "medical_specialty_Unknown": 0.0,
    "primary_diagnosis_'Musculoskeletal Issues'": 0.0,
    "primary_diagnosis_'Respiratory Issues'": 0.0,
    "primary_diagnosis_Diabetes": 0.0,
    "primary_diagnosis_Other": 1.0,
    "max_glu_serum_>300": 0.0,
    "max_glu_serum_Norm": 0.0,
    "max_glu_serum_Not Tested": 1.0,
    "A1Cresult_>8": 0.0,
    "A1Cresult_Norm": 0.0,
    "A1Cresult_Not Tested": 1.0,
    "insulin_No": 0.0,
    "insulin_Steady": 0.0,
    "insulin_Up": 0.0
}

# 3. สร้างช่องรับค่าจากผู้ใช้ (ปรับคำให้ชัดเจน/เป็นทางการขึ้น)
st.sidebar.header("📝 กรอกข้อมูลทางคลินิก")

user_prev_visits = st.sidebar.number_input("ประวัติการเข้ารับการรักษาก่อนหน้า (ครั้ง)", min_value=0, max_value=50, value=0)
user_time_hosp = st.sidebar.slider("ระยะเวลาที่พักรักษาตัวใน รพ. (วัน)", min_value=1, max_value=14, value=3)
user_diabetes_med = st.sidebar.selectbox("มีการสั่งจ่ายยารักษาเบาหวานหรือไม่?", ["Yes (มีการสั่งจ่าย)", "No (ไม่มี)"])
user_insulin = st.sidebar.selectbox("รูปแบบการให้ยาอินซูลิน (Insulin)", ["ไม่มีการใช้ (No)", "ใช้ปริมาณคงที่ (Steady)", "มีการปรับปริมาณ (Up/Down)"])

if st.button("🔍 ประเมินความเสี่ยง", type="primary"):
    
    # 4. นำค่าที่ผู้ใช้กรอก ไปเขียนทับค่า Default
    default_patient['previous_visits'] = user_prev_visits
    default_patient['time_in_hospital'] = user_time_hosp
    default_patient['diabetesMed'] = 1.0 if user_diabetes_med == "Yes (มีการสั่งจ่าย)" else 0.0
    
    # จัดการตัวแปร Dummy ของ Insulin
    if user_insulin == "ไม่มีการใช้ (No)":
        default_patient['insulin_No'] = 1.0
        default_patient['insulin_Steady'] = 0.0
    elif user_insulin == "ใช้ปริมาณคงที่ (Steady)":
        default_patient['insulin_No'] = 0.0
        default_patient['insulin_Steady'] = 1.0
    else:
        default_patient['insulin_No'] = 0.0
        default_patient['insulin_Steady'] = 0.0

    # 5. แปลงเป็น DataFrame 1 แถว เพื่อส่งให้โมเดลทำนาย
    input_df = pd.DataFrame([default_patient])
    
    # 6. ทำนายผลความน่าจะเป็น
    risk_score = model.predict_proba(input_df)[0, 1]
    
    # 7. แสดงผลลัพธ์
    st.subheader("📊 ผลการประเมิน")
    if risk_score > 0.5:
        st.error(f"🚨 **ความเสี่ยงสูง (High Risk)** - โอกาสแอดมิทซ้ำ: {risk_score*100:.1f}%")
        st.write("👉 คำแนะนำ: ควรจัดเตรียมทีมแพทย์เพื่อติดตามอาการใกล้ชิด (Home Care Follow-up)")
    else:
        st.success(f"✅ **ความเสี่ยงต่ำ (Low Risk)** - โอกาสแอดมิทซ้ำ: {risk_score*100:.1f}%")
        st.write("👉 คำแนะนำ: ดำเนินการจำหน่ายผู้ป่วยตามมาตรฐานปกติ")