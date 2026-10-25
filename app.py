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

# 2. วาง Dictionary ค่าเฉลี่ย 42 คอลัมน์ (เอาผลลัพธ์ที่ก๊อปปี้จากสเต็ป 1 มาวางตรงนี้ทั้งหมด)
default_patient = {
    "time_in_hospital": 4.0,
    "num_medications": 15.0,
    "previous_visits": 0.0,
    "diabetesMed": 1.0,
    "insulin_No": 1.0,
    "insulin_Steady": 0.0
    # ... (ต้องใส่ให้ครบ 42 บรรทัดตาม X_train.columns เป๊ะๆ) ...
}

# 3. สร้างช่องรับค่าจากผู้ใช้ (เฉพาะตัวที่สำคัญ)
st.sidebar.header("📝 กรอกข้อมูลทางคลินิก")

user_prev_visits = st.sidebar.number_input("จำนวนครั้งที่เคยมา รพ. (previous_visits)", min_value=0, max_value=50, value=0)
user_time_hosp = st.sidebar.slider("ระยะเวลาที่พักรักษาตัว (time_in_hospital)", min_value=1, max_value=14, value=3)
user_diabetes_med = st.sidebar.selectbox("ได้รับยารักษาเบาหวานหรือไม่? (diabetesMed)", ["Yes", "No"])
user_insulin = st.sidebar.selectbox("การใช้ Insulin", ["ไม่ใช้ (No)", "ใช้ปริมาณคงที่ (Steady)", "ปรับปริมาณขึ้น/ลง (Up/Down)"])

if st.button("🔍 ประเมินความเสี่ยง", type="primary"):
    
    # 4. นำค่าที่ผู้ใช้กรอก ไปเขียนทับค่า Default
    default_patient['previous_visits'] = user_prev_visits
    default_patient['time_in_hospital'] = user_time_hosp
    default_patient['diabetesMed'] = 1.0 if user_diabetes_med == "Yes" else 0.0
    
    # จัดการตัวแปร Dummy ของ Insulin
    if user_insulin == "ไม่ใช้ (No)":
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
    
    # 6. ทำนายผลความน่าจะเป็น (Probability)
    risk_score = model.predict_proba(input_df)[0, 1]
    
    # 7. แสดงผลลัพธ์จริง
    st.subheader("📊 ผลการประเมิน")
    if risk_score > 0.5:
        st.error(f"🚨 **ความเสี่ยงสูง (High Risk)** - โอกาสแอดมิทซ้ำ: {risk_score*100:.1f}%")
        st.write("👉 คำแนะนำ: ควรจัดเตรียมทีมแพทย์เพื่อติดตามอาการใกล้ชิด (Home Care Follow-up)")
    else:
        st.success(f"✅ **ความเสี่ยงต่ำ (Low Risk)** - โอกาสแอดมิทซ้ำ: {risk_score*100:.1f}%")
        st.write("👉 คำแนะนำ: ดำเนินการจำหน่ายผู้ป่วยตามมาตรฐานปกติ")
