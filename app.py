import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. โหลดโมเดลและ Scaler (ใช้ @st.cache_resource เพื่อไม่ให้โหลดใหม่ทุกครั้งที่กดปุ่ม)
@st.cache_resource
def load_resources():
    model = joblib.load('diabetes_hgb_model.joblib')
    scaler = joblib.load('diabetes_scaler.joblib')
    return model, scaler

model, scaler = load_resources()

# 2. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Patient Risk Assessment", layout="centered")
st.title("🏥 Patient Readmission Risk Assessment")
st.markdown("เครื่องมือประเมินความเสี่ยงการกลับมาแอดมิทซ้ำของผู้ป่วยโรคเบาหวาน (ภายใน 30 วัน)")

# 3. สร้างฟอร์มรับข้อมูลจากผู้ใช้ (ดึงมาจาก Top Features ของโมเดลคุณ)
st.sidebar.header("📝 กรอกข้อมูลทางคลินิก")
time_in_hospital = st.sidebar.slider("ระยะเวลาที่อยู่ รพ. (วัน)", 1, 14, 3)
num_medications = st.sidebar.slider("จำนวนยาที่ได้รับ", 1, 80, 15)
number_diagnoses = st.sidebar.slider("จำนวนการวินิจฉัยโรค", 1, 16, 5)

st.write("### ข้อมูลผู้ป่วยเบื้องต้น")
st.write(f"- **ระยะเวลาที่พักรักษาตัว:** {time_in_hospital} วัน")
st.write(f"- **จำนวนยา:** {num_medications} ชนิด")
st.write(f"- **จำนวนการวินิจฉัย:** {number_diagnoses} รายการ")

# 4. ปุ่มกดทำนายผล
if st.button("🔍 ประเมินความเสี่ยง", type="primary"):
    # หมายเหตุ: ในการทำนายจริง โมเดลคุณต้องการข้อมูลครบทั้ง 42 คอลัมน์ตาม X_train
    # โค้ดส่วนนี้เป็นการจำลอง (Mock-up) เพื่อโชว์ UI ให้ HR เห็นภาพ
    
    # สมมติการคำนวณผลลัพธ์
    risk_score = np.random.uniform(0.3, 0.8) # สุ่มค่าความเสี่ยงจำลอง
    
    st.markdown("---")
    st.subheader("📊 ผลการประเมิน")
    
    if risk_score > 0.5:
        st.error(f"🚨 **ความเสี่ยงสูง (High Risk)**")
        st.write(f"ผู้ป่วยมีโอกาสกลับมาแอดมิทซ้ำ: **{risk_score*100:.1f}%**")
        st.write("👉 *คำแนะนำ: ควรจัดโปรแกรมติดตามอาการที่บ้านอย่างใกล้ชิด (Home Care Follow-up)*")
    else:
        st.success(f"✅ **ความเสี่ยงต่ำ (Low Risk)**")
        st.write(f"ผู้ป่วยมีโอกาสกลับมาแอดมิทซ้ำ: **{risk_score*100:.1f}%**")
        st.write("👉 *คำแนะนำ: ดำเนินการจำหน่ายผู้ป่วยตามมาตรฐานปกติ*")
