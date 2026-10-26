import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Readmission Risk", page_icon="🏥", layout="wide")


# ---------- โหลดโมเดล ----------
@st.cache_resource
def load_bundle():
    return joblib.load("readmission_model.joblib")


bundle = load_bundle()
model = bundle["model"]
threshold = bundle["threshold"]
base_rate = bundle["base_rate"]
features = bundle["feature_names"]
medians = bundle["medians"].reindex(features).astype(float)
scaler = bundle["scaler"]
num_cols = bundle["num_cols"]

# ค่ามัธยฐานแบบหน่วยจริง (ก่อน scale) ใช้เป็นค่าเริ่มต้นของฟอร์ม
raw_median = pd.Series(
    scaler.inverse_transform(pd.DataFrame([medians[num_cols].values], columns=num_cols))[0],
    index=num_cols,
)


def default_int(col, fallback, low, high):
    value = int(round(raw_median[col])) if col in raw_median.index else fallback
    return max(low, min(high, value))


INSULIN_OPTIONS = {
    "ไม่มีการใช้ (No)": "No",
    "ใช้ในขนาดเท่าเดิม (Steady)": "Steady",
    "เพิ่มขนาดยา (Up)": "Up",
    "ลดขนาดยา (Down)": "Down",
}


# ---------- สร้างข้อมูล 1 แถวสำหรับโมเดล ----------
def build_input(previous_visits, time_in_hospital, number_diagnoses, diabetes_med, insulin):
    row = medians.copy()

    # 1) ตัวแปรตัวเลข: แก้ในหน่วยจริง แล้ว scale ด้วย scaler ตัวเดิม
    raw = raw_median.copy()
    for col, val in {
        "previous_visits": previous_visits,
        "time_in_hospital": time_in_hospital,
        "number_diagnoses": number_diagnoses,
    }.items():
        if col in raw.index:
            raw[col] = val
    row[num_cols] = scaler.transform(pd.DataFrame([raw.values], columns=num_cols))[0]

    # 2) ยารักษาเบาหวาน (0/1)
    if "diabetesMed" in row.index and "diabetesMed" not in num_cols:
        row["diabetesMed"] = 1.0 if diabetes_med else 0.0

    # 3) อินซูลิน (one-hot)
    insulin_cols = [c for c in features if c.startswith("insulin_")]
    row[insulin_cols] = 0.0
    if f"insulin_{insulin}" in insulin_cols:
        row[f"insulin_{insulin}"] = 1.0

    return pd.DataFrame([row.values], columns=features)


# ---------- Sidebar: ฟอร์ม ----------
with st.sidebar:
    st.header("📝 กรอกข้อมูลทางคลินิก")
    previous_visits = st.number_input(
        "ประวัติการเข้ารับการรักษาก่อนหน้า (ครั้ง)",
        min_value=0, max_value=80, value=default_int("previous_visits", 0, 0, 80), step=1,
    )
    time_in_hospital = st.slider(
        "ระยะเวลาที่พักรักษาตัวใน รพ. (วัน)",
        min_value=1, max_value=14, value=default_int("time_in_hospital", 3, 1, 14),
    )
    number_diagnoses = st.slider(
        "จำนวนการวินิจฉัยโรค",
        min_value=1, max_value=16, value=default_int("number_diagnoses", 8, 1, 16),
    )
    diabetes_med = st.selectbox("มีการสั่งจ่ายยารักษาเบาหวานหรือไม่?", ["Yes (มีการสั่งจ่าย)", "No (ไม่มี)"]) \
        .startswith("Yes")
    insulin_label = st.selectbox("รูปแบบการให้อินซูลิน (Insulin)", list(INSULIN_OPTIONS.keys()))
    st.caption("ข้อมูลอื่นที่ไม่ได้กรอกจะใช้ค่ามัธยฐานของผู้ป่วยในชุดข้อมูล")


# ---------- หน้าหลัก ----------
st.title("🏥 เครื่องมือประเมินความเสี่ยงการกลับมาแอดมิทซ้ำ (ภายใน 30 วัน)")
st.write("ประเมินโอกาสที่ผู้ป่วยเบาหวานจะกลับมานอนโรงพยาบาลซ้ำภายใน 30 วันหลังออกจากโรงพยาบาล")

if st.button("🔍 ประเมินความเสี่ยง", type="primary"):
    input_df = build_input(previous_visits, time_in_hospital, number_diagnoses,
                           diabetes_med, INSULIN_OPTIONS[insulin_label])
    prob = float(model.predict_proba(input_df)[:, 1][0])

    st.subheader("📊 ผลการประเมิน")
    c1, c2 = st.columns(2)
    c1.metric(
        "โอกาสกลับมาแอดมิทใน 30 วัน (โดยประมาณ)",
        f"{prob:.1%}",
        delta=f"{prob - base_rate:+.1%} เทียบค่าเฉลี่ย ({base_rate:.1%})",
        delta_color="inverse",
    )
    c2.metric("เทียบกับผู้ป่วยโดยเฉลี่ย", f"{prob / base_rate:.1f} เท่า")
    st.progress(min(prob, 1.0))

    if prob >= threshold:
        st.warning(f"⚠️ กลุ่มควรติดตามอาการ (สูงกว่าเกณฑ์ {threshold:.0%})\n\n"
                   "คำแนะนำ: พิจารณานัดติดตามอาการหรือโทรสอบถามหลังจำหน่าย")
    else:
        st.success(f"✅ ความเสี่ยงต่ำกว่าเกณฑ์ ({threshold:.0%})\n\n"
                   "คำแนะนำ: ดำเนินการจำหน่ายผู้ป่วยตามมาตรฐานปกติ")
else:
    st.info("👈 กรอกข้อมูลทางซ้าย แล้วกดปุ่ม **ประเมินความเสี่ยง**")

with st.expander("ℹ️ เกี่ยวกับโมเดล"):
    st.markdown(
        f"""
- **ข้อมูล:** Diabetes 130-US Hospitals (1999–2008) ประมาณ 101K encounters
- **โมเดล:** HistGradientBoosting + isotonic calibration
- **ผลบน test set:** ROC-AUC 0.64, PR-AUC 0.177, Recall 0.60, Precision 0.158
- **เกณฑ์ติดตามอาการ:** {threshold:.0%} (เลือกจาก validation set)
- **อัตราการกลับมาแอดมิทเฉลี่ย:** {base_rate:.1%}
"""
    )

st.caption("⚠️ โปรเจกต์เพื่อการศึกษาเท่านั้น ไม่ได้ออกแบบมาเพื่อใช้ตัดสินใจทางการแพทย์จริง")