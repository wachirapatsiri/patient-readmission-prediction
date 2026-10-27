"""Streamlit demo: 30-day readmission risk for diabetic patients.

Loads one artifact (models/readmission_model.joblib) that contains the full
pipeline, so the app applies exactly the same preprocessing as the notebook.
"""
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

import features  # noqa: F401  (the saved pipeline references features.FeatureBuilder)
from features import BINARY_COLS

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "models" / "readmission_model.joblib"
ASSETS_DIR = APP_DIR / "assets"

st.set_page_config(page_title="Readmission Risk Estimator", page_icon="🏥", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_bundle():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.relative_to(APP_DIR)}. Run the notebook to create it.")
    st.stop()

bundle = load_bundle()
model = bundle["model"]
threshold = bundle["threshold"]
base_rate = bundle["base_rate"]
defaults = bundle["defaults"]
choices = bundle["choices"]
ranges = bundle["ranges"]
feature_columns = bundle["feature_columns"]
metrics = bundle.get("metrics", {})

if bundle.get("sklearn_version") != sklearn.__version__:
    st.warning(
        f"Model was trained with scikit-learn {bundle.get('sklearn_version')} but this app runs "
        f"{sklearn.__version__}. Pin the training version in requirements.txt."
    )

# --------------------------------------------------------------------------- #
# Input widgets (sidebar). Widget keys equal column names so reset can set them.
# --------------------------------------------------------------------------- #
LABELS = {
    "age": "Age group",
    "gender": "Gender",
    "race": "Race",
    "admission_source_id": "Admission source",
    "discharge_disposition_id": "Discharged to",
    "time_in_hospital": "Days in hospital",
    "num_lab_procedures": "Lab procedures",
    "num_procedures": "Other procedures",
    "num_medications": "Medications given",
    "number_diagnoses": "Number of diagnoses",
    "primary_diagnosis": "Primary diagnosis group",
    "medical_specialty": "Admitting specialty",
    "max_glu_serum": "Max glucose serum test",
    "A1Cresult": "HbA1c test result",
    "insulin": "Insulin",
    "change": "Diabetes medication changed during stay",
    "diabetesMed": "Diabetes medication prescribed",
    "had_inpatient_days": "Prior inpatient stay",
    "had_emergency": "Prior emergency visit",
    "had_outpatient_days": "Prior outpatient visit",
    "medicare": "Medicare",
    "medicaid": "Medicaid",
}


def reset_inputs():
    for col, value in defaults.items():
        st.session_state[col] = bool(value) if col in BINARY_COLS else value


# Initial values live in session state (set once), so widgets take no default argument
# and "reset" can simply overwrite the same keys.
for _col, _value in defaults.items():
    st.session_state.setdefault(_col, bool(_value) if _col in BINARY_COLS else _value)

inputs = {}


def pick(container, col):
    inputs[col] = container.selectbox(LABELS[col], choices[col], key=col)


def number(container, col):
    low, high = ranges[col]
    inputs[col] = container.slider(LABELS[col], low, high, key=col)


def flag(container, col):
    inputs[col] = int(container.checkbox(LABELS[col], key=col))


st.sidebar.header("Patient inputs")
st.sidebar.button("Reset to typical patient", on_click=reset_inputs, use_container_width=True)

with st.sidebar.expander("Patient", expanded=True):
    for col in ["age", "gender", "race"]:
        pick(st, col)

with st.sidebar.expander("Admission and stay", expanded=True):
    for col in ["admission_source_id", "discharge_disposition_id"]:
        pick(st, col)
    for col in ["time_in_hospital", "number_diagnoses", "num_medications",
                "num_lab_procedures", "num_procedures"]:
        number(st, col)
    for col in ["primary_diagnosis", "medical_specialty"]:
        pick(st, col)

with st.sidebar.expander("Labs and diabetes treatment"):
    for col in ["max_glu_serum", "A1Cresult", "insulin"]:
        pick(st, col)
    for col in ["change", "diabetesMed"]:
        flag(st, col)

with st.sidebar.expander("Prior visits and insurance"):
    for col in ["had_inpatient_days", "had_emergency", "had_outpatient_days", "medicare", "medicaid"]:
        flag(st, col)

# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
row = pd.DataFrame([inputs])[feature_columns]
risk = float(model.predict_proba(row)[0, 1])
flagged = risk >= threshold

# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.title("🏥 30-Day Readmission Risk Estimator")
st.caption("Diabetic inpatient encounters · calibrated gradient-boosting model · Diabetes 130-US Hospitals data")
st.warning("Educational project. Not a medical device and not intended for clinical decisions.", icon="⚠️")

tab_estimate, tab_model = st.tabs(["Risk estimate", "About the model"])

with tab_estimate:
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated 30-day readmission risk", f"{risk:.1%}")
    c2.metric(
        "Compared with the average patient",
        f"{risk / base_rate:.2f}×",
        delta=f"{(risk - base_rate) * 100:+.1f} percentage points",
        delta_color="off",
    )
    c3.metric("Screening flag", "Flagged" if flagged else "Not flagged")

    if flagged:
        st.warning(
            f"Risk is at or above the screening threshold ({threshold:.0%}). "
            "In a screening workflow, this patient would be prioritised for follow-up."
        )
    else:
        st.info(
            f"Risk is below the screening threshold ({threshold:.0%}). "
            "In a screening workflow, this patient would not be prioritised."
        )

    with st.expander("How to read this"):
        st.markdown(
            f"""
- **Risk** is a calibrated probability: among patients with similar inputs, roughly
  {risk:.0%} were readmitted within 30 days. The average patient in the data is at
  **{base_rate:.1%}**.
- The **screening threshold ({threshold:.0%})** was chosen on a validation set to catch about
  {metrics.get("target_recall", 0.6):.0%} of future readmissions. A lower threshold flags more patients
  and catches more readmissions but with more false alarms.
- The model separates higher- and lower-risk groups only **moderately** (see the *About the model* tab),
  so use it to rank groups of patients, not to judge an individual.
"""
        )

with tab_model:
    st.subheader("Performance on a held-out test set")
    if metrics:
        roc_ci = metrics.get("roc_auc_ci95", [None, None])
        pr_ci = metrics.get("pr_auc_ci95", [None, None])
        rows = [
            ("ROC-AUC", f"{metrics['roc_auc']:.3f} (95% CI {roc_ci[0]:.3f}-{roc_ci[1]:.3f})"),
            (
                "PR-AUC",
                f"{metrics['pr_auc']:.3f} (95% CI {pr_ci[0]:.3f}-{pr_ci[1]:.3f}); "
                f"random baseline {metrics['test_positive_rate']:.3f}",
            ),
            (f"Recall at threshold {threshold:.2f}", f"{metrics['recall']:.1%}"),
            (
                "Precision at threshold",
                f"{metrics['precision']:.1%} ({metrics['lift_vs_base_rate']:.2f}× the base rate)",
            ),
            ("Share of patients flagged", f"{metrics['flagged_share']:.1%}"),
            (
                "Mean predicted vs observed rate",
                f"{metrics['mean_predicted_probability']:.3f} vs {metrics['test_positive_rate']:.3f}",
            ),
        ]
        st.table(pd.DataFrame(rows, columns=["Metric", "Test set"]).set_index("Metric"))
    else:
        st.info("No metrics stored in the model file.")

    left, right = st.columns(2)
    calibration_png = ASSETS_DIR / "calibration.png"
    importance_png = ASSETS_DIR / "feature_importance.png"
    if calibration_png.exists():
        left.image(str(calibration_png), caption="Calibration: predicted vs observed readmission rate")
    if importance_png.exists():
        right.image(str(importance_png), caption="Which inputs the model relies on most")
    elif bundle.get("top_features"):
        right.bar_chart(pd.DataFrame(bundle["top_features"]).set_index("feature"))

    st.subheader("Limitations")
    st.markdown(
        """
- Moderate discriminative power; suitable for screening and prioritisation, not individual decisions.
- Roughly four in ten patients are flagged at the chosen threshold, and most flags are false alarms.
- The dataset has no patient ID, so splits could not be grouped by patient.
- US hospitals, 1999-2008; practice patterns have changed.
"""
    )
