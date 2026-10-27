"""Shared preprocessing for the patient-readmission project.

Used by BOTH the training notebook and the Streamlit app, so the two can never
drift apart. The notebook imports this module, and the saved joblib artifact
pickles `FeatureBuilder` by reference to it -> keep this file next to app.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Column definitions
# --------------------------------------------------------------------------- #
TARGET = "readmit_30_days"

# Alternative target encodings of the same outcome -> would leak the answer.
LEAKAGE_COLS = ["readmitted", "readmit_binary"]

# Missing values hidden behind placeholder strings in the OpenML version.
HIDDEN_MISSING = ["Missing", "?", "Unknown", "None", "Unknown/Invalid"]
MISSING_TO_UNKNOWN = ["race", "medical_specialty"]
MISSING_TO_NOT_TESTED = ["max_glu_serum", "A1Cresult"]  # not measured != unknown

BINARY_MAP = {"no": 0, "yes": 1, "ch": 1, "false": 0, "true": 1, "0": 0, "1": 1}
BINARY_COLS = [
    "change",
    "diabetesMed",
    "medicare",
    "medicaid",
    "had_emergency",
    "had_inpatient_days",
    "had_outpatient_days",
]

RAW_NUMERIC_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
]
CATEGORICAL_COLS = [
    "race",
    "gender",
    "age",
    "discharge_disposition_id",
    "admission_source_id",
    "medical_specialty",
    "primary_diagnosis",
    "max_glu_serum",
    "A1Cresult",
    "insulin",
]

# Numeric columns that exist AFTER FeatureBuilder (these get scaled).
ENGINEERED_NUMERIC_COLS = [
    "time_in_hospital",
    "number_diagnoses",
    "num_medications_log",
    "time_x_meds",
    "total_procedures",
    "care_intensity",
]


# --------------------------------------------------------------------------- #
# Deterministic cleaning (nothing here is learned from data -> safe before split)
# --------------------------------------------------------------------------- #
def _strip_quotes(value):
    """OpenML stores some labels as "'Over 60 years'" (with literal quotes)."""
    return value.strip("'") if isinstance(value, str) else value


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leakage columns, resolve hidden missing values, encode binary flags.

    Idempotent: calling it on already-cleaned data (e.g. one row built by the
    app, with 0/1 flags) returns the same values.
    """
    out = df.drop(columns=LEAKAGE_COLS, errors="ignore").copy()

    for col in out.select_dtypes(exclude=["number", "bool"]).columns:
        out[col] = out[col].astype("object").map(_strip_quotes)

    for col in MISSING_TO_UNKNOWN + MISSING_TO_NOT_TESTED + ["gender"]:
        if col in out.columns:
            out[col] = out[col].where(~out[col].isin(HIDDEN_MISSING), np.nan)

    if "gender" in out.columns:  # only 3 rows -> drop rather than impute
        out = out.dropna(subset=["gender"])

    for col in MISSING_TO_UNKNOWN:
        if col in out.columns:
            out[col] = out[col].fillna("Unknown")
    for col in MISSING_TO_NOT_TESTED:
        if col in out.columns:
            out[col] = out[col].fillna("Not Tested")

    for col in BINARY_COLS:
        if col in out.columns:
            mapped = out[col].astype(str).str.lower().str.strip().map(BINARY_MAP)
            if mapped.isna().any():
                bad = out.loc[mapped.isna(), col].unique()[:5]
                raise ValueError(f"Unmapped values in binary column '{col}': {bad}")
            out[col] = mapped.astype(int)

    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Learned feature engineering (fit on TRAIN only, via Pipeline)
# --------------------------------------------------------------------------- #
class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Clip length of stay, then build derived numeric features.

    The clipping cap is learned in `fit`, so it comes from the training data
    only (no test-set leakage).
    """

    def __init__(self, clip_quantile: float = 0.99):
        self.clip_quantile = clip_quantile

    def fit(self, X, y=None):
        self.time_cap_ = float(X["time_in_hospital"].quantile(self.clip_quantile))
        return self

    def transform(self, X):
        X = X.copy()
        X["time_in_hospital"] = X["time_in_hospital"].clip(upper=self.time_cap_)
        X["num_medications_log"] = np.log1p(X["num_medications"])
        X["time_x_meds"] = X["time_in_hospital"] * X["num_medications"]
        X["total_procedures"] = X["num_lab_procedures"] + X["num_procedures"]
        X["care_intensity"] = (
            X["total_procedures"] + X["num_medications"] + X["number_diagnoses"]
        )
        return X.drop(columns=["num_medications", "num_lab_procedures", "num_procedures"])


def build_preprocessor() -> ColumnTransformer:
    """Scale engineered numerics, one-hot encode categoricals, pass binary flags."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), ENGINEERED_NUMERIC_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
            ("bin", "passthrough", BINARY_COLS),
        ]
    )
