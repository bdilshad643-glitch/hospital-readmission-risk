"""Download, clean and featurise the UCI "Diabetes 130-US Hospitals (1999-2008)" dataset.

Source: Strack et al., "Impact of HbA1c Measurement on Hospital Readmission Rates",
BioMed Research International, 2014. https://doi.org/10.24432/C5230J (CC BY 4.0)
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# discharge dispositions meaning the patient died or went to hospice: they cannot be readmitted
EXCLUDED_DISCHARGE = {11, 13, 14, 19, 20, 21}

MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
    "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]


def download(data_dir=DATA_DIR):
    """Fetch the CSV once and cache it under data/."""
    csv = data_dir / "diabetic_data.csv"
    if not csv.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL, timeout=120) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(data_dir)
    return csv


def icd9_group(code):
    """Map an ICD-9 diagnosis code to one of nine clinical groups (as in Strack et al.)."""
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith(("V", "E")):
        return "Other"
    value = float(code)
    if int(value) == 250:
        return "Diabetes"
    if 390 <= value <= 459 or int(value) == 785:
        return "Circulatory"
    if 460 <= value <= 519 or int(value) == 786:
        return "Respiratory"
    if 520 <= value <= 579 or int(value) == 787:
        return "Digestive"
    if 800 <= value <= 999:
        return "Injury"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 580 <= value <= 629 or int(value) == 788:
        return "Genitourinary"
    if 140 <= value <= 239:
        return "Neoplasms"
    return "Other"


def load(csv=None):
    """Return (features DataFrame, binary target Series, patient ids Series)."""
    df = pd.read_csv(csv or download(), na_values="?", low_memory=False)
    df = df[~df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE)]
    df = df[df["gender"] != "Unknown/Invalid"].reset_index(drop=True)

    y = (df["readmitted"] == "<30").astype(int).rename("readmitted_30d")
    groups = df["patient_nbr"]

    X = pd.DataFrame(index=df.index)
    # numeric
    X["age"] = df["age"].str.extract(r"\[(\d+)-").astype(float)[0] + 5  # bracket midpoint
    for c in ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
              "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]:
        X[c] = df[c]
    X["prior_visits_total"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    X["n_med_changes"] = sum((df[m].isin(["Up", "Down"])).astype(int) for m in MEDICATIONS)
    X["n_meds_active"] = sum((df[m] != "No").astype(int) for m in MEDICATIONS)

    # categorical
    X["race"] = df["race"].fillna("Missing")
    X["gender"] = df["gender"]
    X["admission_type"] = df["admission_type_id"].astype(str)
    X["discharge_disposition"] = df["discharge_disposition_id"].astype(str)
    X["admission_source"] = df["admission_source_id"].astype(str)
    top_specialties = df["medical_specialty"].value_counts().index[:10]
    X["medical_specialty"] = np.where(df["medical_specialty"].isna(), "Missing",
                                      np.where(df["medical_specialty"].isin(top_specialties),
                                               df["medical_specialty"], "Other"))
    for i in (1, 2, 3):
        X[f"diag_{i}_group"] = df[f"diag_{i}"].map(icd9_group)
    X["A1C_result"] = df["A1Cresult"].fillna("Not measured")
    X["max_glucose_serum"] = df["max_glu_serum"].fillna("Not measured")
    for m in ["metformin", "insulin", "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "glimepiride"]:
        X[f"med_{m}"] = df[m]
    X["med_change"] = df["change"]
    X["diabetes_med"] = df["diabetesMed"]
    return X, y, groups


def categorical_columns(X):
    return [c for c in X.columns if X[c].dtype == object]
