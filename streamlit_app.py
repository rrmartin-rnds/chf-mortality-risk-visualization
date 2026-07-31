import streamlit as st
import pandas as pd

st.set_page_config(page_title="CHF Mortality Risk Explorer")

st.title("CHF Mortality Risk Explorer")

st.write(
    """
    Explore mortality risk factors in congestive heart failure patients.
    Select variables below to filter the dataset.
    """
)

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

st.subheader("Dataset preview")

st.dataframe(df.head())

st.subheader("Risk factor filters")

show_age = st.checkbox("Age ≥ 70")
show_anaemia = st.checkbox("Anaemia")
show_diabetes = st.checkbox("Diabetes")
show_hbp = st.checkbox("High blood pressure")

filtered = df.copy()

if show_age:
    filtered = filtered[filtered["age"] >= 70]

if show_anaemia:
    filtered = filtered[filtered["anaemia"] == 1]

if show_diabetes:
    filtered = filtered[filtered["diabetes"] == 1]

if show_hbp:
    filtered = filtered[filtered["high_blood_pressure"] == 1]

mortality = filtered["DEATH_EVENT"].mean() * 100

st.metric("Mortality rate", f"{mortality:.1f}%")

st.write(f"Patients remaining: {len(filtered)}")
