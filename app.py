import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Medical Insurance Cost Prediction", page_icon="🩺")
st.title("🩺 Medical Insurance Cost Prediction")

# Try common model file names
MODEL_CANDIDATES = ["model.pkl", "insurance_model.pkl", "best_model.pkl"]
model = None
for f in MODEL_CANDIDATES:
    if os.path.exists(f):
        model = joblib.load(f)
        break

if model is None:
    st.error("Model file not found. Add model.pkl (or insurance_model.pkl / best_model.pkl) to repo root.")
    st.stop()

st.subheader("Enter details")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["female", "male"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Basic encoding (adjust if your training pipeline used different encoding)
sex_val = 1 if sex == "male" else 0
smoker_val = 1 if smoker == "yes" else 0
region_map = {"northeast":0, "northwest":1, "southeast":2, "southwest":3}
region_val = region_map[region]

# Common feature order
X = pd.DataFrame([{
    "age": age,
    "sex": sex_val,
    "bmi": bmi,
    "children": children,
    "smoker": smoker_val,
    "region": region_val
}])

if st.button("Predict Insurance Cost"):
    try:
        pred = model.predict(X)[0]
        st.success(f"Estimated Insurance Cost: ${pred:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Your model may expect different feature columns/order. Update X accordingly.")
