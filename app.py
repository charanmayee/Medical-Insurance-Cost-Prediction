import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Medical Insurance Cost Prediction", layout="centered")
st.title("Medical Insurance Cost Prediction")
st.write(
    "Predict insurance charges from age, BMI, children, smoker status, sex, and region."
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file `model.joblib` not found. "
            "Please run `python train_and_save.py` first to generate it."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


model = load_model()

# --- Inputs ---
# Default values match the "Example Prediction" in README.md (age 31, BMI 25.74).
age = st.number_input("Age", min_value=0, max_value=120, value=31)
bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=25.74, step=0.01)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
sex = st.selectbox("Sex", ["female", "male"])
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox(
    "Region", ["northeast", "northwest", "southeast", "southwest"]
)

# --- Prediction ---
if st.button("Predict Charges"):
    X = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region,
            }
        ]
    )
    pred = model.predict(X)[0]
    st.success(f"Estimated insurance charges: **${pred:,.2f}**")
