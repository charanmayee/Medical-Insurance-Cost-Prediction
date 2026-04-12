import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Medical Insurance Cost Prediction", layout="centered")
st.title("Medical Insurance Cost Prediction")
st.write("Predict insurance charges from age, BMI, children, smoker, sex, and region.")

MODEL_PATH = Path("model.joblib")
DATA_PATH = Path("insurance.csv")

@st.cache_resource
def load_or_train_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    if not DATA_PATH.exists():
        st.error("Dataset not found: insurance.csv")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
    y = df["charges"]

    cat = ["sex", "smoker", "region"]
    num = ["age", "bmi", "children"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", "passthrough", num),
        ]
    )

    model = RandomForestRegressor(random_state=42)
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X, y)

    joblib.dump(pipe, MODEL_PATH)
    return pipe

model = load_or_train_model()

age = st.number_input("Age", min_value=0, max_value=120, value=31)
bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=25.74, step=0.01)
children = st.number_input("Children", min_value=0, max_value=10, value=0, step=1)
sex = st.selectbox("Sex", ["female", "male"])
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"]) 

if st.button("Predict charges"):
    X = pd.DataFrame([
        {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
        }
    ])
    pred = model.predict(X)[0]
    st.success(f"Predicted charges: ${pred:,.2f}")
