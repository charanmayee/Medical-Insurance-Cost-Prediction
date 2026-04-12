import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("insurance.csv")

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

joblib.dump(pipe, "model.joblib")
print("Saved model.joblib")
