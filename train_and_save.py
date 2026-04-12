"""
train_and_save.py
=================
Train a RandomForestRegressor pipeline on the insurance dataset and save it
to `model.joblib` in the repository root.

Dataset
-------
Place `insurance.csv` in the same directory as this script (the repo root).
The CSV must contain the following columns:
    age, sex, bmi, children, smoker, region, charges

You can download a public version of the dataset from Kaggle:
    https://www.kaggle.com/datasets/mirichoi0218/insurance

Usage
-----
    python train_and_save.py
"""

import sys
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Paths (all relative to this script so it works from any working directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "insurance.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.joblib")

REQUIRED_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
NUMERIC_FEATURES = ["age", "bmi", "children"]

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
if not os.path.exists(DATASET_PATH):
    print(
        f"ERROR: Dataset file not found at '{DATASET_PATH}'.\n"
        "Please place `insurance.csv` in the repository root before running this script.\n"
        "You can download it from: https://www.kaggle.com/datasets/mirichoi0218/insurance"
    )
    sys.exit(1)

df = pd.read_csv(DATASET_PATH)

missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
if missing:
    print(
        f"ERROR: The dataset is missing the following required columns: {missing}\n"
        f"Found columns: {list(df.columns)}"
    )
    sys.exit(1)

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df["charges"]

# ---------------------------------------------------------------------------
# Build preprocessing + model pipeline
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERIC_FEATURES),
    ]
)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
)

# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training RandomForestRegressor …")
pipeline.fit(X_train, y_train)

r2 = pipeline.score(X_test, y_test)
print(f"Test R² score: {r2:.4f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
joblib.dump(pipeline, MODEL_PATH)
print(f"Model saved to '{MODEL_PATH}'")
