# Medical Insurance Cost Prediction

Predict medical insurance charges using machine learning based on customer attributes such as age, BMI, number of children, smoking status, sex, and region.

## Project Overview
Medical insurance cost (charges) can vary significantly depending on lifestyle and demographic factors. This project builds a regression model to estimate an individual's insurance charges from structured input features.

**Problem type:** Supervised Learning (Regression)

## Dataset
Typical columns used in this project:
- `age`: Age of primary beneficiary
- `sex`: Insurance contractor gender (`male`, `female`)
- `bmi`: Body mass index
- `children`: Number of dependents covered
- `smoker`: Smoking status (`yes`, `no`)
- `region`: Residential area (`northeast`, `northwest`, `southeast`, `southwest`)
- `charges`: Medical insurance cost (target variable)

> If your dataset source is public (e.g., Kaggle), add the link here.


## Approach
1. Load and explore the dataset (EDA)
2. Handle missing values (if any)
3. Encode categorical variables (One-Hot Encoding / Label Encoding)
4. Split into train/test sets
5. Train regression models (examples below)
6. Evaluate model performance
7. Save the best model for future predictions (optional)

## Models (Examples)
You can experiment with:
- Linear Regression
- Ridge / Lasso Regression
- Random Forest Regressor
- Gradient Boosting / XGBoost (optional)

## Evaluation Metrics
Common regression metrics used:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error) / RMSE
- R² Score

## Example Prediction
- Age: 31 
- BMI: 25.74 
- Children: 0  
- Smoker: No  
- Sex: Female  
- Region: southeast  

Output:
- Predicted charges: $3760.0805765

## Results
Add your final model and scores here, e.g.:
- Best model: RandomForestRegressor
- Test RMSE: 4,200
- Test R²: 0.7447273869684076

## Future Improvements
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Feature engineering (e.g., BMI categories, interaction terms)
- Model explainability (SHAP / permutation importance)

---

## Web App (Streamlit)

The repository includes a Streamlit app (`app.py`) that lets you enter patient
details and get an instant insurance-cost prediction.

### Prerequisites

The trained model file `model.joblib` must exist before running the app.
Generate it by running the training script once (requires `insurance.csv` in
the repo root):

```bash
python train_and_save.py
```

> **Dataset**: Place `insurance.csv` in the repo root before training.  
> You can download a public version from  
> <https://www.kaggle.com/datasets/mirichoi0218/insurance>

### Run locally

```bash
pip install -r requirements.txt
python train_and_save.py   # only needed once
streamlit run app.py
```

Open <http://localhost:8501> in your browser.

---

## Deploy on Hugging Face Spaces

[Hugging Face Spaces](https://huggingface.co/spaces) supports Streamlit apps
with zero extra configuration.

### Step 1 – Train and export the model locally

```bash
pip install -r requirements.txt
python train_and_save.py
```

This creates `model.joblib` (~a few MB for the default Random Forest).

### Step 2 – Create a new Space

1. Go to <https://huggingface.co/spaces> and click **Create new Space**.
2. Fill in a name (e.g., `medical-insurance-cost`).
3. Select **Streamlit** as the Space SDK.
4. Set visibility to **Public** or **Private**.
5. Click **Create Space** – Hugging Face creates a git repository for the Space.

### Step 3 – Push your files

Clone the Space repo and copy the required files:

```bash
git clone https://huggingface.co/spaces/<your-hf-username>/medical-insurance-cost
cd medical-insurance-cost

# Copy files from this repo
cp /path/to/Medical-Insurance-Cost-Prediction/app.py .
cp /path/to/Medical-Insurance-Cost-Prediction/requirements.txt .
cp /path/to/Medical-Insurance-Cost-Prediction/model.joblib .
# Optional: copy the Streamlit theme config
mkdir -p .streamlit
cp /path/to/Medical-Insurance-Cost-Prediction/.streamlit/config.toml .streamlit/

git add app.py requirements.txt model.joblib .streamlit/
git commit -m "Add Streamlit app"
git push
```

Hugging Face will automatically build the Space and launch the app.  
Your app will be live at:  
`https://huggingface.co/spaces/<your-hf-username>/medical-insurance-cost`

### Alternatively – upload via the web UI

Open your Space → **Files** tab → upload:
- `app.py`
- `requirements.txt`
- `model.joblib`
- `.streamlit/config.toml` (optional)

Hugging Face will detect the Streamlit SDK from the Space settings and build
the app automatically.

### Notes

- `model.joblib` must be committed/uploaded to the Space alongside `app.py`.
  The app will display an error and stop if the file is missing.
- If `model.joblib` is larger than 10 MB, consider using
  [Git LFS](https://huggingface.co/docs/hub/repositories-getting-started#terminal)
  to track it (`git lfs track "model.joblib"`).
- The existing Jupyter notebooks are not required for the web app and can be
  omitted from the Space upload.

