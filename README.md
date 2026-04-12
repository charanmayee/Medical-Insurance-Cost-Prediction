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
Example input:input_data = (31,1,25.74,0,1,0)
- Age: 31 
- BMI: 25.74 
- Children: 2  
- Smoker: No  
- Sex: Female  
- Region: Northwest  

Output:
- Predicted charges: `$XXXX.XX`

## Results
Add your final model and scores here, e.g.:
- Best model: RandomForestRegressor
- Test RMSE: 4,200
- Test R²: 0.86

## Future Improvements
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Feature engineering (e.g., BMI categories, interaction terms)
- Model explainability (SHAP / permutation importance)
- Deploy as a web app (Streamlit / Flask)

