import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv('insurance.csv')

# Encode categorical variables
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df['sex'] = le_sex.fit_transform(df['sex'])
df['smoker'] = le_smoker.fit_transform(df['smoker'])
df['region'] = le_region.fit_transform(df['region'])

# Separate features and target
X = df.drop('charges', axis=1)
y = df['charges']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'model.pkl')
print("✓ Model trained and saved as 'model.pkl'")
print(f"  Training samples: {len(X)}")
print(f"  Features: {list(X.columns)}")
print(f"  R² score: {model.score(X, y):.4f}")
