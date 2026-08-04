import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("dataset/carbon_dataset.csv")

# Encode text columns
crop_encoder = LabelEncoder()
soil_encoder = LabelEncoder()

df["Crop"] = crop_encoder.fit_transform(df["Crop"])
df["Soil"] = soil_encoder.fit_transform(df["Soil"])

# Features
X = df.drop("Carbon", axis=1)

# Target
y = df["Carbon"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# Save model
joblib.dump(model, "models/carbon_model.pkl")
joblib.dump(crop_encoder, "models/carbon_crop_encoder.pkl")
joblib.dump(soil_encoder, "models/carbon_soil_encoder.pkl")

print("Carbon Model Saved Successfully!")