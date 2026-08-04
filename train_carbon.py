import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("dataset/carbon_dataset.csv")

# =====================================================
# Encode Categorical Columns
# =====================================================

crop_encoder = LabelEncoder()
soil_encoder = LabelEncoder()

df["Crop"] = crop_encoder.fit_transform(df["Crop"])
df["Soil"] = soil_encoder.fit_transform(df["Soil"])

# =====================================================
# Features & Target
# =====================================================

X = df.drop("Carbon", axis=1)
y = df["Carbon"]

# =====================================================
# Train Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# Train Model
# =====================================================

model = RandomForestRegressor(
    n_estimators=50,      # Smaller model
    random_state=42,
    n_jobs=-1             # Faster training
)

model.fit(X_train, y_train)

# =====================================================
# Evaluation
# =====================================================

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("=" * 50)
print("Carbon Model Training Complete")
print("=" * 50)
print(f"MAE       : {mae:.4f}")
print(f"R2 Score  : {r2:.4f}")
print("=" * 50)

# =====================================================
# Save Compressed Models
# =====================================================

joblib.dump(
    model,
    "models/carbon_model.pkl",
    compress=3
)

joblib.dump(
    crop_encoder,
    "models/carbon_crop_encoder.pkl",
    compress=3
)

joblib.dump(
    soil_encoder,
    "models/carbon_soil_encoder.pkl",
    compress=3
)

print("Compressed Carbon Model Saved Successfully!")