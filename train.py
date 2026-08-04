import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset/farmers.csv")

# Convert text columns into numbers
crop_encoder = LabelEncoder()
soil_encoder = LabelEncoder()

df["Crop"] = crop_encoder.fit_transform(df["Crop"])
df["Soil"] = soil_encoder.fit_transform(df["Soil"])

# Features (inputs)
X = df.drop("Verified", axis=1)

# Target (output)
y = df["Verified"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy*100:.2f}%")

# Save model
joblib.dump(model, "models/farmer_verification_model.pkl")
joblib.dump(crop_encoder, "models/crop_encoder.pkl")
joblib.dump(soil_encoder, "models/soil_encoder.pkl")

print("Model Saved Successfully!")