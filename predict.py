import joblib
import pandas as pd

# Load saved model and encoders
model = joblib.load("models/farmer_verification_model.pkl")
crop_encoder = joblib.load("models/crop_encoder.pkl")
soil_encoder = joblib.load("models/soil_encoder.pkl")

# Sample farmer data
crop = "Rice"
soil = "Clay"

new_farmer = pd.DataFrame([{
    "Area": 5.5,
    "Crop": crop_encoder.transform([crop])[0],
    "Soil": soil_encoder.transform([soil])[0],
    "GPS": 1,
    "Documents": 1,
    "NDVI": 0.82,
    "Rainfall": 900
}])

# Predict
prediction = model.predict(new_farmer)

if prediction[0] == 1:
    print("✅ Farmer Verified")
else:
    print("❌ Farmer Rejected")