import joblib
import pandas as pd

model = joblib.load("models/carbon_model.pkl")
crop_encoder = joblib.load("models/carbon_crop_encoder.pkl")
soil_encoder = joblib.load("models/carbon_soil_encoder.pkl")

crop = "Rice"
soil = "Clay"

farm = pd.DataFrame([{
    "Area": 5,
    "Crop": crop_encoder.transform([crop])[0],
    "Soil": soil_encoder.transform([soil])[0],
    "NDVI": 0.80,
    "Rainfall": 900
}])

prediction = model.predict(farm)

print(f"Estimated Carbon Absorption: {prediction[0]:.2f} tons/year")