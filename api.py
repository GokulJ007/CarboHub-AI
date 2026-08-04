from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pool_service import create_pool
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CarboHub AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For hackathon only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ======================================================
# Load Models
# ======================================================

verification_model = joblib.load("models/farmer_verification_model.pkl")
crop_encoder = joblib.load("models/crop_encoder.pkl")
soil_encoder = joblib.load("models/soil_encoder.pkl")

carbon_model = joblib.load("models/carbon_model.pkl")
carbon_crop_encoder = joblib.load("models/carbon_crop_encoder.pkl")
carbon_soil_encoder = joblib.load("models/carbon_soil_encoder.pkl")

# ======================================================
# Request Models
# ======================================================

class Farmer(BaseModel):
    area: float
    crop: str
    soil: str
    gps: int
    documents: int
    ndvi: float
    rainfall: int


class CarbonRequest(BaseModel):
    area: float
    crop: str
    soil: str
    ndvi: float
    rainfall: int


class PoolRequest(BaseModel):
    required_carbon: float


# ======================================================
# Farmer Verification
# ======================================================

@app.post("/verify")
def verify_farmer(farmer: Farmer):

    crop = crop_encoder.transform([farmer.crop])[0]
    soil = soil_encoder.transform([farmer.soil])[0]

    df = pd.DataFrame([{
        "Area": farmer.area,
        "Crop": crop,
        "Soil": soil,
        "GPS": farmer.gps,
        "Documents": farmer.documents,
        "NDVI": farmer.ndvi,
        "Rainfall": farmer.rainfall
    }])

    prediction = verification_model.predict(df)[0]

    probability = verification_model.predict_proba(df)[0]
    confidence = round(max(probability) * 100, 2)

    reasons = []

    if farmer.gps == 1:
        reasons.append("GPS Verified")

    if farmer.documents == 1:
        reasons.append("Documents Uploaded")

    if farmer.ndvi >= 0.7:
        reasons.append("Healthy Vegetation")
    elif farmer.ndvi >= 0.5:
        reasons.append("Moderate Vegetation")
    else:
        reasons.append("Low Vegetation")

    if farmer.rainfall >= 800:
        reasons.append("Adequate Rainfall")

    if confidence >= 95 and farmer.ndvi >= 0.75 and farmer.documents == 1 and farmer.gps == 1:
        risk = "Low"
    elif confidence >= 80:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "verified": bool(prediction),
        "verification_confidence": confidence,
        "risk": risk,
        "reasons": reasons
    }


# ======================================================
# Carbon Prediction
# ======================================================

@app.post("/predict-carbon")
def predict_carbon(farm: CarbonRequest):

    crop = carbon_crop_encoder.transform([farm.crop])[0]
    soil = carbon_soil_encoder.transform([farm.soil])[0]

    df = pd.DataFrame([{
        "Area": farm.area,
        "Crop": crop,
        "Soil": soil,
        "NDVI": farm.ndvi,
        "Rainfall": farm.rainfall
    }])

    carbon = float(carbon_model.predict(df)[0])

    if carbon >= 10:
        grade = "Excellent"
    elif carbon >= 7:
        grade = "High"
    elif carbon >= 4:
        grade = "Medium"
    else:
        grade = "Low"

    if carbon >= 10:
        recommendation = "Highly Recommended"
    elif carbon >= 7:
        recommendation = "Recommended"
    elif carbon >= 4:
        recommendation = "Average"
    else:
        recommendation = "Needs Improvement"

    quality_score = round(min((carbon / 10) * 100, 100), 2)

    return {
        "carbon_absorption": round(carbon, 2),
        "unit": "tons/year",
        "grade": grade,
        "quality_score": quality_score,
        "recommendation": recommendation
    }


# ======================================================
# Create Pool
# ======================================================

@app.post("/create-pool")
def create_pool_api(request: PoolRequest):

    return create_pool(request.required_carbon)


# ======================================================
# Home
# ======================================================

@app.get("/")
def home():

    return {
        "message": "Welcome to CarboHub AI API",
        "available_endpoints": [
            "/verify",
            "/predict-carbon",
            "/create-pool"
        ]
    }