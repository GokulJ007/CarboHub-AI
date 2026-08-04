import pandas as pd
import joblib
import uuid

# -----------------------------
# Load AI Models
# -----------------------------
carbon_model = joblib.load("models/carbon_model.pkl")
carbon_crop_encoder = joblib.load("models/carbon_crop_encoder.pkl")
carbon_soil_encoder = joblib.load("models/carbon_soil_encoder.pkl")

verification_model = joblib.load("models/farmer_verification_model.pkl")
verify_crop_encoder = joblib.load("models/crop_encoder.pkl")
verify_soil_encoder = joblib.load("models/soil_encoder.pkl")

# -----------------------------
# Load Dataset
# -----------------------------
farmers = pd.read_csv("dataset/carbon_dataset.csv")

# Give Farmer IDs
farmers["FarmerID"] = [
    f"F{i+1001}" for i in range(len(farmers))
]

# -----------------------------
# Encode for Carbon Model
# -----------------------------
carbon_df = farmers.copy()

carbon_df["Crop"] = carbon_crop_encoder.transform(carbon_df["Crop"])
carbon_df["Soil"] = carbon_soil_encoder.transform(carbon_df["Soil"])

X = carbon_df[[
    "Area",
    "Crop",
    "Soil",
    "NDVI",
    "Rainfall"
]]

farmers["PredictedCarbon"] = carbon_model.predict(X)

# -----------------------------
# Simulate Verification Confidence
# -----------------------------
farmers["VerificationConfidence"] = (
    farmers["NDVI"] * 40 +
    (farmers["Rainfall"] / 1500) * 30 +
    (farmers["Area"] / 15) * 30
)

farmers["VerificationConfidence"] = farmers["VerificationConfidence"].clip(0,100)

# -----------------------------
# AI Score
# -----------------------------
farmers["AIScore"] = (

    farmers["PredictedCarbon"] * 40 +

    farmers["VerificationConfidence"] * 0.3 +

    farmers["NDVI"] * 100 * 0.2 +

    (farmers["Area"]/15)*100*0.1

)

# -----------------------------
# Rank Farmers
# -----------------------------
farmers = farmers.sort_values(
    by="AIScore",
    ascending=False
)

# -----------------------------
# Pooling
# -----------------------------
required_carbon = 120

selected = []

total_carbon = 0

for _, farmer in farmers.iterrows():

    if total_carbon >= required_carbon:
        break

    selected.append({

        "FarmerID": farmer["FarmerID"],

        "Carbon": round(
            farmer["PredictedCarbon"],
            2
        ),

        "AI Score": round(
            farmer["AIScore"],
            2
        )

    })

    total_carbon += farmer["PredictedCarbon"]

# -----------------------------
# Pool Statistics
# -----------------------------
efficiency = (
    required_carbon /
    total_carbon
) * 100

print("="*50)

print("POOL CREATED")

print("="*50)

print(f"Pool ID : {uuid.uuid4().hex[:8].upper()}")

print(f"Industry Requirement : {required_carbon:.2f} Tons")

print(f"Total Carbon : {total_carbon:.2f} Tons")

print(f"Pool Efficiency : {efficiency:.2f}%")

print()

print("Selected Farmers")

print()

for farmer in selected:

    print(farmer)

print()

print("="*50)