import uuid
import joblib
import pandas as pd


# Load AI model
carbon_model = joblib.load("models/carbon_model.pkl")
crop_encoder = joblib.load("models/carbon_crop_encoder.pkl")
soil_encoder = joblib.load("models/carbon_soil_encoder.pkl")


def create_pool(required_carbon):

    # Load farmers
    farmers = pd.read_csv("dataset/carbon_dataset.csv")

    # Farmer IDs
    farmers["FarmerID"] = [
        f"F{i+1001}" for i in range(len(farmers))
    ]

    # Encode categorical data
    encoded = farmers.copy()

    encoded["Crop"] = crop_encoder.transform(encoded["Crop"])
    encoded["Soil"] = soil_encoder.transform(encoded["Soil"])

    # Predict carbon
    X = encoded[
        [
            "Area",
            "Crop",
            "Soil",
            "NDVI",
            "Rainfall"
        ]
    ]

    farmers["PredictedCarbon"] = carbon_model.predict(X)

    # --------------------------
    # AI Score
    # --------------------------

    farmers["VerificationConfidence"] = (
        farmers["NDVI"] * 100
    )

    farmers["AIScore"] = (

        farmers["PredictedCarbon"] * 0.45 +

        farmers["VerificationConfidence"] * 0.30 +

        farmers["NDVI"] * 100 * 0.15 +

        (farmers["Area"] / 15) * 100 * 0.10

    )

    # Rank farmers
    farmers = farmers.sort_values(
        by="AIScore",
        ascending=False
    )

    selected = []

    total = 0

    for _, farmer in farmers.iterrows():

        if total >= required_carbon:
            break

        selected.append({

            "FarmerID": farmer["FarmerID"],

            "Area": round(
                farmer["Area"],
                2
            ),

            "Carbon": round(
                farmer["PredictedCarbon"],
                2
            ),

            "AIScore": round(
                farmer["AIScore"],
                2
            )

        })

        total += farmer["PredictedCarbon"]

    efficiency = round(
        (required_carbon / total) * 100,
        2
    )

    return {

        "pool_id": uuid.uuid4().hex[:8].upper(),

        "required_carbon": required_carbon,

        "total_carbon": round(total, 2),

        "efficiency": efficiency,

        "selected_farmers": selected

    }