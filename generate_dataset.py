import pandas as pd
import random

# Number of farmers to generate
NUM_FARMERS = 20000

crop_types = [
    "Rice",
    "Wheat",
    "Sugarcane",
    "Cotton",
    "Maize",
    "Millets"
]

soil_types = [
    "Clay",
    "Loam",
    "Sandy",
    "Black"
]

data = []

for _ in range(NUM_FARMERS):

    area = round(random.uniform(0.5, 15), 2)

    crop = random.choice(crop_types)

    soil = random.choice(soil_types)

    gps = random.choice([0, 1])

    documents = random.choice([0, 1])

    ndvi = round(random.uniform(0.2, 0.95), 2)

    rainfall = random.randint(300, 1500)

    # Verification Rule
    if area >= 2 and gps == 1 and documents == 1 and ndvi >= 0.50:
        verified = 1
    else:
        verified = 0

    data.append([
        area,
        crop,
        soil,
        gps,
        documents,
        ndvi,
        rainfall,
        verified
    ])

df = pd.DataFrame(data, columns=[
    "Area",
    "Crop",
    "Soil",
    "GPS",
    "Documents",
    "NDVI",
    "Rainfall",
    "Verified"
])

df.to_csv("dataset/farmers.csv", index=False)

print(df.head())
print("\nDataset Created Successfully!")