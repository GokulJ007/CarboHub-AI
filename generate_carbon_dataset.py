import pandas as pd
import random

NUM_FARMS = 20000

crop_types = {
    "Rice": 0.8,
    "Wheat": 0.7,
    "Sugarcane": 1.5,
    "Cotton": 0.6,
    "Maize": 0.9,
    "Millets": 0.5
}

soil_types = {
    "Clay": 1.0,
    "Loam": 1.2,
    "Sandy": 0.8,
    "Black": 1.1
}

data = []

for _ in range(NUM_FARMS):

    area = round(random.uniform(1, 15), 2)
    crop = random.choice(list(crop_types.keys()))
    soil = random.choice(list(soil_types.keys()))
    ndvi = round(random.uniform(0.3, 0.95), 2)
    rainfall = random.randint(300, 1500)

    crop_factor = crop_types[crop]
    soil_factor = soil_types[soil]

    # Synthetic carbon formula
    carbon = (
        area
        * crop_factor
        * soil_factor
        * ndvi
        * (rainfall / 1000)
    )

    carbon = round(carbon, 2)

    data.append([
        area,
        crop,
        soil,
        ndvi,
        rainfall,
        carbon
    ])

df = pd.DataFrame(data, columns=[
    "Area",
    "Crop",
    "Soil",
    "NDVI",
    "Rainfall",
    "Carbon"
])

df.to_csv("dataset/carbon_dataset.csv", index=False)

print(df.head())
print("\nCarbon Dataset Created!")