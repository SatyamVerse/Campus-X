import pandas as pd

print("Campus X Transportation AI")
print("Creating transportation data...")

data = {
    "bus_id": ["B01", "B02", "B03", "B04", "B05"],
    "route": ["R01", "R02", "R03", "R01", "R02"],
    "passengers": [35, 42, 18, 50, 27],
    "speed_kmph": [32, 25, 40, 18, 35],
    "traffic": ["Low", "Medium", "Low", "High", "Medium"],
    "weather": ["Clear", "Clear", "Rain", "Rain", "Cloudy"],
    "delay_minutes": [0, 3, 1, 8, 4]
}

df = pd.DataFrame(data)

print("\nTransportation Data:")
print(df)
df.to_csv("data/raw/transportation_data.csv", index=False)

print("\nDataset Saved Successfully!")
