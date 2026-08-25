import pandas as pd

print("Campus X ML Module")
print("Loading transportation dataset...")

df = pd.read_csv("data/raw/transportation_data.csv")

print("\nDataset loaded successfully!")
print(df)

print("\nConverting text data into numbers...")

df["traffic"] = df["traffic"].map({
    "Low": 0,
    "Medium": 1,
    "High": 2
})

df["weather"] = df["weather"].map({
    "Clear": 0,
    "Cloudy": 1,
    "Rain": 2
})

print("\nProcessed dataset:")
print(df)
print("\nProcessed dataset:")
print(df)
