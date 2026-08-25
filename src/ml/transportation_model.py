import pandas as pd

print("Campus X ML Module")
print("Loading transportation dataset...")

df = pd.read_csv("data/raw/transportation_data.csv")

print("\nDataset loaded successfully!")
print(df)