import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
# Features used by the ML model
X = df[
    [
        "passengers",
        "speed_kmph",
        "traffic",
        "weather"
    ]
]

# Target that the model will predict
y = df["delay_minutes"]

print("\nFeatures:")
print(X)

print("\nTarget:")
print(y)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
# Create the ML model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("\nCampus X ML model trained successfully!")
