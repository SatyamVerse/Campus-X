import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

print("========================================")
print("       CAMPUS X ML MODULE")
print("========================================")

# ========================================
# 1. LOAD DATASET
# ========================================

print("\nLoading transportation dataset...")

df = pd.read_csv("data/raw/transportation_data.csv")

print("\nDataset loaded successfully!")
print(df)

# ========================================
# 2. DATA PREPROCESSING
# ========================================

print("\nConverting text data into numbers...")

# Traffic encoding
df["traffic"] = df["traffic"].map({
    "Low": 0,
    "Medium": 1,
    "High": 2
})

# Weather encoding
df["weather"] = df["weather"].map({
    "Clear": 0,
    "Cloudy": 1,
    "Rain": 2
})

# Time period encoding
df["time_period"] = df["time_period"].map({
    "Morning": 0,
    "Afternoon": 1,
    "Evening": 2,
    "Night": 3
})

# Day type encoding
df["day_type"] = df["day_type"].map({
    "Weekday": 0,
    "Weekend": 1
})

print("\nProcessed dataset:")
print(df)

# ========================================
# 3. SELECT FEATURES of Entities
# ========================================

X = df[
    [
        "passengers",
        "speed_kmph",
        "traffic",
        "weather",
        "time_period",
        "day_type",
        "congestion_score"
    ]
]

# ========================================
# 4. SELECT TARGET
# ========================================

y = df["delay_minutes"]

print("\nFeatures:")
print(X)

print("\nTarget:")
print(y)

# ========================================
# 5. TRAIN / TEST SPLIT
# ========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ========================================
# 6. CREATE RANDOM FOREST MODEL
# ========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# ========================================
# 7. TRAIN MODEL
# ========================================

model.fit(X_train, y_train)

print("\nCampus X ML model trained successfully!")

# ========================================
# 8. NEW BUS SCENARIO
# ========================================

new_bus = pd.DataFrame({
    "passengers": [40],
    "speed_kmph": [20],
    "traffic": [2],
    "weather": [2],
    "time_period": [2],
    "day_type": [0],
    "congestion_score": [75]
})

# ========================================
# 9. PREDICT DELAY
# ========================================

predicted_delay = model.predict(new_bus)

print("\n========================================")
print("       CAMPUS X PREDICTION")
print("========================================")

print("\nNew bus situation:")
print(new_bus)

print(
    f"\nPredicted delay: "
    f"{predicted_delay[0]:.2f} minutes"
)
