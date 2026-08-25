import pandas as pd
import numpy as np

print("========================================")
print("       CAMPUS X DATA GENERATOR")
print("========================================")

# Make results reproducible
np.random.seed(42)

# Number of transportation records
NUM_RECORDS = 10000

print(f"\nGenerating {NUM_RECORDS} transportation records...")

# -----------------------------
# Basic transportation data
# -----------------------------

bus_ids = [f"B{str(i).zfill(3)}" for i in range(1, 51)]

routes = ["R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08"]

traffic_levels = ["Low", "Medium", "High"]

weather_conditions = [
    "Clear",
    "Cloudy",
    "Rain"
]

time_periods = [
    "Morning",
    "Afternoon",
    "Evening",
    "Night"
]

day_types = [
    "Weekday",
    "Weekend"
]

# -----------------------------
# Generate random data
# -----------------------------

bus_id = np.random.choice(
    bus_ids,
    NUM_RECORDS
)

route = np.random.choice(
    routes,
    NUM_RECORDS
)

passengers = np.random.randint(
    5,
    61,
    NUM_RECORDS
)

traffic = np.random.choice(
    traffic_levels,
    NUM_RECORDS,
    p=[0.35, 0.40, 0.25]
)

weather = np.random.choice(
    weather_conditions,
    NUM_RECORDS,
    p=[0.55, 0.25, 0.20]
)

time_period = np.random.choice(
    time_periods,
    NUM_RECORDS
)

day_type = np.random.choice(
    day_types,
    NUM_RECORDS,
    p=[0.75, 0.25]
)

# -----------------------------
# Generate speed
# -----------------------------

base_speed = np.random.randint(
    20,
    51,
    NUM_RECORDS
)

traffic_penalty = np.select(
    [
        traffic == "Low",
        traffic == "Medium",
        traffic == "High"
    ],
    [
        0,
        8,
        16
    ]
)

weather_penalty = np.where(
    weather == "Rain",
    7,
    0
)

speed_kmph = (
    base_speed
    - traffic_penalty
    - weather_penalty
)

speed_kmph = np.clip(
    speed_kmph,
    8,
    60
)

# -----------------------------
# Generate congestion score
# -----------------------------

traffic_score = np.select(
    [
        traffic == "Low",
        traffic == "Medium",
        traffic == "High"
    ],
    [
        1,
        2,
        3
    ]
)

congestion_score = (
    traffic_score * 20
    + passengers * 0.5
    + np.random.normal(0, 5, NUM_RECORDS)
)

congestion_score = np.clip(
    congestion_score,
    0,
    100
)

# -----------------------------
# Generate realistic delay
# -----------------------------

delay_minutes = (
    traffic_score * 1.8
    + passengers * 0.04
    + np.where(weather == "Rain", 2.5, 0)
    + np.where(time_period == "Morning", 1.5, 0)
    + np.where(time_period == "Evening", 2.0, 0)
    + np.where(day_type == "Weekday", 1.0, 0)
    + (50 - speed_kmph) * 0.08
    + np.random.normal(0, 1.5, NUM_RECORDS)
)

delay_minutes = np.clip(
    delay_minutes,
    0,
    30
)

delay_minutes = np.round(
    delay_minutes,
    2
)

# -----------------------------
# Create DataFrame
# -----------------------------

df = pd.DataFrame({
    "bus_id": bus_id,
    "route": route,
    "passengers": passengers,
    "speed_kmph": speed_kmph,
    "traffic": traffic,
    "weather": weather,
    "time_period": time_period,
    "day_type": day_type,
    "congestion_score": np.round(
        congestion_score,
        2
    ),
    "delay_minutes": delay_minutes
})

# -----------------------------
# Save dataset
# -----------------------------

output_path = (
    "data/raw/transportation_data.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("\nDataset generated successfully!")

print(f"Records: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nSample data:")
print(df.head())

print("\nDataset saved to:")
print(output_path)