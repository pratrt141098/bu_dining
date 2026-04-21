import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json

df = pd.read_csv("bu_dining_swipes_week.csv")

# Feature engineering
df["ts"] = pd.to_datetime(df["swipe_ts"])
df["hour"] = df["ts"].dt.hour
df["minute"] = df["ts"].dt.minute
df["hour_float"] = df["hour"] + df["minute"] / 60
df["day_of_week"] = df["ts"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# Encode categoricals
le_hall = LabelEncoder()
le_meal = LabelEncoder()
df["hall_enc"] = le_hall.fit_transform(df["hall"])
df["meal_enc"] = le_meal.fit_transform(df["meal_period"])

FEATURES = [
    "hall_enc",
    "meal_enc",
    "hour_float",
    "day_of_week",
    "is_weekend",
    "occupancy_rate",
]

# --- Model 1: wait time ---
X = df[FEATURES]
y_wait = df["wait_time_sec"]

X_train, X_test, y_train, y_test = train_test_split(X, y_wait, test_size=0.2, random_state=42)

wait_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
)
wait_model.fit(X_train, y_train)

y_pred = wait_model.predict(X_test)
print(f"Wait Time MAE: {mean_absolute_error(y_test, y_pred):.1f} sec")
print(f"Wait Time R²:  {r2_score(y_test, y_pred):.3f}")

# --- Model 2: occupancy rate ---
# Use lagged features — drop occupancy_rate from features for this model
OCC_FEATURES = [f for f in FEATURES if f != "occupancy_rate"]
y_occ = df["occupancy_rate"]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    df[OCC_FEATURES], y_occ, test_size=0.2, random_state=42
)

occ_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
)
occ_model.fit(X_train2, y_train2)

y_pred2 = occ_model.predict(X_test2)
print(f"Occupancy MAE: {mean_absolute_error(y_test2, y_pred2):.3f}")
print(f"Occupancy R²:  {r2_score(y_test2, y_pred2):.3f}")

# Save models + encoders
joblib.dump(wait_model, "wait_model.joblib")
joblib.dump(occ_model, "occ_model.joblib")
joblib.dump(le_hall, "le_hall.joblib")
joblib.dump(le_meal, "le_meal.joblib")

# Save label classes for the API to use
meta = {
    "hall_classes": list(le_hall.classes_),
    "meal_classes": list(le_meal.classes_),
    "features_wait": FEATURES,
    "features_occ": OCC_FEATURES,
}
with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved: wait_model.joblib, occ_model.joblib, le_hall.joblib, le_meal.joblib, model_meta.json")