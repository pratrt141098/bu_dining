import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("bu_dining_swipes_week.csv")
df["ts"] = pd.to_datetime(df["swipe_ts"])
df["hour"] = df["ts"].dt.hour
df["minute"] = df["ts"].dt.minute
df["hour_float"] = df["hour"] + df["minute"] / 60
df["day_of_week"] = df["ts"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

le_hall = joblib.load("le_hall.joblib")
le_meal = joblib.load("le_meal.joblib")
wait_model = joblib.load("wait_model.joblib")
occ_model = joblib.load("occ_model.joblib")

with open("model_meta.json") as f:
    meta = json.load(f)

df["hall_enc"] = le_hall.transform(df["hall"])
df["meal_enc"] = le_meal.transform(df["meal_period"])

df["wait_pred"] = wait_model.predict(df[meta["features_wait"]])
df["occ_pred"] = occ_model.predict(df[meta["features_occ"]])

print("=== Wait Time MAE by Hall ===")
for hall in df["hall"].unique():
    sub = df[df["hall"] == hall]
    mae = mean_absolute_error(sub["wait_time_sec"], sub["wait_pred"])
    print(f"  {hall}: {mae:.1f} sec")

print("\n=== Occupancy MAE by Hall ===")
for hall in df["hall"].unique():
    sub = df[df["hall"] == hall]
    mae = mean_absolute_error(sub["occupancy_rate"], sub["occ_pred"])
    print(f"  {hall}: {mae:.3f}")

# Feature importance plot
importances = wait_model.feature_importances_
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(meta["features_wait"], importances, color="#00C896")
ax.set_title("Wait Time Model — Feature Importance")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("\nSaved: feature_importance.png")

# Predicted vs actual — Marciano lunch
marciano = df[(df["hall"] == "Marciano Commons") & (df["meal_period"] == "lunch")]
hourly = marciano.groupby("hour")[["wait_time_sec", "wait_pred"]].mean()

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(hourly.index, hourly["wait_time_sec"], label="Actual", color="#7a9e9b")
ax2.plot(hourly.index, hourly["wait_pred"], label="Predicted", color="#00C896", linestyle="--")
ax2.set_title("Marciano Commons — Actual vs Predicted Wait Time (Lunch)")
ax2.set_xlabel("Hour of Day")
ax2.set_ylabel("Wait Time (sec)")
ax2.legend()
plt.tight_layout()
plt.savefig("marciano_lunch_prediction.png", dpi=150)
print("Saved: marciano_lunch_prediction.png")