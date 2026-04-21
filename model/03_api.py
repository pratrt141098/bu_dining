from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import joblib
import json
import numpy as np

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ff86d35e-29d3-4976-a495-1de68cd43f07.lovableproject.com",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load models + encoders once at startup
wait_model = joblib.load("wait_model.joblib")
occ_model = joblib.load("occ_model.joblib")
le_hall = joblib.load("le_hall.joblib")
le_meal = joblib.load("le_meal.joblib")

with open("model_meta.json") as f:
    meta = json.load(f)

HALL_CAPACITIES = {
    "Marciano Commons": 800,
    "Stuvi2 / towers": 400,
    "Sargent Choice Café": 300,
    "Warren Towers Dining": 600,
    "West Campus Dining": 500,
}

def get_meal_period(hour: int) -> str:
    if 6 <= hour < 11:
        return "breakfast"
    elif 11 <= hour < 16:
        return "lunch"
    elif 16 <= hour < 21:
        return "dinner"
    return "closed"

def occupancy_status(rate: float) -> str:
    if rate < 0.60:
        return "Normal"
    elif rate < 0.85:
        return "Busy"
    return "High"


class PredictRequest(BaseModel):
    hall_name: str
    timestamp: str | None = None  # ISO format, defaults to now


class PredictResponse(BaseModel):
    hall_name: str
    timestamp: str
    meal_period: str
    predicted_wait_sec: float
    predicted_wait_min: float
    predicted_occupancy_rate: float
    occupancy_pct: int
    status: str
    confidence_note: str


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    ts = datetime.fromisoformat(req.timestamp) if req.timestamp else datetime.now()

    hour = ts.hour
    minute = ts.minute
    hour_float = hour + minute / 60
    day_of_week = ts.weekday()
    is_weekend = int(day_of_week in [5, 6])
    meal_period = get_meal_period(hour)

    if req.hall_name not in le_hall.classes_:
        return {"error": f"Unknown hall: {req.hall_name}. Options: {list(le_hall.classes_)}"}

    hall_enc = le_hall.transform([req.hall_name])[0]
    meal_enc = le_meal.transform([meal_period])[0]

    occ_features = np.array([[hall_enc, meal_enc, hour_float, day_of_week, is_weekend]])
    occ_pred = float(np.clip(occ_model.predict(occ_features)[0], 0, 1))

    wait_features = np.array([[hall_enc, meal_enc, hour_float, day_of_week, is_weekend, occ_pred]])
    wait_pred = float(np.clip(wait_model.predict(wait_features)[0], 0, 1200))

    return PredictResponse(
        hall_name=req.hall_name,
        timestamp=ts.isoformat(),
        meal_period=meal_period,
        predicted_wait_sec=round(wait_pred, 1),
        predicted_wait_min=round(wait_pred / 60, 1),
        predicted_occupancy_rate=round(occ_pred, 3),
        occupancy_pct=int(occ_pred * 100),
        status=occupancy_status(occ_pred),
        confidence_note="Predicted from synthetic swipe data. 15-min granularity.",
    )


@app.get("/predict/all")
def predict_all(timestamp: str | None = None):
    """Returns predictions for all 5 halls at the given time (defaults to now)."""
    results = []
    for hall in HALL_CAPACITIES:
        req = PredictRequest(hall_name=hall, timestamp=timestamp)
        results.append(predict(req))
    results.sort(key=lambda x: x.predicted_wait_sec)
    return {"timestamp": timestamp or datetime.now().isoformat(), "halls": results}


@app.get("/health")
def health():
    return {"status": "ok", "halls": list(le_hall.classes_), "meals": list(le_meal.classes_)}