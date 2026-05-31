from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = joblib.load("storeguard_rf_model.joblib")

CUISINE_MAP = {"American": 0, "Chinese": 1, "Indian": 2, "Italian": 3, "Japanese": 4, "Mexican": 5, "Vegan": 6}

class RestaurantInput(BaseModel):
    cuisine_type: str
    average_meal_price: float
    seating_capacity: int
    staff_count: int
    delivery_service: str
    marketing_budget: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(input: RestaurantInput):
    features = np.array([[
        input.average_meal_price,
        input.seating_capacity,
        input.staff_count,
        input.marketing_budget,
        1 if input.delivery_service == "Yes" else 0,
        CUISINE_MAP.get(input.cuisine_type, 0)
    ]])

    proba = float(model.predict_proba(features)[0][1])
    risk_score = int(round((1 - proba) * 100))

    if risk_score < 20: grade = "A"
    elif risk_score < 35: grade = "B"
    elif risk_score < 50: grade = "C"
    elif risk_score < 65: grade = "D"
    elif risk_score < 80: grade = "E"
    else: grade = "F"

    return {"risk_score": risk_score, "prob_closure": round(1 - proba, 3), "grade": grade}
