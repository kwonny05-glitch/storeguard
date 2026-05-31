from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("output/storeguard_rf_model.joblib")

class RestaurantInput(BaseModel):
    city: str
    cuisine_type: str
    average_meal_price: float
    seating_capacity: int
    years_in_business: int
    google_rating: float
    social_media_followers: int
    weekend_reservations: int
    staff_count: int
    delivery_service: str
    marketing_budget: float
    health_inspection_score: float
    annual_revenue: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(input: RestaurantInput):
    df = pd.DataFrame([{
        "City": input.city,
        "Cuisine_Type": input.cuisine_type,
        "Average_Meal_Price": input.average_meal_price,
        "Seating_Capacity": input.seating_capacity,
        "Years_in_Business": input.years_in_business,
        "Google_Rating": input.google_rating,
        "Social_Media_Followers": input.social_media_followers,
        "Weekend_Reservations": input.weekend_reservations,
        "Staff_Count": input.staff_count,
        "Delivery_Service": input.delivery_service,
        "Marketing_Budget": input.marketing_budget,
        "Health_Inspection_Score": input.health_inspection_score,
        "Annual_Revenue": input.annual_revenue
    }])

    proba = float(model.predict_proba(df)[0, 1])
    risk_score = int(round((1 - proba) * 100))  # 성공확률 → 실패리스크로 변환

    if risk_score < 20:
        grade = "A"
    elif risk_score < 35:
        grade = "B"
    elif risk_score < 50:
        grade = "C"
    elif risk_score < 65:
        grade = "D"
    elif risk_score < 80:
        grade = "E"
    else:
        grade = "F"

    return {
        "risk_score": risk_score,
        "prob_1y_closure": round(1 - proba, 3),
        "grade": grade
    }
