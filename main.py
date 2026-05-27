# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS (로컬 HTML에서 호출할 수 있게 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 개발용이라 * 허용
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드
model = joblib.load("output/storeguard_rf_model.joblib")

class StoreInput(BaseModel):
    region: str
    category: str
    daily_foot_traffic: int
    competitors: int
    vacancy_rate: float
    trend_index: int
    rent_monthly: float
    expected_sales: float
    equity_ratio: float
    experience_level: int
    franchise_scale: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(input: StoreInput):
    # 파생 변수 계산
    rent_to_sales_ratio = input.rent_monthly / max(input.expected_sales, 1) * 100

    df = pd.DataFrame([{
        "region": input.region,
        "category": input.category,
        "daily_foot_traffic": input.daily_foot_traffic,
        "competitors": input.competitors,
        "vacancy_rate": input.vacancy_rate,
        "trend_index": input.trend_index,
        "rent_monthly": input.rent_monthly,
        "expected_sales": input.expected_sales,
        "equity_ratio": input.equity_ratio,
        "experience_level": input.experience_level,
        "franchise_scale": input.franchise_scale,
        "rent_to_sales_ratio": rent_to_sales_ratio
    }])

    proba = float(model.predict_proba(df)[0, 1])
    risk_score = int(round(proba * 100))

    # 등급 규칙
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
        "prob_1y_closure": proba,
        "grade": grade
    }