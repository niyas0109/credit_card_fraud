import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# IMPORTANT: Exactly 'app = FastAPI()' (capital F, capital API, no space in class name)
app = FastAPI(title="Credit Card Fraud Detection API")

MODEL_PATH = "fraud_model.joblib"

# Load model pipeline safely
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


class TransactionData(BaseModel):
    amount: float
    transaction_hour: int
    merchant_category: str
    foreign_transaction: int
    location_mismatch: int
    device_trust_score: float
    velocity_last_24h: int
    cardholder_age: int


@app.get("/")
def home():
    return {"status": "online", "message": "Fraud Detection API Operational"}


@app.post("/predict")
def predict_fraud(data: TransactionData):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model file 'fraud_model.joblib' not found. Run 'python train_model.py' first.",
        )

    # Convert request payload into DataFrame matching trained features
    input_df = pd.DataFrame([data.model_dump()])

    # Get fraud probability
    fraud_proba = model.predict_proba(input_df)[0][1]
    is_fraud = int(fraud_proba >= 0.5)

    # Calculate risk category
    if fraud_proba >= 0.7:
        risk_level = "HIGH RISK"
    elif fraud_proba >= 0.3:
        risk_level = "MEDIUM RISK"
    else:
        risk_level = "LOW RISK"

    return {
        "fraud_probability": round(float(fraud_proba), 4),
        "is_fraud": is_fraud,
        "risk_level": risk_level,
    }