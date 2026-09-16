from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

from src.config import MODEL_PATH, METADATA_PATH


app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML API for predicting customer churn",
    version="1.0.0"
)


model = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH)

THRESHOLD = metadata["threshold"]


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
def predict(customer: Customer):

    customer_data = customer.model_dump()

    df = pd.DataFrame([customer_data])

    probability = float(
        model.predict_proba(df)[0, 1]
    )

    prediction = (
        "Yes"
        if probability >= THRESHOLD
        else "No"
    )

    return {
        "prediction": prediction,
        "churn_probability": round(probability, 4),
        "threshold": THRESHOLD
    }