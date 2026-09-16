import joblib
import pandas as pd

from src.config import MODEL_PATH, METADATA_PATH


def load_model():
    return joblib.load(MODEL_PATH)


def load_metadata():
    return joblib.load(METADATA_PATH)


def predict(customer):
    model = load_model()
    metadata = load_metadata()

    df = pd.DataFrame([customer])

    probability = model.predict_proba(df)[0, 1]

    threshold = metadata["threshold"]

    prediction = (
        "Yes"
        if probability >= threshold
        else "No"
    )

    return {
        "prediction": prediction,
        "probability": float(probability)
    }


if __name__ == "__main__":

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 427.5
    }

    result = predict(customer)

    print("Prediction:", result["prediction"])
    print("Probability:", result["probability"])