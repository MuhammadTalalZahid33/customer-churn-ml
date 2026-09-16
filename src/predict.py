import joblib
import pandas as pd

from config import MODEL_PATH, METADATA_PATH


def load_model():
    return joblib.load(MODEL_PATH)


def load_metadata():
    return joblib.load(METADATA_PATH)


def predict(customer):
    model = load_model()
    metadata = load_metadata()

    df = pd.DataFrame([customer])

    probability = model.predict_proba(
        df
    )[0, 1]

    threshold = metadata["threshold"]

    prediction = (
        "Yes"
        if probability >= threshold
        else "No"
    )

    return {
        "prediction": prediction,
        "probability": probability,
    }