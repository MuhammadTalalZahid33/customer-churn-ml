from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "churn_model.pkl"

METADATA_PATH = MODEL_DIR / "churn_model_metadata.pkl"

RANDOM_STATE = 42

TEST_SIZE = 0.2