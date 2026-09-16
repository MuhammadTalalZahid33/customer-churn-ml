import pandas as pd

from src.preprocessing import (
    clean_data,
    prepare_features_and_target,
)


def test_clean_data_converts_total_charges():
    df = pd.DataFrame({
        "tenure": [0, 5],
        "TotalCharges": [" ", "427.5"]
    })

    result = clean_data(df)

    assert result["TotalCharges"].isna().sum() == 0
    assert result.loc[0, "TotalCharges"] == 0.0
    assert result.loc[1, "TotalCharges"] == 427.5


def test_prepare_features_and_target():
    df = pd.DataFrame({
        "customerID": ["A", "B"],
        "tenure": [5, 10],
        "Churn": ["Yes", "No"]
    })

    X, y = prepare_features_and_target(df)

    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert len(X) == 2
    assert len(y) == 2