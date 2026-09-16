import pandas as pd


def load_data(data_path):
    return pd.read_csv(data_path)


def clean_data(df):
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].str.strip(),
        errors="coerce"
    )

    df.loc[df["tenure"].eq(0), "TotalCharges"] = 0.0

    return df


def prepare_features_and_target(df):
    X = df.drop(
        columns=["Churn", "customerID"]
    )

    y = df["Churn"]

    return X, y