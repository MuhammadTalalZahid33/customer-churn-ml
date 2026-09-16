import joblib

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    METADATA_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)

from src.preprocessing import (
    load_data,
    clean_data,
    prepare_features_and_target,
)

def build_model(X_train):
    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000
                )
            )
        ]
    )

    return model


def main():
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data(DATA_PATH)

    df = clean_data(df)

    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = build_model(X_train)

    model.fit(X_train, y_train)

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("Training completed.")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()