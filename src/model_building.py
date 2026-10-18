import logging
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


LOG_DIR = "logs"
MODEL_PATH = os.path.join("models", "model.pkl")
TRAIN_DATA_PATH = os.path.join("data", "processed_data", "train_tfidf.csv")
TEST_DATA_PATH = os.path.join("data", "processed_data", "test_tfidf.csv")
TARGET_COLUMN = "target"

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "model_building.log")
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load a processed dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            raise ValueError(f"Dataset is empty: {file_path}")
        logger.info("Loaded %s with shape %s", file_path, data.shape)
        return data
    except Exception:
        logger.exception("Failed to load dataset from %s", file_path)
        raise


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any] | None = None,
) -> RandomForestClassifier:
    """Train a random forest classifier using the supplied training data."""
    if X_train.shape[0] == 0:
        raise ValueError("Training data must contain at least one sample.")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("Training features and labels must have the same number of samples.")

    model_params = params or {}
    model = RandomForestClassifier(
        n_estimators=model_params.get("n_estimators", 100),
        random_state=model_params.get("random_state", 42),
        n_jobs=model_params.get("n_jobs", -1),
    )

    logger.info("Training RandomForestClassifier on %d samples", X_train.shape[0])
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Evaluate the trained model and log its test-set metrics."""
    if X_test.shape[0] == 0:
        raise ValueError("Test data must contain at least one sample.")
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError("Test features and labels must have the same number of samples.")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logger.info("Test accuracy: %.4f", accuracy)
    logger.info(
        "Classification report:\n%s",
        classification_report(y_test, predictions, zero_division=0),
    )
    return accuracy


def save_model(model: RandomForestClassifier, file_path: str) -> None:
    """Serialize the trained model to disk."""
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    try:
        with open(file_path, "wb") as model_file:
            pickle.dump(model, model_file)
        logger.info("Model saved to %s", file_path)
    except Exception:
        logger.exception("Failed to save model to %s", file_path)
        raise


def _split_features_and_target(
    data: pd.DataFrame,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"{dataset_name} dataset is missing '{TARGET_COLUMN}'.")

    features = data.drop(columns=[TARGET_COLUMN])
    if features.empty or features.shape[1] == 0:
        raise ValueError(f"{dataset_name} dataset has no feature columns.")
    return features, data[TARGET_COLUMN]


def main() -> None:
    """Train, evaluate, and persist the spam-classification model."""
    try:
        params = {"n_estimators": 25, "random_state": 2}
        train_data = load_data(TRAIN_DATA_PATH)
        test_data = load_data(TEST_DATA_PATH)

        X_train, y_train = _split_features_and_target(train_data, "Training")
        X_test, y_test = _split_features_and_target(test_data, "Test")

        if not X_train.columns.equals(X_test.columns):
            raise ValueError(
                "Training and test datasets must have identical feature columns."
            )

        model = train_model(X_train.to_numpy(), y_train.to_numpy(), params)
        evaluate_model(model, X_test.to_numpy(), y_test.to_numpy())
        save_model(model, MODEL_PATH)
    except Exception:
        logger.exception("Model-building pipeline failed")
        raise


if __name__ == "__main__":
    main()
