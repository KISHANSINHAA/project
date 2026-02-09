import pandas as pd
import joblib
from metrics import calculate_regression_metrics


def load_test_data():
    """
    Load test datasets.

    Returns:
        X_test (pd.DataFrame)
        y_test (pd.Series)
    """

    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    return X_test, y_test


def load_models():
    """
    Load trained ML models.

    Returns:
        dict: Model name -> model object
    """

    models = {
        "Random Forest": joblib.load("models/random_forest.pkl"),
        "XGBoost": joblib.load("models/xgboost.pkl")
    }

    return models


def evaluate_models() -> pd.DataFrame:
    """
    Evaluate all ML models and return comparison table.

    Returns:
        pd.DataFrame: Evaluation metrics for each model
    """

    X_test, y_test = load_test_data()
    models = load_models()

    results = []

    for model_name, model in models.items():
        predictions = model.predict(X_test)
        metrics = calculate_regression_metrics(y_test, predictions)

        results.append({
            "Model": model_name,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "R2": metrics["R2"]
        })

    return pd.DataFrame(results)


# Optional: Run as script
if __name__ == "__main__":
    eval_df = evaluate_models()
    print("\n📊 Model Evaluation Results")
    print(eval_df.round(3))
