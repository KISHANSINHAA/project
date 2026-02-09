import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def load_feature_data(path: str) -> pd.DataFrame:
    """
    Load feature-engineered dataset.

    Parameters:
        path (str): Path to feature CSV

    Returns:
        pd.DataFrame
    """
    return pd.read_csv(path, parse_dates=["date"])


def time_series_split(df: pd.DataFrame, split_ratio: float = 0.8):
    """
    Perform time-based train-test split.

    Parameters:
        df (pd.DataFrame): Feature dataframe
        split_ratio (float): Train ratio

    Returns:
        X_train, X_test, y_train, y_test
    """

    split_index = int(len(df) * split_ratio)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train = train_df.drop(columns=["date", "daily_revenue"])
    y_train = train_df["daily_revenue"]

    X_test = test_df.drop(columns=["date", "daily_revenue"])
    y_test = test_df["daily_revenue"]

    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    """
    Train Random Forest regressor.
    """
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """
    Train XGBoost regressor.
    """
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def save_artifacts(X_train, X_test, y_train, y_test, rf_model, xgb_model):
    """
    Save datasets and trained models.
    """

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save datasets
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    # Save models
    joblib.dump(rf_model, "models/random_forest.pkl")
    joblib.dump(xgb_model, "models/xgboost.pkl")

    print("✅ Training artifacts saved successfully")


if __name__ == "__main__":
    FEATURE_PATH = "data/processed/daily_data_features.csv"

    df = load_feature_data(FEATURE_PATH)

    X_train, X_test, y_train, y_test = time_series_split(df)

    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    save_artifacts(X_train, X_test, y_train, y_test, rf_model, xgb_model)

    print("✅ ML model training completed")
