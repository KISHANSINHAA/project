import pandas as pd
import numpy as np


# ------------------------------------------------------------------
# Time / Calendar Features
# ------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["day_of_week"] = df["date"].dt.dayofweek          # 0 = Monday
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df


# ------------------------------------------------------------------
# Lag Features (Memory)
# ------------------------------------------------------------------
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    lags = [1, 7, 14, 30]

    for lag in lags:
        df[f"revenue_lag_{lag}"] = df["daily_revenue"].shift(lag)
        df[f"transactions_lag_{lag}"] = df["num_transactions"].shift(lag)
        df[f"quantity_lag_{lag}"] = df["total_quantity"].shift(lag)

    return df


# ------------------------------------------------------------------
# Rolling Statistics (Smoothing + Volatility)
# ------------------------------------------------------------------
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    windows = [7, 14, 30]

    for window in windows:
        df[f"revenue_roll_mean_{window}"] = (
            df["daily_revenue"].rolling(window).mean()
        )
        df[f"revenue_roll_std_{window}"] = (
            df["daily_revenue"].rolling(window).std()
        )

    return df


# ------------------------------------------------------------------
# Trend & Momentum Features (VERY IMPORTANT)
# ------------------------------------------------------------------
def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Absolute momentum
    df["revenue_diff_1"] = df["daily_revenue"] - df["daily_revenue"].shift(1)
    df["revenue_diff_7"] = df["daily_revenue"] - df["daily_revenue"].shift(7)

    # Percentage momentum
    df["revenue_pct_change_7"] = df["daily_revenue"].pct_change(7)

    return df


# ------------------------------------------------------------------
# Final Dataset Preparation
# ------------------------------------------------------------------
def prepare_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full ML feature engineering pipeline.
    """

    df = df.sort_values("date").copy()

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_trend_features(df)

    # Drop rows created by lag / rolling operations
    df = df.dropna().reset_index(drop=True)

    return df


# ------------------------------------------------------------------
# Script Execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    INPUT_PATH = "data/processed/daily_data.csv"
    OUTPUT_PATH = "data/processed/daily_data_features.csv"

    daily_df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    feature_df = prepare_ml_dataset(daily_df)

    feature_df.to_csv(OUTPUT_PATH, index=False)

    print("✅ Feature engineering completed successfully")
    print(f"📁 Saved to {OUTPUT_PATH}")
