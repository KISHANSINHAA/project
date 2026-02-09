import pandas as pd
import numpy as np


def forecast_next_30_days(
    model,
    last_known_data: pd.DataFrame,
    feature_columns: list
) -> pd.DataFrame:
    """
    Recursive 30-day forecasting using ML model.
    Updates only lag_1 and rolling features (safe & realistic).

    Parameters:
        model: trained ML model
        last_known_data: last row of feature-engineered data
        feature_columns: feature list used in training

    Returns:
        DataFrame with future dates and forecasted revenue
    """

    future_preds = []
    current_row = last_known_data.copy()

    for _ in range(30):
        X_input = current_row[feature_columns].values.reshape(1, -1)
        y_pred = model.predict(X_input)[0]

        future_preds.append(y_pred)

        # -----------------------------
        # Update lag_1 safely
        # -----------------------------
        if "lag_1" in current_row.columns:
            current_row["lag_1"] = y_pred

        # -----------------------------
        # Update rolling features safely
        # -----------------------------
        for col in current_row.columns:
            if "rolling" in col:
                window = int(col.split("_")[1])
                current_row[col] = (
                    (current_row[col] * (window - 1)) + y_pred
                ) / window

    start_date = last_known_data["date"].iloc[0] + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=start_date, periods=30)

    return pd.DataFrame({
        "date": future_dates,
        "forecasted_revenue": future_preds
    })
