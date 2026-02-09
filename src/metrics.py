import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_regression_metrics(y_true, y_pred) -> dict:
    """
    Calculate regression evaluation metrics.

    Parameters:
        y_true (array-like): Actual values
        y_pred (array-like): Predicted values

    Returns:
        dict: RMSE, MAE, R2
    """

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }
