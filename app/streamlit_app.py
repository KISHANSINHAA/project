import sys
import os

# -------------------------------------------------
# Add src/ to Python path
# -------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from evaluate_models import evaluate_models
from forecast_30_days import forecast_next_30_days

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Retail Sales Forecasting Dashboard",
    layout="wide"
)

st.title("📈 Retail Sales Forecasting Dashboard")
st.markdown(
    """
    This dashboard demonstrates **machine learning–based daily revenue forecasting**
    using historical retail sales data.
    """
)

# -------------------------------------------------
# Load Data
# -------------------------------------------------
daily_df = pd.read_csv(
    "data/processed/daily_data.csv",
    parse_dates=["date"]
)

daily_features = pd.read_csv(
    "data/processed/daily_data_features.csv",
    parse_dates=["date"]
)

X_test = pd.read_csv("data/processed/X_test.csv")

# -------------------------------------------------
# Model Evaluation
# -------------------------------------------------
st.subheader("🤖 Model Evaluation Results")

eval_df = evaluate_models()

st.dataframe(
    eval_df.style.format({
        "RMSE": "{:.2f}",
        "MAE": "{:.2f}",
        "R2": "{:.3f}"
    }),
    use_container_width=True
)

# -------------------------------------------------
# Select Best Model
# -------------------------------------------------
best_row = eval_df.sort_values("RMSE").iloc[0]
best_model_name = best_row["Model"]

st.success(
    f"""
    🏆 **Best Performing Model: {best_model_name}**

    • RMSE: {best_row['RMSE']:.2f}  
    • MAE: {best_row['MAE']:.2f}  
    • R² Score: {best_row['R2']:.3f}
    """
)

# -------------------------------------------------
# Load Best Model Object
# -------------------------------------------------
MODEL_PATHS = {
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}

best_model = joblib.load(MODEL_PATHS[best_model_name])

# -------------------------------------------------
# 30-Day Forecasting
# -------------------------------------------------
st.subheader("🔮 30-Day Revenue Forecast")

forecast_df = forecast_next_30_days(
    model=best_model,
    last_known_data=daily_features.tail(1),
    feature_columns=X_test.columns.tolist()
)

# -------------------------------------------------
# Plot: Historical + Forecast (SAME GRAPH)
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    daily_df["date"],
    daily_df["daily_revenue"],
    label="Historical Revenue",
    color="blue",
    linewidth=2
)

ax.plot(
    forecast_df["date"],
    forecast_df["forecasted_revenue"],
    label="30-Day Forecast",
    color="orange",
    linestyle="--",
    linewidth=2
)

ax.set_title("Daily Revenue: Historical vs 30-Day Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Revenue")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
