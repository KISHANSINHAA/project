import joblib
import pandas as pd
import numpy as np
from src.features import create_all_features

# Load models and feature info
try:
    model = joblib.load('artifacts/best_model.pkl')
    feature_info = joblib.load('artifacts/feature_info.pkl')
    feature_columns = feature_info['feature_columns']
except FileNotFoundError:
    print("Model or feature info not found. Please train the model first.")
    model = None
    feature_columns = []

def predict(df):
    """Make predictions on the input dataframe"""
    if model is None:
        raise ValueError("Model not loaded. Please train the model first.")
    
    # Apply the same feature engineering as in training
    df = create_all_features(df)
    
    # Select only the features used in training
    X = df[feature_columns]
    
    # Handle missing values
    X = X.fillna(0)
    
    # Make predictions
    if hasattr(model, 'predict'):
        preds = model.predict(X)
    else:
        # For LightGBM model
        preds = model.predict(X)
    
    return preds

def forecast_next_30_days(store_data):
    """Forecast sales for the next 30 days"""
    if model is None:
        raise ValueError("Model not loaded. Please train the model first.")
    
    # Get the last date in the dataset
    last_date = pd.to_datetime(store_data['date'].max())
    
    # Create future dates
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
    
    # Create a copy of the last known data for each product
    last_data = store_data.groupby('product').tail(1).reset_index(drop=True)
    
    forecasts = []
    for date in future_dates:
        # Create forecast data for this date
        forecast_data = last_data.copy()
        forecast_data['date'] = date
        
        # Make predictions
        predictions = predict(forecast_data)
        
        # Add predictions to results
        forecast_data['predicted_sales'] = predictions
        forecasts.append(forecast_data)
    
    # Combine all forecasts
    forecast_df = pd.concat(forecasts, ignore_index=True)
    return forecast_df

def identify_stock_needs(forecast_df, threshold=0.8):
    """Identify which products need to be bought based on forecast"""
    # Calculate average predicted sales per product
    avg_sales = forecast_df.groupby('product')['predicted_sales'].mean().reset_index()
    
    # Sort by average sales (descending)
    avg_sales = avg_sales.sort_values('predicted_sales', ascending=False)
    
    # Identify products that need to be bought (high demand)
    needs_buying = avg_sales[avg_sales['predicted_sales'] > threshold * avg_sales['predicted_sales'].max()]
    
    # Identify products that don't need to be bought (low demand)
    dont_buy = avg_sales[avg_sales['predicted_sales'] <= threshold * avg_sales['predicted_sales'].max()]
    
    return needs_buying, dont_buy