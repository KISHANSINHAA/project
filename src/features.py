import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_lag_features(df, lags=[1, 3, 7, 14, 28], col='sales'):
    """Create lag features for time series forecasting"""
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df.groupby(['store', 'product'])[col].shift(lag)
    return df

def create_rolling_features(df, windows=[3, 7, 14, 28], col='sales'):
    """Create rolling window features"""
    for w in windows:
        df[f'{col}_roll_mean_{w}'] = df.groupby(['store', 'product'])[col].shift(1).rolling(window=w, min_periods=1).mean()
        df[f'{col}_roll_std_{w}'] = df.groupby(['store', 'product'])[col].shift(1).rolling(window=w, min_periods=1).std()
        df[f'{col}_roll_min_{w}'] = df.groupby(['store', 'product'])[col].shift(1).rolling(window=w, min_periods=1).min()
        df[f'{col}_roll_max_{w}'] = df.groupby(['store', 'product'])[col].shift(1).rolling(window=w, min_periods=1).max()
    return df

def create_expanding_features(df, col='sales'):
    """Create expanding window features"""
    df[f'{col}_expanding_mean'] = df.groupby(['store', 'product'])[col].expanding(min_periods=1).mean().reset_index(level=[0,1], drop=True)
    df[f'{col}_expanding_std'] = df.groupby(['store', 'product'])[col].expanding(min_periods=1).std().reset_index(level=[0,1], drop=True)
    return df

def create_date_features(df, date_col='date'):
    """Extract date-related features"""
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['week_of_year'] = df[date_col].dt.isocalendar().week.astype(int)
    df['quarter'] = df[date_col].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = (df[date_col].dt.is_month_start).astype(int)
    df['is_month_end'] = (df[date_col].dt.is_month_end).astype(int)
    
    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def create_price_features(df):
    """Create price-related features"""
    # Price change indicators
    df['price_change'] = df.groupby(['store', 'product'])['price'].diff()
    df['price_change_pct'] = df.groupby(['store', 'product'])['price'].pct_change()
    
    # Price relative to category average
    category_avg_price = df.groupby(['category', 'date'])['price'].transform('mean')
    df['price_vs_category_avg'] = df['price'] / category_avg_price
    
    return df

def encode_categorical_features(df):
    """Encode categorical features"""
    le_store = LabelEncoder()
    le_product = LabelEncoder()
    le_category = LabelEncoder()
    
    df['store_encoded'] = le_store.fit_transform(df['store'])
    df['product_encoded'] = le_product.fit_transform(df['product'])
    df['category_encoded'] = le_category.fit_transform(df['category'])
    
    return df

def create_all_features(df):
    """Create all features for the model"""
    print("📅 Creating date features...")
    df = create_date_features(df)
    
    print("🏷️ Encoding categorical features...")
    df = encode_categorical_features(df)
    
    print("🔢 Creating lag features...")
    df = create_lag_features(df)
    
    print("📈 Creating rolling features...")
    df = create_rolling_features(df)
    
    print("📊 Creating expanding features...")
    df = create_expanding_features(df)
    
    print("💰 Creating price features...")
    df = create_price_features(df)
    
    return df