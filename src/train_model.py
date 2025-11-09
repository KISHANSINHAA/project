import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from src.features import create_all_features

def train_lightgbm(X, y, params=None):
    """Train LightGBM model"""
    try:
        import lightgbm as lgb
        dtrain = lgb.Dataset(X, label=y)
        params = params or {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1}
        model = lgb.train(params, dtrain, num_boost_round=200)
        return model
    except ImportError:
        print("LightGBM not installed, skipping...")
        return None

def train_adaboost(X, y):
    """Train AdaBoost model"""
    base_estimator = DecisionTreeRegressor(max_depth=5)
    model = AdaBoostRegressor(
        estimator=base_estimator,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)
    return model

def train_gradient_boosting(X, y):
    """Train Gradient Boosting model"""
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        loss='squared_error'
    )
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    if hasattr(model, 'predict'):
        preds = model.predict(X_test)
    else:
        # For LightGBM model
        preds = model.predict(X_test)
        
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    
    print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return rmse, mae

if __name__ == '__main__':
    print("🔄 Loading processed data...")
    df = pd.read_csv('data/processed/train.csv')
    print(f"📊 Loaded data with shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    print("⚙️ Creating features...")
    df = create_all_features(df)
    print(f"📊 Data shape after feature engineering: {df.shape}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"❓ Missing values:\n{missing_values[missing_values > 0]}")
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    print(f"📊 Data shape after dropping missing values: {df.shape}")
    
    if df.shape[0] == 0:
        print("❌ No data left after dropping missing values. Exiting.")
        exit(1)
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col not in ['sales', 'date', 'store', 'product', 'category']]
    print(f"📈 Number of features: {len(feature_columns)}")
    
    X = df[feature_columns]
    y = df['sales']
    
    print(f"🎯 Target: sales")
    print(f"📊 X shape: {X.shape}")
    print(f"📊 y shape: {y.shape}")
    
    # Check if we have enough data
    if X.shape[0] < 10:
        print("❌ Not enough data to train models. Need at least 10 samples.")
        exit(1)
    
    # Split data into train and test sets (75%/25%)
    print("✂️ Splitting data into train/test sets (75%/25%)...")
    # Reset index to ensure proper splitting
    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    
    split_index = int(len(X_reset) * 0.75)
    X_train = X_reset[:split_index]
    X_test = X_reset[split_index:]
    y_train = y_reset[:split_index]
    y_test = y_reset[split_index:]
    
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Check if we have enough data in each set
    if len(X_train) < 5 or len(X_test) < 5:
        print("❌ Not enough data in train or test sets. Need at least 5 samples in each.")
        exit(1)
    
    # Train models
    print("🚀 Training LightGBM model...")
    lgb_model = train_lightgbm(X_train, y_train)
    
    print("🚀 Training AdaBoost model...")
    ada_model = train_adaboost(X_train, y_train)
    
    print("🚀 Training Gradient Boosting model...")
    gb_model = train_gradient_boosting(X_train, y_train)
    
    # Evaluate models
    print("\n📊 Model Evaluation:")
    print("="*50)
    
    models = {}
    if lgb_model is not None:
        lgb_rmse, lgb_mae = evaluate_model(lgb_model, X_test, y_test, "LightGBM")
        models['LightGBM'] = (lgb_model, lgb_rmse)
    
    ada_rmse, ada_mae = evaluate_model(ada_model, X_test, y_test, "AdaBoost")
    models['AdaBoost'] = (ada_model, ada_rmse)
    
    gb_rmse, gb_mae = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")
    models['Gradient Boosting'] = (gb_model, gb_rmse)
    
    # Select best model based on RMSE
    best_model_name = min(models, key=lambda x: models[x][1])
    best_model = models[best_model_name][0]
    
    print(f"\n🏆 Best model: {best_model_name}")
    
    # Save the best model
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(best_model, f'artifacts/best_model.pkl')
    
    if lgb_model is not None:
        joblib.dump(lgb_model, 'artifacts/lightgbm_model.pkl')
    joblib.dump(ada_model, 'artifacts/adaboost_model.pkl')
    joblib.dump(gb_model, 'artifacts/gradient_boosting_model.pkl')
    
    print("✅ Models saved to artifacts/")
    
    # Save feature columns for prediction
    feature_info = {
        'feature_columns': feature_columns,
        'best_model': best_model_name
    }
    joblib.dump(feature_info, 'artifacts/feature_info.pkl')
    print("✅ Feature info saved to artifacts/feature_info.pkl")