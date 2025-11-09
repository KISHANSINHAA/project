import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from src.features import create_all_features

print("=== Simple Training Script ===")

try:
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/processed/train.csv')
    print(f"Data loaded: {df.shape}")
    
    # Create features
    print("Creating features...")
    df = create_all_features(df)
    print(f"After feature engineering: {df.shape}")
    
    # Clean data
    df = df.dropna()
    print(f"After cleaning: {df.shape}")
    
    # Prepare for training
    feature_columns = [col for col in df.columns if col not in ['sales', 'date', 'store', 'product', 'category']]
    X = df[feature_columns]
    y = df['sales']
    
    print(f"Features: {X.shape}")
    print(f"Target: {y.shape}")
    
    # Split data
    split_idx = int(len(X) * 0.75)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train models
    print("Training AdaBoost...")
    ada_model = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=3),
        n_estimators=50,
        random_state=42
    )
    ada_model.fit(X_train, y_train)
    
    print("Training Gradient Boosting...")
    gb_model = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=3,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    
    # Save models
    print("Saving models...")
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(ada_model, 'artifacts/adaboost_model.pkl')
    joblib.dump(gb_model, 'artifacts/gradient_boosting_model.pkl')
    
    # Save the AdaBoost model as best model
    joblib.dump(ada_model, 'artifacts/best_model.pkl')
    
    # Save feature info
    feature_info = {
        'feature_columns': feature_columns,
        'best_model': 'AdaBoost'
    }
    joblib.dump(feature_info, 'artifacts/feature_info.pkl')
    
    print("✅ Training completed successfully!")
    print("Models saved to artifacts/ directory")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()