import joblib
import pandas as pd

def load_model(model_name):
    paths = {
        "Random Forest": "models/random_forest.pkl",
        "XGBoost": "models/xgboost.pkl",
        "SARIMAX": "models/sarimax.pkl"
    }
    return joblib.load(paths[model_name])


def predict_ml(model, X):
    return model.predict(X)


def predict_sarimax(model, steps=14):
    return model.forecast(steps=steps)
