# 📈 Retail Sales Forecasting (Machine Learning Project)

A production-grade retail sales forecasting system built using Python and Machine Learning.
The project demonstrates an end-to-end ML workflow — from raw data ingestion and feature engineering to model evaluation and an interactive Streamlit dashboard.

## 🚀 Project Overview

Retail sales data is highly volatile and influenced by temporal patterns.
This project focuses on predicting daily revenue using historical transaction data and advanced feature engineering, achieving high predictive accuracy.

## 🔑 Key Highlights

- Leakage-free ML pipeline
- Strong feature engineering (lags, rolling stats, momentum)
- Multiple ML models compared
- Interactive Streamlit UI
- Clean, modular, production-style codebase

## 🧠 Models Used

| Model | Purpose |
|-------|---------|
| Random Forest Regressor | Baseline ensemble model |
| XGBoost Regressor | Final high-performance model |

Feature engineering contributed more to performance improvement than adding new models.

## 📊 Final Model Performance

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Random Forest | 279.53 | 175.40 | 0.909 |
| XGBoost (Best) | 240.12 | 153.10 | 0.933 |

✅ XGBoost explains 93.3% of daily revenue variance

## 🏗️ Project Architecture

```
retail-sale-forcasting/
│
├── app/
│   └── streamlit_app.py          # Streamlit dashboard
│
├── src/
│   ├── data_loader.py            # Load & aggregate raw data
│   ├── feature_engineering.py    # Lag, rolling, momentum features
│   ├── train_ml_models.py        # Model training
│   ├── evaluate_models.py        # Model evaluation
│   └── metrics.py                # RMSE, MAE, R²
│
├── data/
│   ├── raw/
│   │   └── retail_sales_dataset.csv
│   └── processed/
│       ├── daily_data.csv
│       ├── daily_data_features.csv
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── models/
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── requirements.txt
└── README.md
```

## 🔧 Feature Engineering Strategy

To capture real retail behavior, the following features were engineered:

✅ **Memory (Lags)**
- Revenue lags: 1, 7, 14, 30 days
- Transaction & quantity lags

✅ **Trend & Momentum**
- Revenue difference (1-day, 7-day)
- Percentage change (7-day)

✅ **Volatility**
- Rolling standard deviation (7, 14, 30 days)

✅ **Smoothed Baselines**
- Rolling mean revenue (7, 14, 30 days)

All features use past values only → no data leakage.

## 🖥️ Streamlit Dashboard

The Streamlit UI provides:

- 📈 Historical daily revenue visualization
- 📊 Model comparison table
- 🏆 Automatic best model selection
- 💼 Business-friendly interpretation of results

### ▶️ Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone <repo-url>
cd retail-sale-forcasting
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run pipeline (optional)

```bash
python src/data_loader.py
python src/feature_engineering.py
python src/train_ml_models.py
python src/evaluate_models.py
```

## 💼 Business Interpretation

- The model predicts daily revenue with an average error of ~150 units
- High accuracy achieved through temporal feature engineering
- Suitable for:
  - Sales planning
  - Inventory optimization
  - Revenue trend analysis

## 🎤 Interview-Ready Summary

"I built a leakage-free retail revenue forecasting pipeline using Random Forest and XGBoost. By focusing on advanced feature engineering — including long-term lags, rolling statistics, and momentum — the model achieved an R² of 0.93. I deployed results using a Streamlit dashboard for business-friendly interpretation."

## 🧰 Tech Stack

- Language: Python
- Libraries: Pandas, NumPy, Scikit-learn, XGBoost
- Visualization: Streamlit
- ML Techniques: Feature Engineering, Ensemble Learning, Time-aware Validation