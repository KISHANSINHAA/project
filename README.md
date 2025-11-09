# Retail Demand Forecasting System

A comprehensive retail forecasting solution with advanced ML algorithms, interactive dashboard, and API endpoints.

## 🛍️ Retail Demand Forecasting Dashboard

## 📋 Project Overview
This project is a retail demand forecasting solution designed for CDAC students to learn advanced machine learning concepts. The application provides sales predictions, stock recommendations, and interactive visualizations to help retail businesses optimize their inventory management.

## 🎯 Key Features
- **30-Day Sales Forecasting**: Predict future sales for individual products or all products
- **Stock Purchase Recommendations**: Get actionable insights on what products to buy and avoid
- **Interactive Dashboard**: User-friendly interface with dropdown selections and visualizations
- **Multiple ML Algorithms**: Uses AdaBoost and Gradient Boosting for accurate predictions
- **Team Learning Focus**: Designed for teams of 3 members to enhance CV/CDAC projects

## 🚀 Quick Start
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python start_app.py
   ```

3. **Access the Dashboard**:
   Open your browser and go to `http://localhost:8501`

## 📁 Project Structure
```
retail-forecasting/
├── app/
│   ├── dashboard.py          # Streamlit dashboard interface
│   └── api.py               # API endpoints (if needed)
├── src/
│   ├── data_preprocessing.py # Data preparation and cleaning
│   ├── features.py          # Feature engineering
│   ├── train_model.py       # Model training logic
│   └── predict.py           # Prediction functions
├── data/
│   ├── processed/           # Cleaned and processed data
│   └── sample_sales_data.csv # Sample dataset
├── artifacts/               # Trained models and feature information
├── requirements.txt         # Python dependencies
├── simple_train.py          # Model training script
├── start_app.py             # Application startup script
└── README.md                # This file
```

## 🧠 Machine Learning Models
The project implements multiple advanced ML algorithms:
- **AdaBoost**: Adaptive Boosting for ensemble learning
- **Gradient Boosting**: Sequential decision tree building
- **Model Selection**: Automatic selection of best performing model based on RMSE

## 📊 Dashboard Features
1. **30-Day Forecast**:
   - Select store and product from dropdown menus
   - View individual product trends or all products together
   - Download forecast data as CSV

2. **Stock Recommendations**:
   - Automatically generated "buy" and "avoid" lists
   - Recommended quantities for high-demand products
   - Clear reasoning for each recommendation

3. **Model Insights**:
   - Performance metrics for different algorithms
   - Feature importance visualization

## 🛠️ Development Setup
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Run data preprocessing: `python src/data_preprocessing.py`
4. Train models: `python simple_train.py`
5. Start dashboard: `python start_app.py`

## 📈 Dataset Information
- Uses a smaller sample dataset (~7,240 rows) instead of large 7.6GB files
- 75% train / 25% test split for model evaluation
- Synthetic retail data with trend, seasonality, and randomness

## 👥 Team Learning Benefits
This project helps CDAC teams of 3 members to:
- Learn advanced ML algorithms (AdaBoost, Gradient Boosting)
- Understand data preprocessing and feature engineering
- Gain experience with Streamlit dashboard development
- Practice model evaluation and selection techniques
- Enhance CV with a complete ML pipeline project

## 📞 Support
For issues and questions, please contact the project maintainers.
