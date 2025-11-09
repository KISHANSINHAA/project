# Retail Demand Forecasting System

A comprehensive retail forecasting solution with advanced ML algorithms, interactive dashboard, and API endpoints.

## 🎯 Project Overview

This project forecasts retail demand for the next 30 days, identifies which products to stock, and provides actionable insights through an interactive dashboard. It uses advanced machine learning algorithms including AdaBoost and Gradient Boosting to provide accurate predictions.

## 🚀 Features

- **Advanced ML Models**: AdaBoost, Gradient Boosting
- **Interactive Dashboard**: Streamlit-based UI with 3D visualizations
- **API Endpoints**: FastAPI for programmatic access
- **Data Pipeline**: Complete preprocessing and training workflow
- **Stock Recommendations**: Identifies products to buy and avoid with quantities
- **30-Day Forecasting**: Predicts sales for the next month

## 📁 Project Structure

```
retail-forecasting/
├── app/
│   ├── api.py          # FastAPI endpoints
│   └── dashboard.py    # Streamlit dashboard
├── src/
│   ├── data_preprocessing.py  # Data preparation
│   ├── features.py     # Feature engineering
│   ├── train_model.py  # Model training
│   └── predict.py      # Prediction functions
├── data/
│   ├── processed/      # Processed data
│   └── sample_sales_data.csv  # Sample dataset
├── artifacts/          # Trained models
├── tests/              # Unit tests
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration
├── Jenkinsfile         # Jenkins pipeline
├── projectfinalpipeline.groovy  # Specific Jenkins pipeline
└── README.md
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/KISHANSINHAA/project.git
cd retail-forecasting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## ▶️ Usage

### Run the Complete Pipeline

```bash
# Run preprocessing and training
python run_complete_project.py
```

### Manual Execution

1. **Data Preprocessing**:
```bash
python src/data_preprocessing.py
```

2. **Model Training**:
```bash
python simple_train.py
```

3. **Start Dashboard**:
```bash
streamlit run app/dashboard.py
```

4. **Start API Server**:
```bash
uvicorn app.api:app --reload
```

## 📊 Dashboard Features

The Streamlit dashboard provides:

- **Data Upload & Prediction**: Upload CSV data for sales predictions
- **30-Day Forecast**: Generate sales forecasts for the next month
- **Stock Recommendations**: Identify which products to buy and avoid with recommended quantities
- **Model Insights**: Feature importance and model performance
- **3D Visualizations**: Interactive 3D charts for data analysis

## 🔌 API Endpoints

The FastAPI server provides:

- `POST /predict`: Make sales predictions
- Interactive API documentation at `/docs`

## 🧠 Machine Learning Models

This project implements multiple advanced ML algorithms:

- **AdaBoost**: Adaptive boosting algorithm
- **Gradient Boosting**: Traditional gradient boosting regressor

The system automatically selects the best performing model based on RMSE evaluation.

## 📈 Sample Dataset

The project includes a sample retail dataset with:
- Date information
- Store identifiers
- Product categories
- Sales quantities
- Pricing information

## 👥 Team Project

This project is designed for a team of 3 members to learn:
- Advanced ML concepts
- Data preprocessing techniques
- Feature engineering
- Model evaluation
- Dashboard development
- API development

Perfect for CDAC projects and CV building!

## 🐳 Docker Deployment

Build and run the Docker container:

```bash
docker build -t retail-forecasting-app .
docker run -p 8501:8501 retail-forecasting-app
```

## 🔄 Jenkins Pipeline

The project includes Jenkins pipeline configurations for CI/CD:
- `Jenkinsfile`: Main pipeline configuration
- `projectfinalpipeline.groovy`: Specific pipeline for the project

## 📝 Requirements

- Python 3.8+
- See `requirements.txt` for complete dependencies

## 🏁 Getting Started

1. Run the preprocessing script to generate sample data
2. Train the models using the training script
3. Launch the dashboard to interact with predictions
4. Use the API for programmatic access

## 🤝 Contributing

This project is designed as a learning tool for teams. Feel free to extend and modify it for your specific needs.