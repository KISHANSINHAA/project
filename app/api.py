from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.predict import predict

app = FastAPI(title="Retail Demand Forecasting API")

class Payload(BaseModel):
    data: list

@app.post('/predict')
def predict_endpoint(payload: Payload):
    df = pd.DataFrame(payload.data)
    predictions = predict(df)
    return {'predictions': predictions.tolist()}
