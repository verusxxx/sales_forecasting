from fastapi import FastAPI, Request
import pandas as pd
from src.model import load_model
from src.forecast import forecast
from src.preprocessing import clean_data, feature_engineering

app = FastAPI()
model = load_model()

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    df = pd.DataFrame(data)
    df = clean_data(df)
    df = feature_engineering(df)
    preds = forecast(model, df)
    return {"predictions": preds.tolist()}