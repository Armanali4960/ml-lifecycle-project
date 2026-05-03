from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# 👇 explicitly enable docs (safe)
app = FastAPI(docs_url="/docs", redoc_url="/redoc")

# Load trained model
model = joblib.load("model.pkl")

# Input schema
class InputData(BaseModel):
    features: list

# Home route
@app.get("/")
def home():
    return {"message": "ML API is running 🚀"}

# Prediction route
@app.post("/predict")
def predict(data: InputData):
    try:
        arr = np.array(data.features).reshape(1, -1)
        prediction = model.predict(arr)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}