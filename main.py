from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Initialize FastAPI app
app = FastAPI()

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
        # Convert input to numpy array
        arr = np.array(data.features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(arr)

        # Return result
        return {"prediction": prediction.tolist()}

    except Exception as e:
        return {"error": str(e)}