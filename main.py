from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# ✅ MUST come first
app = FastAPI()

# ✅ Load model
model = joblib.load("model.pkl")

# ✅ Input schema
class InputData(BaseModel):
    features: list

# ✅ Home route
@app.get("/")
def home():
    return {"message": "ML API is running 🚀"}

# ✅ Predict route
@app.post("/predict")
def predict(data: InputData):
    try:
        arr = np.array(data.features).reshape(1, -1)
        prediction = model.predict(arr)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}