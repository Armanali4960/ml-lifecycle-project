import joblib
from train_model import model

# Step 1: Training already done

# Step 2: Save model
joblib.dump(model, "model.pkl")

# Step 3: Load model
loaded_model = joblib.load("model.pkl")

# Step 4: Predict
import pandas as pd

sample = pd.DataFrame([[30, 1]], columns=["temperature", "rain"])
result = loaded_model.predict(sample)

print("✅ Pipeline result:", result)