import joblib
import pandas as pd

def lambda_handler(event):
    model = joblib.load("model.pkl")

    data = pd.DataFrame(
        [[event["temp"], event["rain"]]],
        columns=["temperature", "rain"]
    )

    prediction = model.predict(data)

    return {"prediction": int(prediction[0])}


# Test locally
if __name__ == "__main__":
    event = {"temp": 30, "rain": 1}
    print("✅ Serverless Output:", lambda_handler(event))