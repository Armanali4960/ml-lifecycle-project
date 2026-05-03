@app.post("/predict")
def predict(data: InputData):
    try:
        arr = np.array(data.features).reshape(1, -1)
        pred = model.predict(arr)[0]

        if pred == 0:
            result = "Low Traffic"
        elif pred == 1:
            result = "Medium Traffic"
        else:
            result = "High Traffic"

        return {
            "prediction_value": int(pred),
            "traffic_level": result
        }

    except Exception as e:
        return {"error": str(e)}