import ray
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Start Ray
ray.init()

@ray.remote
def train_model(data):
    X, y = data
    model = RandomForestClassifier()
    model.fit(X, y)
    return "Model trained"

# Create dummy dataset
X = np.random.rand(100, 2)
y = np.random.randint(0, 3, 100)

# Split data into 2 parts
data_splits = [(X[:50], y[:50]), (X[50:], y[50:])]

# Run training in parallel
results = ray.get([train_model.remote(d) for d in data_splits])

print(results)