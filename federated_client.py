import flwr as fl
import numpy as np

class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [np.array([1.0])]

    def fit(self, parameters, config):
        return parameters, 1, {}

    def evaluate(self, parameters, config):
        return 0.5, 1, {}

fl.client.start_numpy_client(server_address="localhost:8080", client=Client())