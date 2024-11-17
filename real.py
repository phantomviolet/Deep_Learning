import numpy as np

def sigomoid(x):
    return 1 / (1 + np.exp(-x))

def forward_propagation(X, network):
    W1, W2 = network["W1"], network["W2"]
    A1 = np.dot(X, W1)
    Z1 = sigomoid(A1)
    A2 = np.dot(Z1, W2)
    Z2 = sigomoid(A2)

    return Z2

def init_network():
    network = {}
    network["W1"] = np.array([[0.5, 0.5 ,0.5], [0.5, 0.5, 0.5]])
    network["W2"] = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    
    return network

network = init_network()
X = np.array([1.0, 0.5])
Z2 = forward_propagation(X, network)

print(Z2)
