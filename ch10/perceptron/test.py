from pyimagesearch.nn import neuralnetwork

import numpy as np
import matplotlib.pyplot as plt
import Layers

X = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
])
y = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

nn = neuralnetwork.NeuralNetwork([10, 10, 1], alpha=1)
nn.fit(X, y, epochs= 10000)

for (x, target) in zip(X, y):
    pred = nn.predict(x)[0][0]
    step = 1 if pred >1 else 0
    print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(x, target[0], pred, step))