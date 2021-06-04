import numpy as np

class Perceptron: 
    def __init__(self, N, alpha=0.1):
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        slef.history = {}
        self.history["loss"] = []
        X = np.c_[X, np.ones((X.shape[0]))]
 
        for epoch in np.arange(0, epochs):
            errors = 0.0
            trials = 0

            for (x, target) in zip(X, y):
                p = self.step(np.dot(x, self.W))

                if p != target:
                    error = p - target
                    errors =  p- target
                    trials += abs(error[0])
                    self.W += -self.alpha * error * x
            self.history["loss"].append(errors/trials)
            print("Epoch {}/{} - loss: {}, W: {}".format(epoch, epochs, erros/trials, self.W))
                    

    def predict(self, X, addBias=True):
        X = np.atleast_2d(X)
        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]
            return self.step(np.dot(X, self.W))
