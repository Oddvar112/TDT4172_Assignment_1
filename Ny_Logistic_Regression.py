import numpy as np


class LogisticRegression:


    def __init__(self, lr=0.5, epochs=3000, harmonics=4, degree=2):
        self.lr = lr
        self.epochs = epochs
        self.harmonics = int(harmonics)
        self.degree = int(degree)
        self.weights = None
        self.bias = 0.0

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _features_for_sample(self, x0, x1):
        theta = np.arctan2(x1, x0)
        return np.array([np.tan(theta)], dtype=float)



    def fit(self, X0, X1, y):
        X0 = np.asarray(X0)
        X1 = np.asarray(X1)
        y = np.asarray(y)
        n = len(X0)

        first_feats = self._features_for_sample(X0[0], X1[0])
        d = first_feats.size

        self.weights = np.zeros(d)
        self.bias = 0.0

        for i in range(self.epochs):
            feil_total = 0.0
            grad_w = np.zeros(d)

            for j in range(n):
                sirkel = self._features_for_sample(X0[j], X1[j])
                z = np.dot(sirkel, self.weights) + self.bias #en skalar
                pred = self.sigmoid(z)
                feil = pred - y[j]

                feil_total += feil
                grad_w += feil * sirkel

            self.weights -= self.lr * (1.0 / n) * grad_w  #bruker Logistisk loss
            self.bias -= self.lr * (1.0 / n) * feil_total

        return self

    def predict_proba(self, X0, X1):
        X0 = np.asarray(X0)
        X1 = np.asarray(X1)
        probs = []
        for j in range(len(X0)):
            feats = self._features_for_sample(X0[j], X1[j])
            z = np.dot(feats, self.weights) + self.bias
            probs.append(self.sigmoid(z))
        return np.array(probs)

    def predict(self, X0, X1):
        probs = self.predict_proba(X0, X1)
        return (probs >= 0.5).astype(int)