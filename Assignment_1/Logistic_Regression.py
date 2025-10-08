import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self):
        self.vekt0 = None    # Vekt for x0
        self.vekt1 = None    # Vekt for x1  
        self.bias = None     # Bias term
        self.lr = 0.1        
        self.epochs = 10000

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X0, X1, y):
        # Start med null-gjetning
        self.vekt0 = 0.0
        self.vekt1 = 0.0
        self.bias = 0.0
        
        # Samme hovedløkke som før
        for epoch in range(self.epochs):
            feil_total = 0
            vekt0_gradient = 0
            vekt1_gradient = 0
            
            # Gå gjennom hvert datapunkt (samme som før)
            for j in range(len(X0)):
                z = X0[j] * self.vekt0 + X1[j] * self.vekt1 + self.bias
                prediksjon = self.sigmoid(z)
                feil = prediksjon - y[j]
                feil_total += feil
                vekt0_gradient += feil * X0[j]
                vekt1_gradient += feil * X1[j]
            
            # STEG 5: Oppdater parametere (samme som før)
            self.vekt0 -= self.lr * (1/len(X0)) * vekt0_gradient
            self.vekt1 -= self.lr * (1/len(X0)) * vekt1_gradient  
            self.bias -= self.lr * (1/len(X0)) * feil_total


    def predict_proba(self, X0, X1):
        # Returner sannsynligheter (0 til 1)
        probabilities = []
        for j in range(len(X0)):
            z = X0[j] * self.vekt0 + X1[j] * self.vekt1 + self.bias
            prob = self.sigmoid(z)
            probabilities.append(prob)
        return probabilities

    def predict(self, X0, X1):
        #Returner binære prediksjoner (0 eller 1)
        probabilities = self.predict_proba(X0, X1)
        return [1 if prob >= 0.5 else 0 for prob in probabilities]
