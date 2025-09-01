import numpy as np

class LinearRegression():
    
    def __init__(self):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.vekt = None    # Ax +c dette er A  
        self.bias = None    # konstanten 
        self.lr = 0.001      # læringsrate
        self.epochs = 1000  # antall iterasjoner

        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # TODO: Implement
        self.vekt = 0.0
        self.bias = 0.0

        for _ in range(self.epochs): #dette kan gjøres veldig raskt med matise men jeg finner det vanskelig å forstår hva som skjer da så begynner med dette og går over til matrise etter hvert 
            feil_total = 0
            vekt_gradient = 0

            for j in range(len(X)):
                prediksjon = X[j] * self.vekt + self.bias
                feil = prediksjon - y[j]
                feil_total += feil
                vekt_gradient += feil * X[j]

            self.vekt -= self.lr * (2/len(X)) * vekt_gradient
            self.bias -= self.lr * (2/len(X)) * feil_total

            
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        prediksjoner = []
        for j in range(len(X)):
            prediksjon = X[j] * self.vekt + self.bias
            prediksjoner.append(prediksjon)
        return prediksjoner




