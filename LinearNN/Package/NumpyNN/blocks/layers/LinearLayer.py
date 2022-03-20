import numpy as np
from ..Block import Block


class LinearLayer(Block):
    def __init__(self, previous_layer_hidden_units, hidden_units, dtype=np.float64):
        self.hidden_units = hidden_units
        self.previous_layer_hidden_units = previous_layer_hidden_units
        self.dtype = dtype

        #initialize weights
        self.W = np.random.randn(
            self.previous_layer_hidden_units, self.hidden_units)
        self.b = np.random.randn(self.hidden_units)
        
        #initialize gradients
        self.gradient = None # this must be initialized based on the batch size
        self.gradient_B = np.zeros_like(self.b)
        self.gradient_W = np.zeros_like(self.W)

        self.cacheX = None # to calculate backward pass some things must be remembered 

    def forward(self, X : np.array):

        X_flattened = X.reshape(X.shape[0], -1)
        
        output = X_flattened@self.W + self.b
        
        self.cacheX = X
        if self.gradient is None or self.gradient.shape != X.shape:
            self.gradient = np.zeros_like(X)

        return output

    def backward(self, up_gradient : np.array):
        
        # (batch_size, hidden_unit) @ (prev_hidden_unit, hidden_unit).T  = (batch_size, prev_hidden_unit)
        self.gradient += up_gradient @ self.W.T
        # (batch_size, prev_hidden_unit).T @ (batch_size, hidden_unit)  = (prev_hidd_unit, hidden_unit)
        self.gradient_W += self.cacheX.T @ up_gradient
        # we need shape of 10 so to have it we must sum w.r.t axis = 0 so that all rows collapse
        self.gradient_B += up_gradient.sum(axis=0)

    def zero_grad(self,):
        self.gradient.fill(0)
        self.gradient_W.fill(0)
        self.gradient_B.fill(0)

