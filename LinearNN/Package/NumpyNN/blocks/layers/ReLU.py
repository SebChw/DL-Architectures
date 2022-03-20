import copy
from ..Block import Block
import numpy as np
class ReLU(Block):
    def __init__(self,):
        self.cache = None
        self.gradient = None
    def forward(self, X : np.array):
        #What is important here is the fact that, we must copy the X array.
        #We can't do any inplace operation as we may mess up everything
        X_copied = copy.deepcopy(X)
        X_copied[X_copied < 0] = 0

        self.cache = X # In backprop gradient is passed only to elemeents bigger than 0

        if self.gradient is None or self.gradient.shape != X.shape:
            self.gradient= np.zeros_like(X)

        return X_copied

    def backward(self, up_gradient : np.array):
        #Here we need elementwise multiplication, gradient flow only through cells which were greater than 0
        
        self.gradient += (self.cache > 0) * up_gradient

    def zero_grad(self,):
        self.gradient.fill(0)
