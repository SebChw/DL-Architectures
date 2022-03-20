import numpy as np
from ...functional import softmax

class CrossEntropyLoss():
    def __init__(self,):
        self.gradient = None
        self.previous_probabilities = None
        self.previous_targets = None

    def forward(self, prediction_logits : np.array, targets : np.array):
        """this function does:
           logits -> softmax -> negative_cross_entropy

        Args:
            prediction_logits (np.array): output of the network
            targets (np.array): vector with correct classes

        Returns:
            _type_: loss value
        """
        self.previous_targets = targets

        prediction_probabilities = softmax(prediction_logits)
        self.previous_probabilities = prediction_probabilities
        
        target_probabilities = prediction_probabilities[np.arange(targets.shape[0]), targets] # take every row and appropriate class probability from it
        loss_separately = -np.log(target_probabilities)

        loss_averaged = loss_separately.mean()

        if self.gradient is None or self.gradient.shape != self.previous_probabilities.shape:
            self.gradient = np.zeros_like(self.previous_probabilities)

        return loss_averaged
        
    def backward(self, up_gradient : np.array ):
        """Calculates derivative which is in that case quite trivial:
            
            it's sj - 1. When sj is probability of correct class. Rest of classes stays the same

        Args:
            up_gradient (np.array): _description_
        """
        ones_ = np.zeros(self.previous_probabilities.shape)
        ones_[np.arange(self.previous_probabilities.shape[0]), self.previous_targets] = 1

        if up_gradient is not None:
            self.gradient += (self.previous_probabilities - ones_) @ up_gradient
        else:
            self.gradient += (self.previous_probabilities - ones_)

    def zero_grad(self,):
        self.gradient.fill(0)
