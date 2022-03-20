from .Block import Block

import warnings

import numpy as np

class Network:
    def __init__(self, loss_func : Block, *args):
        """
        Args:
            loss_func (Block): Loss function used
            *args: all the blocks that consist the network
        """
        self.blocks = []
        self.loss_func = loss_func
        
        for a in args:
            if isinstance(a, Block):
                self.blocks.append(a)
            else:
                warnings.warn(f"in *args you should pass an Block instances. If you created your own block make it inherit from Block! By now {a} is neglected")

        self.backward_blocks = self.blocks + [loss_func]

    def forward(self, X : np.array) -> np.array:
        """Use this function during inference to get output logits of the network

        Args:
            X (np.array): Batch of size (N, A). where N - size of the batch and A - number of attributes

        Returns:
            np.array: logits
        """
        for block in self.blocks:
            X = block.forward(X)

        return X
    
    def forward_loss(self, X : np.array, targets: np.array) -> float:
        """Use this function during training to get loss and be able to backpropagate everything

        Args:
            X (np.array): Batch of size (N, A). where N - size of the batch and A - number of attributes
            targets (np.array): vector of size (N,). where N - size of the batch, and each value in vector is target

        Returns:
           float: loss value
        """
        X = self.forward(X)

        return self.loss_func.forward(X, targets)

    def backward(self,):
        """Function performing backpropagation for the entire network
        """
        #first_up_gradient = np.ones(self.loss_func.previous_probabilities.shape) # This is dummy matrix of ones just so 
        self.loss_func.backward(None)

        for i in range(len(self.backward_blocks)-1, 0 ,-1):
            self.backward_blocks[i-1].backward(self.backward_blocks[i].gradient)

