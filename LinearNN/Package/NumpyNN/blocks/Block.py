"""Base class for any Layer or Loss function. All classes that can be considered
    as block in a network (layers, activations, loss functions) should implement these three methods
"""
class Block:
    def forward():
        """Function used during forward pass phase.

        Args:
            X (np.array): numpy array with values given by the layer before

            [targets] (np.array): in case of Loss function
        """
        pass

    def backward():
        """Function used during backpropagation phase

        Args:
            up_gradient (np.array): upstream gradient from the layer after.
        """
        pass

    def zero_grad():
        """This function should zero out all gradients, so that they do not accumulate over iterations
        """
        pass