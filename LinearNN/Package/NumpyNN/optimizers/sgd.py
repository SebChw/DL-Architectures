from .Optimizer import Optimizer
from ..blocks.Network import Network

class SGD(Optimizer):
    def __init__(self, network : Network, learning_rate):
        super(SGD, self).__init__(network, learning_rate)

    def one_step_update(self, block):
        """Funtion that updates gradients by - gradient * learning_rate

        Args:
            block (_type_): _description_
        """
        #! Sometimes I saw that gradient should be accumulated using sum sometimes that it should be averaged
        #! basically What it changes is only the good learning rate
        block.W -= block.gradient_W * self.learning_rate #/ batch_size
        block.b -= block.gradient_B * self.learning_rate #/ batch_size
