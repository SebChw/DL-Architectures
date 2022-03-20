from ..blocks.Network import Network
from ..blocks.layers import LinearLayer

class Optimizer():
    """Base class for every optimizer. All optimizers have some things in common. Class based on this one should implement only one_step_update
    """
    def __init__(self, network : Network, learning_rate):
        self.network = network
        self.learning_rate = learning_rate

    def update_weights(self,):
        for block in self.network.blocks:
            if isinstance(block, LinearLayer):
                self.one_step_update(block)

    def zero_grad(self,):
        self.network.loss_func.zero_grad()
        
        for b in self.network.blocks:
            b.zero_grad()

    def one_step_update(self, block):
        pass