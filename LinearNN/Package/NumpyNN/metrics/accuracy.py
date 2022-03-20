import numpy as np

from ..blocks.Network import Network
from ..functional import softmax

def accuracy_multiclass(network : Network, data : np.array, targets: np.array) -> float:
    """Function calculating accuracy on the multiclass classification problem

    Args:
        network (Network): Network from which we get logits
        data (np.array): dataset on which accuracy should be calculated
        targets (np.array): targets with appropriate class labels

    Returns:
        float: accuracy in form in interval [0:1]
    """
    logits = network.forward(data)
    
    probabilities = softmax(logits)
    predictions = np.argmax(probabilities, axis=1)

    scores = (predictions == targets)

    return np.sum(scores) / len(scores)