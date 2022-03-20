import numpy as np

import sys
sys.path.append(".")

from NumpyNN.blocks.losses.CrossEntropyLoss import CrossEntropyLoss

def test_cross_entropy_loss():
    cel = CrossEntropyLoss()
    logits = np.array([[1,1,1,1], [1,1,1,1], [1,1,1,1]]) #since logits are equall within every learning example every class will have equal proability
    targets = np.array([1,3,2])

    loss = - np.log(0.25)
    assert np.allclose(cel.forward(logits, targets), loss)

