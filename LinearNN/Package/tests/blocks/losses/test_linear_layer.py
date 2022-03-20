import numpy as np

import sys
sys.path.append(".")

from NumpyNN.blocks.layers.LinearLayer import LinearLayer


def test_forward_one_layer():
    hidden_size = 10
    batch_size = 30
    previous_hidden_size = 4

    layer = LinearLayer(previous_hidden_size, hidden_size)
    layer.W = np.ones((previous_hidden_size, hidden_size)) # We have 4 attributes at input and we generate 10 at output
    layer.b = np.ones(hidden_size)

    X = np.ones((batch_size,previous_hidden_size)) #we have 30 elements in the batch

    output = layer.forward(X)

    assert output.shape == (batch_size, hidden_size) 
    assert (output == previous_hidden_size + 1).all() #each hidden unit creates a linear combination of the values from previous unit
    #since all inputs are 1 and all weights are 1 and bias are one at the end in each unit we have calculation 1*1 + 1*1 + 1*1 + 1*1 + 1