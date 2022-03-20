import numpy as np

import numpy as np

import sys
sys.path.append(".")

from NumpyNN.blocks import layers, losses


def get_loss(X_test, targets, blocks):
    X = np.copy(X_test)
    for block in blocks[:-1]:
        X = block.forward(X)
    
    return blocks[-1].forward(X, targets)

def test_backward():
    hidden_size = 10
    batch_size = 30
    num_of_arguments = 4
    num_of_classes = 5
    
    X = np.random.randn(batch_size, num_of_arguments)
    X.dtype = np.float64
    X_test = np.copy(X)
    targets = np.random.randint(0, num_of_classes, batch_size)

    layer = layers.LinearLayer(num_of_arguments, hidden_size)
    relu = layers.ReLU()
    layer2 = layers.LinearLayer(hidden_size, num_of_classes)
    cross_entropy = losses.CrossEntropyLoss()
    
    blocks = [layer, relu, layer2, cross_entropy]

    for block in blocks[:-1]:
        X = block.forward(X)

    loss = cross_entropy.forward(X, targets)

    #first_up_gradient = np.ones(cross_entropy.previous_probabilities.shape)
    cross_entropy.backward(None)

    for i in range(len(blocks)-1, 0 ,-1):
        blocks[i-1].backward(blocks[i].gradient)

    epsilon = 1e-7 # This can't be too big as then approximation will be too poor.

    # we will calculate derivative using two sided numerical derivative since it is more accurate d = ()
    for block in blocks[:-1]:
        #print(loss)
        if isinstance(block, layers.LinearLayer):
            W = block.W
            b = block.b
            approx_grad_W = np.zeros_like(W)
            approx_grad_b = np.zeros_like(b)
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    W_plus = np.copy(W)
                    W_plus[i][j] += epsilon
                    
                    block.W = W_plus
                    loss_plus = get_loss(X_test, targets, blocks)
                    #print(loss_plus)

                    W_minus = np.copy(W)
                    W_minus[i][j] -= epsilon
                    block.W = W_minus
                    loss_minus = get_loss(X_test, targets, blocks)
                    #print(loss_minus)
                    approx_grad_W[i][j] = (loss_plus - loss_minus)/ (2* epsilon)
                    #print(approx_grad_W[i][j])
                    #print(block.gradient_W[i][j] / batch_size)
                    
    
            assert np.allclose(approx_grad_W, block.gradient_W / batch_size )

            for i in range(b.shape[0]):
                b_plus = np.copy(b)
                b_plus[i] += epsilon

                block.b = b_plus
                loss_plus = get_loss(X_test, targets, blocks)

                b_minus = np.copy(b)
                b_minus[i] -= epsilon

                block.b = b_minus
                loss_minus = get_loss(X_test, targets, blocks)
        
                approx_grad_b[i] = (loss_plus - loss_minus)/ (2* epsilon)

            assert np.allclose(approx_grad_b, block.gradient_B / batch_size )


        
    


    
    