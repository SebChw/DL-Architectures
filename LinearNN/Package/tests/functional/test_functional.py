import numpy as np
import pytest

import sys
sys.path.append(".")

from NumpyNN import functional
import sklearn.preprocessing as prep
import torch.nn.functional as func
import torch

@pytest.fixture
def data():
    return np.array([[1,2,3,4], [2,5,3,1], [-5, -2, -4,-1]], dtype=np.float32)


def test_normalization(data):
    assert np.allclose(prep.minmax_scale(data), functional.normalize(data))
    #all close uses two parameters rtol and atol
    #absolute(a - b) <= (atol + rtol * absolute(b)) if this is fulfilled then it is okay.
    #absolute(a - b) <= atol I consider two values close if they differ by at most atol
    #absolute(a - b) <= rtol * absolute(b) I consider two values close if they differ by at most rtol%

def test_softmax(data):
    data.dtype = np.float32
    assert np.allclose(func.softmax(torch.tensor(data), dim=1).numpy(), functional.softmax(data, axis=1))

def test_standarization(data):
    #assert (prep.scale(data, axis=0) == standardize(data)).all() this will return assertion error
    assert np.allclose(prep.scale(data, axis=0), functional.standardize(data)) #This works fine