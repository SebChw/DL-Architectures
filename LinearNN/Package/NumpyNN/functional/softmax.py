import numpy as np


def softmax(data: np.array, axis=1) -> np.array:
    """Function performing softmax

    Args:
        data (np.array): _description_
        axis (int, optional): _description_. Defaults to 1.

    Returns:
        np.array: _description_
    """
    #This is a little bit tricky, we need keepdims = True
    #and also this operation must be done rowwise!.
    #since one rows is one learning example
    denominator = np.exp(data).sum(axis=axis, keepdims=True)

    return np.exp(data) / denominator