import numpy as np

def standardize(data: np.array) -> np.array:
    """Function performing (x - mean) / sd standarization

    Args:
        data (np.array): _description_

    Returns:
        np.array: _description_
    """
    mean_ = data.mean(axis=0)
    sd_ = data.std(axis=0)
    
    return (data - mean_) / sd_