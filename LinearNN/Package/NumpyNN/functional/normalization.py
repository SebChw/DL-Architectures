import numpy as np


def normalize(data : np.array) -> np.array:
    """Function performing minmax 0-1 normalization

    Args:
        data (np.array): data to be normalized

    Returns:
        np.array: normalized data
    """
    min_ = data.min(axis=0)
    max_ = data.max(axis=0)

    return ( data - min_ ) / (max_ - min_) #This works due to broadcasting, data has shape (Rows, Columns)
    #min_ and max_ has shape (1, Columns) and are broadcasterd (that 1 row is repeated Rows times) so that shapes match
    #and eventually elementwise operations are performed