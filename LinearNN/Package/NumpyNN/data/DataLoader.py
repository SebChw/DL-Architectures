import numpy as np
from . import Dataset

class DataLoader:
    """Class that based on dataset creates batches 
    """
    def __init__(self, dataset : Dataset, batch_size = 64 ,shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

        
    def get_data(self,) -> np.array:
        """Generator that yields batches. It includes last batch even if it's size is smaller than

        Yields:
            np.array: batch.
        """
        indices = np.arange(self.dataset.__len__())

        if self.shuffle:
            np.random.shuffle(indices) # This function works in-place. This means that modifies original table
            #if function doesn't work in-place then it leave argument as it is and return new table which is modified

        full_batches = self.dataset.__len__() // self.batch_size
        indices_full = np.reshape(indices[: full_batches * self.batch_size], (full_batches, self.batch_size)) # we want to have array with shape (num_of_batches, batch_size)
        #We can write -1 in some dimension and it means that size of this dimension will be inferred based on the other shapes.
        #the problem is that usually dataset won't be equally divisible. One batch will be usually smaller, and we must calculate 
        #shape precisely
        rest_of_indices = indices[full_batches*self.batch_size:]

        for batch_indices in indices_full:
            yield self.dataset.__getitems__(batch_indices) #This is a generator due to yield

        yield self.dataset.__getitems__(rest_of_indices)


