import numpy as np 
import pandas as pd


class Dataset:
    def __init__(self, data_path : str, target_name: str, categorical_attr_names: list, numerical_attr_names : list,
                pipeline_numerical = [], pipeline_categorical = [], separator=","):
        """This class works only for numerical data, that is in the form of csv file

        Args:
            data_path (str): path to the dataset
            target_name (str): name of the column with target attribute
            categorical_attr_names (list): list with names of categorical attributes
            numerical_attr_names (list): list of names with numerical attributes
            pipeline_numerical (list, optional): list of functions to be aplied on numerical data. Defaults to [].
            pipeline_categorical (list, optional): list of functions to be aplied on categorical data. Defaults to [].
            separator (str, optional): string separator in csv file. Defaults to ",".
        """
        self.data_path = data_path

        df = pd.read_csv(data_path, sep=separator)
        
        self.categorical_attr_names = categorical_attr_names
        self.numerical_attr_names = numerical_attr_names

        self.target_classes = df[target_name].unique() # getting target classes names
        
        #categorizing target classes
        self.target = df[target_name]
        self.target.replace(self.target_classes,
                        list(range(len(self.target_classes))), inplace=True)

        #passing numerical attributes through the pipeline
        self.numerical_attr = df[numerical_attr_names].to_numpy()
        for pipe in pipeline_numerical:
            self.numerical_attr = pipe(self.numerical_attr)
        
        #passing categorical attributes through the pipeline
        self.categorical_attr = df[categorical_attr_names].to_numpy()
        for pipe in pipeline_categorical:
            self.categorical_attr = pipe(self.categorical_attr)
        
        #merging numerical and categorical attributes
        self.dataset = np.concatenate((self.numerical_attr, self.categorical_attr), axis=1)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitems__(self, indices : list) -> np.array:
        """Function returning batch of elements at given indices

        Args:
            indices (list): indices of elements in the batch

        Returns:
            np.array: batch of size (N x A) where N = len(indices) and A = amount of attributes
        """
        return self.dataset[indices], self.target[indices]    