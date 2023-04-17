import pandas as pd
import os

class DataExploration(object):
    """
    Data exploration class for the ML projects. Read in, visualize etc.

    ...

    """

    def __init__(self) -> None:

        self.datasets = {}

    def add_dataset(self, path_to_dataset, dataset_ID):

        self.datasets[dataset_ID] = {}
        self.datasets[dataset_ID]["path"] = path_to_dataset

        _, extension = os.path.splitext(path_to_dataset)

        if extension==".csv":
            self.datasets[dataset_ID]["data"] = pd.read_csv(path_to_dataset)
        else:
            raise NotImplementedError

    def describe_dataset(self, dataset_ID):
        print(self.datasets[dataset_ID]["data"].describe())

