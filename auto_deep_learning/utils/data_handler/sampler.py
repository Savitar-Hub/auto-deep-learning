from numpy import np

from auto_deep_learning.utils import Loader

# Set a random seed in numpy
np.random.seed(0)

class DataSetSampler():
    def __init__(
        self,
        loader: Loader,
    ):

        self.loader = loader
    
    def get_sampler(self):
        if self.is_dataframe:
            pass
