from numpy import np

# Set a random seed in numpy
np.random.seed(0)

class DataSetSampler():
    def __init__(
        self,
        data,
        is_dataframe: bool = False
    ):

        self.is_dataframe = is_dataframe
    
    def get_loader(self):
        if self.is_dataframe:
            pass
