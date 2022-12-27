from numpy import np

from torch.utils.data import DataLoader
from auto_deep_learning.utils import DataCreator

# Set a random seed in numpy
np.random.seed(0)

class DatasetSampler():
    def __init__(
        self,
        data_creator: DataCreator,
    ):

        self.data_creator = data_creator
    
    def get_sampler(self):
        # TODO: For very small datasets, we might do not have valid
        return [
            DataLoader(
                self.data_creator.get('train'), 
                batch_size=32, 
            ),
            DataLoader(
                self.data_creator.get('valid'), 
                batch_size=32, 
            ),
            DataLoader(
                self.data_creator.get('test'), 
                batch_size=32, 
            )
        ]
