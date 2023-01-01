from typing import Dict

import numpy as np
from torch.utils.data import DataLoader
from auto_deep_learning.utils import DataCreator
from auto_deep_learning.utils.config import ConfigurationObject

# Set a random seed in numpy
np.random.seed(0)
conf_obj = ConfigurationObject()


class DatasetSampler():
    def __init__(
        self,
        data_creator: DataCreator,
    ):

        # Store the Data Creator
        self.data_creator = data_creator

        # Store the information about the dataframe we are treating and class groups
        self.df = data_creator.df
        self.class_groups = data_creator.class_groups
        self.df_dummies = data_creator.df_dummies
        self.dict_mapping_idx_class = data_creator.dict_mapping_idx_class

    
    def get_sampler(self) -> Dict[str, DataLoader]:
        data_creators = self.data_creator.get_data_creators()

        return {
            data_creator_key: DataLoader(
                data_creators[data_creator_key],
                batch_size=conf_obj.batch_size[data_creator_key],
                num_workers=conf_obj.num_workers,
                shuffle=True,
                drop_last=True
            ) for data_creator_key in data_creators.keys()
        }
