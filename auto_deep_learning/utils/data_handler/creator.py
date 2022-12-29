import os
from typing import Optional, List, Dict

import torch
import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Creator(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        class_groups: List[str],
        transformation: transforms.Compose,
    ):

        # TODO: Use _ for internal variables
        self.df = df     
        self.transformation = transformation
        self.class_groups = class_groups
        
        self._columns = self.df.columns.values.tolist()

        # TODO: Make them as properties
        self.df_dummies = {}
        self.dict_mapping_idx_class = {}
        # Get the dummies & and mapping idx-class_value  for that class group
        for class_group in self.class_groups:
            self.__get_dummies_df(class_group)
            self.__get_dummies_mapping(class_group)

    @property
    def __len__(self) -> int:
        return len(self.df)
    

    @property
    def columns(self) -> List[str]:
        return self._columns


    @columns.setter
    def columns(self, column_names: List[str]) -> List[str]:
        # Update values in df
        self.df.columns = column_names

        # Update values in our internal variable
        self._columns = self.df.columns.values.tolist()

        # Return that internal variable
        return self._columns


    @property
    def class_group_unique(
        self,
        column: str
    ) -> List[str]:
    
        return self.df[column].unique().tolist()


    def __get_dummies_df(
        self,
        class_group: str
    ) -> pd.DataFrame:

        dummy_df: pd.DataFrame = pd.get_dummies(
            pd.DataFrame(self.df.loc[:, class_group]), 
            columns=[class_group],
            prefix='',
            prefix_sep=''
        )

        self.df_dummies[class_group] = dummy_df

        return dummy_df

    
    def __get_dummies_mapping(
        self,
        class_group: str
    ) -> Dict[str, Dict[str, str]]:

        # For that class group, get which are the columns in the dummmies and create the mapping between idx and class name
        self.dict_mapping_idx_class[class_group] = {}

        for idx, class_value in enumerate(self.df_dummies[class_group].columns.values):
            self.dict_mapping_idx_class[class_group][str(idx)] = class_value
        
        return self.dict_mapping_idx_class


    def __getitem__(
        self,
        idx
    ):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get which is the image path
        img_path = self.df.iloc[idx, 0]
        image = Image.open(img_path)

        class_groups_index = {}
        for class_group in self.class_groups:            
            # Get positional index and convert into tensor
            class_groups_index[class_group] = torch.tensor(
                list(self.df_dummies[class_group].iloc[idx]).index(1),
            )
        
        # Return as many classes as class that we have, as well as the image
        sample = [image, class_groups_index]

        # If we have transformations, apply it
        if self.transformation:
            sample[0] = self.transformation(sample[0])

        return sample


class DataCreator:
    def __init__(
        self,
        transformation: transforms.Compose,
        csv_data_path: Optional[str] = 'data.csv',
        df: Optional[pd.DataFrame] = None,
        not_class_info: List[str] = ['image_path', 'split_type'],  # TODO: As constants
        sampler_activated: Optional[bool] = False
    ):
        """Create the DataLoaders for each of the split types.

        Args:
            transformation (transforms.Compose): the transformations that we want to apply to the images.
            csv_data_path (Optional[str], optional): the path to the csv data for the dataframe. Defaults to 'data.csv'.
            df (Optional[pd.DataFrame], optional): dataframe with the information already loaded. Defaults to None.
            not_class_info (List[str], optional): the columns of the dataframe that do not refer to the classifications. Defaults to ['image_path', 'split_type'].
        """

        # TODO: Assess csv_data_path has the .csv extension and is file type of csv
        # TODO: Support other types of extensions

        self.transformation = transformation
        self.sampler_activated = sampler_activated
        self.df = df if df else pd.read_csv(csv_data_path)

        self.class_groups = [
            class_group for class_group in self.df.columns.values.tolist() if class_group not in not_class_info
        ]
    

    @classmethod
    def get_loaders(
        self,
    ) -> Dict[str, Creator]:

        # TODO: upsapler/downsampler depending if is activated, with seed
        if self.sampler_activated:
            pass

        # TODO: Train/Test/Valid split from seed (for small ones, do not do valid)
    
        # TODO: Get loaders with df already ordered by class names (so as all splits have same class, get dummies idx_class will be the same)
        dict_loader = {}
        for split_type in self.df['split_type'].unique().tolist():
            # Create the creator for that split type
            dict_loader[split_type] = Creator(
                df=self.df[self.df['split_type'] == split_type],
                class_groups=self.class_groups,
                transformation=self.transformation,
            )
        
        # TODO: Need to pass dict_loader, but also the group classes & number of class values for each
        return dict_loader


    
        
