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

        self.df = df     
        self.transformation = transformation
        self.class_groups = class_groups
        
        self.df_dummies = {}
        for class_group in class_groups:
            self.df_dummies[class_group] = pd.get_dummies(df, class_group)


    @property
    def __len__(self) -> int:
        return len(self.df)
    

    @property
    def columns(self) -> List[str]:
        return self.df.columns.values.tolist()
    
    """@property
    def class_groups_list(self) -> List[List[str]]:
        return self.class_groups
    

    @property
    def class_group_num(self) -> int:
        return len(self.class_groups)"""


    @property
    def class_group_unique(
        self,
        column: str
    ) -> List[str]:
    
        return self.df[column].unique().tolist()


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

        # TODO: Train/Test/Valid split from seed
    
        dict_loader = {}
        for split_type in self.df['split_tupe'].unique().tolist():
            dict_loader[split_type] = Creator(
                df=self.df,
                class_groups=self.class_groups,
                transformation=self.transformation,
            )
        
        return dict_loader


    
        
