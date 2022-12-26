import os
from typing import Optional, List
from PIL import Image

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
import torchvision.transforms as transforms


# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class Loader(Dataset):
    def __init__(
        self,
        transformation: transforms.Compose,
        csv_data_path: Optional[str] = 'data.csv',
        df: Optional[pd.DataFrame] = None,
        not_class_info: List[str] = ['image_path', 'split_type'],  # TODO: As constans
        sampler_activated: Optional[bool] = False
    ):
        
        self.transformation = transformation
        self.csv_data_path = csv_data_path
        self.df = df if df else pd.read_csv(csv_data_path)
        self.class_groups = [
            class_group for class_group in self.df.columns.values.tolist() if class_group not in not_class_info
        ]

        # TODO: Get dummies for each of the classes


    @property
    def __len__(self) -> int:
        return len(self.df)
    

    @property
    def columns(self) -> List[str]:
        return self.df.columns.values.tolist()
    

    @property
    def class_groups(self) -> List[List[str]]:
        return self.class_groups
    

    @property
    def class_group_num(self) -> int:
        return len(self.class_groups)


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
        img_path = self.df.at[idx, 'image_path']
        image = Image.open(img_path)

        # TODO: Based on the class label, get which is the index based on the get_dummies
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)

        # TODO: Return as many classes as class that we have
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

    """
    If in form of imagefolder, custom one knowing that the folder name is the class.
        hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
                                           transform=data_transform)
        dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)
    If not, should create the custom one, returning also the classes
    """


class DataLoders:
    def __init__(
        self,
        transformation: transforms.Compose,
        csv_data_path: Optional[str] = 'data.csv',
        df: Optional[pd.DataFrame] = None,
        not_class_info: List[str] = ['image_path', 'split_type'],  # TODO: As constans
        sampler_activated: Optional[bool] = False
    ):

        self.transformation = transformation
        self.csv_data_path = csv_data_path
        self.sampler_activated = sampler_activated
        self.df = df if df else pd.read_csv(csv_data_path)

        self.class_groups = [
            class_group for class_group in self.df.columns.values.tolist() if class_group not in not_class_info
        ]
    

    @classmethod
    def get_loaders(
        self,
    ):
        # TODO: upsapler/downsampler depending if is activated
        if self.sampler_activated:
            pass
    
        
