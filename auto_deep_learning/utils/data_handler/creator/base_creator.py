from typing import Any, Dict, List

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class Creator(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        class_groups: List[str],
        df_dummies: pd.DataFrame,
        transformation: transforms.Compose,
    ):

        # TODO: Use _ for internal variables
        self.df = df
        self.transformation = transformation
        self.class_groups = class_groups

        self._columns = self.df.columns.values.tolist()
        self.df_dummies = df_dummies

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

    def __getitem__(
        self,
        idx
    ):

        if torch.is_tensor(idx):
            idx = idx.tolist()      # TODO: Convert to normal int, so we could use .at without returning series

        # Get which is the image path
        img_path: str = self.df.iloc[idx, 0]  # TODO: Try with .at
        image = Image.open(img_path).convert('RGB')  # TODO: Open with numpy maybe is faster

        class_groups_index: Dict[str, Any] = {}
        for class_group in self.class_groups:
            # Get positional index and convert into tensor
            class_groups_index[class_group] = torch.tensor(
                list(self.df_dummies[class_group].iloc[idx]).index(1),
            )

        # If we have transformations, apply it
        if self.transformation:
            image = self.transformation(image)

        # Return as many classes as class that we have, as well as the image
        sample: List[Any] = [image, class_groups_index]

        del img_path
        del class_group

        return sample
