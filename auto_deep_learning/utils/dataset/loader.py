from typing import Optional, List

import pandas as pd
from torch.utils.data import Dataset

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class Loader(Dataset):
    def __init__(
        self,
        transformation,
        csv_data_path: Optional[str] = '.',
        df: Optional[pd.DataFrame] = None,
        not_class_info: List[str] = ['image_path', 'split_type']
    ):
        
        # TODO: Type hint transformation
        print(type(transformation), 'transformation type')
        self.transformation = transformation
        self.csv_data_path = csv_data_path
        self.df = df if df else pd.read_csv(csv_data_path)

        self.class_groups = [
            class_group for class_group in self.df.columns.values.tolist() if class_group not in not_class_info
        ]


    