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
    def class_group_unique(
        self,
        column: str
    ) -> List[str]:
    
        return self.df[column].unique().tolist()

    

    """
    If in form of imagefolder, custom one knowing that the folder name is the class.
        hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
                                           transform=data_transform)
        dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)
    If not, should create the custom one, returning also the classes
    """