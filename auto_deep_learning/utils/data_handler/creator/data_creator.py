from typing import Dict, List, Optional

import pandas as pd
import torchvision.transforms as transforms

from auto_deep_learning.utils.config import ConfigurationObject

from .base_creator import Creator

conf_obj = ConfigurationObject()


class DataCreator:
    def __init__(
            self,
            csv_data_path: Optional[str] = 'data.csv',
            df: Optional[pd.DataFrame] = None,
            not_class_info: List[str] = ['image_path', 'split_type'],  # TODO: As constants
            sampler_activated: Optional[bool] = False
    ):
        """Create the DataLoaders for each of the split types.

        Args:
            csv_data_path (Optional[str], optional): the path to the csv data for the dataframe. Defaults to 'data.csv'.
            df (Optional[pd.DataFrame], optional): dataframe with the information already loaded. Defaults to None.
            not_class_info (List[str], optional): the columns of the dataframe that do not refer to the classifications. Defaults to ['image_path', 'split_type'].
        """

        # TODO: Assess csv_data_path has the .csv extension and is file type of csv
        # TODO: Support other types of extensions

        self.sampler_activated = sampler_activated
        self.df = df if df else pd.read_csv(csv_data_path)

        self.class_groups = [
            class_group for class_group in self.df.columns.values.tolist() if class_group not in not_class_info
        ]

        # TODO: Make them as properties
        self.df_dummies = {}
        self.dict_mapping_idx_class = {}
        # Get the dummies & and mapping idx-class_value  for that class group
        for class_group in self.class_groups:
            self.__get_dummies_df(class_group)
            self.__get_dummies_mapping(class_group)

    @classmethod
    def get_data_creators(
            cls,
    ) -> Dict[str, Creator]:

        # TODO: upsapler/downsampler depending if is activated, with seed
        if cls.sampler_activated:
            pass

        # TODO: Train/Test/Valid split from seed (for small ones, do not do valid)

        # TODO: Get loaders with df already ordered by class names (so as all splits have same class, get dummies idx_class will be the same)
        dict_loader = {}
        for split_type in cls.df['split_type'].unique().tolist():
            transformation: transforms.Compose = conf_obj.img_transformers[split_type].create()

            # Create the creator for that split type
            dict_loader[split_type] = Creator(
                df=cls.df[cls.df['split_type'] == split_type],
                class_groups=cls.class_groups,
                df_dummies=cls.df_dummies,
                transformation=transformation,
            )

        # TODO: Need to pass dict_loader, but also the group classes & number of class values for each
        return dict_loader

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
