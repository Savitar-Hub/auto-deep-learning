import os
import logging
import pandas as pd
from typing import List, Dict

from auto_deep_learning.exceptions.utils.functions import (
    ImbalancedClassError, 
    IncorrectFolderStructure,
    InvalidFileType
)


def check_numerical_value(
    category_type: str
):  
    """Check if inside a string we have a numerical value.
    This is becasue in the category_type provided to our model object, must only have characters.

    Args:
        category_type (str): the category type of the dataset
    """

    return any(char.isdigit() for char in category_type)


def create_df_image_folder(
    columns: List[str], dtypes: Dict[str, str]
) -> pd.DataFrame:
    """
    Create the dataframe that will be the base for the image folder dataset.
    
    Parameters
    ----------
    columns : list
        The columns that will be used for the dataframe
    
    dtypes : dict
        Dictionary for the column name : column dtype for the dataframe
    
    Returns
    -------
    df : pd.DataFrame
        The DataFrame that will be used for organizing the Image Folder information
    """

    df = pd.DataFrame(columns=columns)

    df = df.astype(dtypes)

    return df


def image_folder_convertion(
    parent_folder_path: str,
    columns: List[str] = [
        'image_path',
        'class',
        'split_type'
    ],
    dtypes: Dict[str, str] = {
        'image_path': 'object',
        'class': 'category',
        'split_type': 'category'
    },
    save: bool = False,
    save_path: str = 'data.csv'

):
    """
    Function to make the conversion from the image folder structure to a dataframe that is wanted.
    
    Parameters
    ----------
    parent_folder_path : str
        The folder for which we have the ['train', 'test'] folders.

    columns : list
        The columns that will be used for the dataframe
    
    dtypes : dict
        Dictionary for the column name : column dtype for the dataframe
    
    save : bool
        If we want to save the dataframe that we get
    
    Returns
    -------
    df : pd.DataFrame
        The DataFrame that will be used for organizing the Image Folder information
    """

    # TODO: Assess parent_folder str has not trailing /

    # Get the df we will use as the base one 
    df = create_df_image_folder(
        columns=columns,
        dtypes=dtypes
    )

    # We get the folders, which we expect to be ['train', 'test', 'valid']
    split_folders: List[str] = os.listdir(parent_folder_path)

    if len(split_folders) == 0:
        raise IncorrectFolderStructure(msg='No folders found')

    # TODO: Assess folder structure is correct
    for folder in split_folders:
        if folder not in ['train', 'valid', 'test']:
            raise IncorrectFolderStructure(folder_structure=split_folders)
    
    for split_folder in split_folders:
        # We get the path of .../train and .../test
        split_path: str = parent_folder_path + '/' + split_folder

        # List the values that we have inside of each folder, which would be the classes
        children_classes: List[str] = os.listdir(split_path)

        # For each class of each folder
        for child_class in children_classes:
            # Get the path of .../train/class
            images_path: str = split_path + '/' + child_class

            # List the images we have inside that folder
            images_child_class_path: List[str] = os.listdir(images_path) 

            for idx, file in enumerate(images_child_class_path):
                if not file.endswith('.jpg') or file.endswith('.png'):
                    raise InvalidFileType(file=file)

                images_child_class_path[idx] = images_path + '/' + file

            class_list: List[str] = [child_class] * len(images_child_class_path)
            dtype_list: List[str] = [split_folder] * len(images_child_class_path)

            # TODO: This is hardtypping the columns, so not flexible (ImageFolder is not flexible neither)
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data = {
                            'class': class_list,
                            'split_type': dtype_list,
                            'image_path': images_child_class_path,
                    })
                ], axis=0
            )

            del images_path
            del images_child_class_path
            del class_list
            del dtype_list

    
    # Checks once the pandas dataframe created, classes in train = test
    unique_per_class = df.groupby('split_type')['class'].unique()

    unique_per_class_test = list(set(unique_per_class[0]))
    unique_per_class_train = list(set(unique_per_class[1]))
    unique_per_class_valid = list(set(unique_per_class[2]))


    for idx in range(max(
        len(unique_per_class_test),
        len(unique_per_class_train),
        len(unique_per_class_valid)
    )):

        try:
            if not unique_per_class_test[idx] == unique_per_class_train[idx] == unique_per_class_valid[idx]:
                raise ImbalancedClassError()

        except IndexError:
            raise ImbalancedClassError()

    if save:
        df.to_csv(save_path, index=False, header=True)
        
    return df


# TODO: tensor_dict = torch.load('model.dat', map_location='cpu') # OrderedDict
# TODO: In Model, not only save hp but also the whole model if asked -> save on cpu