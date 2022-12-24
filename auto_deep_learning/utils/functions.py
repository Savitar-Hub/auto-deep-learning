import os
import logging
import pandas as pd
from typing import List, Dict


def create_df_image_folder(
    columns: List[str] = [
        'image_path',
        'class',
        'split_type'
    ],
    dtypes: Dict[str, str] = {
        'image_path': 'object',
        'class': 'category',
        'split_type': 'category'
    }
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

