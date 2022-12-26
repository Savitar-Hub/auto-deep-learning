import os
import shutil
import pytest
from typing import List, Dict

import pandas as pd

from auto_deep_learning.utils.functions import (
    create_df_image_folder, 
    image_folder_convertion
)


BASE_DIR = './tests/utils/images'
CHILD_DIR = ['train', 'test', 'valid']


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """
    Remove the images directory once it is finished
    """
    
    def remove_test_dir():
        for child in CHILD_DIR:
            child_path = BASE_DIR + '/' + child

            if os.path.exists(child_path) and \
                os.path.isdir(child_path):

                shutil.rmtree(child_path)

    request.addfinalizer(remove_test_dir)


class TestCreateDfImageFolder:
    def test_creation_df(
        self
    ):

        # TODO: Pass this as constants
        columns: List[str] = ['image_path', 'class', 'split_type']

        dtypes: Dict[str, str] = {
            'image_path': 'object',
            'class': 'category',
            'split_type': 'category'
        }

        df = create_df_image_folder(
            columns=columns,
            dtypes=dtypes
        )

        assert df.empty
        assert df.columns.values.tolist() == ['image_path', 'class', 'split_type']
        assert isinstance(df, pd.DataFrame)


class TestImageFolderConvertion:
    def test_creation_df(
        self
    ):
        for child in CHILD_DIR:
            child_path = BASE_DIR + '/' + child

            os.mkdir(child_path)
        

        