import base64
import os
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from auto_deep_learning.exceptions.utils.functions import \
    IncorrectFolderStructure
from auto_deep_learning.utils.functions import (create_df_image_folder,
                                                image_folder_convertion)

BASE_DIR = './tests/utils'
CHILD_DIR = ['train', 'test', 'valid']  # TODO: As constants, which are the allowed split types -> Schemas Enum.


def remove_test_dir():
    """Remove all the images that are previously in the images folder."""

    for child in CHILD_DIR:
        child_path = BASE_DIR + '/images/' + child

        if os.path.exists(child_path) and \
                os.path.isdir(child_path):

            shutil.rmtree(child_path)


@pytest.fixture()
def fill_images():
    """Convert the binary files into the images that we want"""
    remove_test_dir()

    for child in CHILD_DIR:
        child_path_raw = BASE_DIR + '/raw_images/' + child
        child_path_clean = BASE_DIR + '/images/' + child

        for img_class in os.listdir(child_path_raw):
            images_path = child_path_raw + '/' + img_class

            for img in os.listdir(images_path):
                img_path = images_path + '/' + img

                # Open the binary libraries and read them
                with open(img_path, 'rb') as f:
                    png_encoded = base64.b64decode(f.read())

                # Write them in the new location
                output_path = child_path_clean + '/' + img_class
                os.makedirs(output_path, exist_ok=True)

                output_file = output_path + '/' + img[:-3] + 'jpg'
                with open(output_file, 'wb') as f:
                    f.write(png_encoded)


@pytest.fixture(scope='session', autouse=True)
def cleanup(request):
    """
    Remove the images directory once it is finished
    """

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
        self,
        fill_images,
    ):

        df = image_folder_convertion('./tests/utils/images')

        assert not df.empty
        assert df.shape == (105, 3)
        assert set(df['split_type'].unique()) == {
            'test',
            'valid',
            'train'
        }

        assert set(df['class'].unique()) == {
            'ADONIS',
            'CRECENT',
            'ZEBRA LONG WING',
            'RED SPOTTED PURPLE',
            'GREY HAIRSTREAK'
        }

        for idx, row in df.iterrows():
            if not row['image_path'].endswith('.jpg'):
                assert False

    def test_no_images_folder(
        self
    ):
        remove_test_dir()

        try:
            image_folder_convertion('./tests/utils/images')
            assert False

        except IncorrectFolderStructure:
            assert True

    def test_imbalanced_classes(
        self,
        fill_images
    ):
        """
        Test to check that we need to have the same number of unique classes in both train/valid/test.

        Args:
            fill_images (function): fill the images folder with certain images
        """

        # We remove a certain class
        shutil.rmtree('./tests/utils/images/test/ADONIS')

        try:
            df = image_folder_convertion('./tests/utils/images')

        except:
            assert True
