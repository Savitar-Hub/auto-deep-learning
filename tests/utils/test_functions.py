import os
import shutil
import pytest
from auto_deep_learning.utils.functions import image_folder_convertion

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


class TestImageFolderConvertion:
    def test_creation_df(
        self
    ):
        os.mkdir(BASE_DIR + '/train')
        