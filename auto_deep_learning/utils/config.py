from typing import Dict, Any

from .singleton import Singleton
from auto_deep_learning.utils.data_handler.transform.augmentation import ImageTransformer


class ConfigurationObject(metaclass=Singleton):
    def __init__(
        self,
        img_transformers: Dict[str, Any] =  {
            'train': ImageTransformer(
                rotation=3.0,
                color_jitter_brightness=3.0,
                color_jitter_contrast=3.0,
                color_jitter_hue=3.0,
                color_jitter_saturation=3.0,
                color_jitter_enabled=True,
                resized_crop_enabled=True
            ),
            'valid': ImageTransformer(),
            'test': ImageTransformer()
        },
        batch_size_train: int = 64,
        batch_size_valid: int = 128,
        batch_size_test: int = 128,
        valid_size: float = 64,
        test_size: float = 0.05,
        image_size: int = 224,
        num_workers: int = 6,
    ):

        self._img_transformers = img_transformers
        self._batch_size: Dict[str, int] = {
            'train': batch_size_train,
            'valid': batch_size_valid,
            'test': batch_size_test
        }
        self._valid_size: int = valid_size
        self._test_size: int = test_size
        self._image_size: int = image_size
        self._num_workers: int = num_workers  # TODO: https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4

    @property
    def batch_size(self):
        return self.batch_size
    

    @batch_size.setter
    def batch_size(self, split_type, new_batch_size):
        self._batch_size[split_type] = new_batch_size

        return self._batch_size
    

    @property
    def num_workers(self):
        return self._num_workers
    

    @num_workers.setter
    def num_workers(self, new_num_workers):
        self._num_workers = new_num_workers

        return self._num_workers

    @property
    def valid_size(self):
        return self._valid_size
    

    @valid_size.setter
    def valid_size(self, new_valid_size):
        self._valid_size = new_valid_size
        return self._valid_size

    
    @property
    def test_size(self):
        return self._test_size
    

    @test_size.setter
    def test_size(self, new_test_size):
        self._test_size = new_test_size
        return self._test_size


    @property
    def image_size(self):
        return self._image_size
    

    @image_size.setter
    def image_size(self, new_image_size):
        self._image_size = new_image_size
        return self._image_size