from .singleton import Singleton


class ConfigurationObject(metaclass=Singleton):
    def __init__(
        self,
        batch_size: int = 64,
        valid_size: float = 64,
        test_size: float = 0.05,
        image_size: int = 224
    ):

        self._batch_size = batch_size
        self._valid_size = valid_size
        self._test_size = test_size
        self._image_size = image_size
    

    @property
    def batch_size(self):
        return self.batch_size
    

    @batch_size.setter
    def batch_size(self, new_batch_size):
        self._batch_size = new_batch_size

        return self._batch_size
    

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