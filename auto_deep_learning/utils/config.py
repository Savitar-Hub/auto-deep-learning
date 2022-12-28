from singleton import Singleton

# For batch processing
BATCH_SIZE = 64

# Train/Valid/Test Split
TEST_SIZE = 0.05
VALID_SIZE = 0.1


class ConfigurationObject(metaclass=Singleton):
    def __init__(
        self,
        batch_size: int = 64,
        valid_size: float = 64,
        test_size: float = 0.05
    ):

        self._batch_size = batch_size
        self._valid_size = valid_size
        self._test_size = test_size
    

    @property
    def batch_size(
        self
    ):
        return self.batch_size
    

    @batch_size.setter
    def batch_size(self, new_batch_size):
        self._batch_size = new_batch_size

        return self._batch_size
    

    @property
    def valid_size(
        self
    ):
        return self._valid_size
    

    @valid_size.setter
    def valid_size(self, new_valid_size):
        self._valid_size = new_valid_size
        return self._valid_size

    
    @property
    def test_size(
        self
    ):
        return self._test_size
    

    @test_size.setter
    def test_size(self, new_test_size):
        self._test_size = new_test_size
        return self._test_size
