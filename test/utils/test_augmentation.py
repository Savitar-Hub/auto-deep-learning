from auto_deep_learning.utils import ImageTransformer

class TestImageAugmentation():
    def test_instantiation(self):
        new_transformer = ImageTransformer()

        assert str(new_transformer) == 'ImageTransformer(0, 224, 224, False, 0, 0, 0, 0, True, False, True, False)'
        assert repr(new_transformer) == 'ImageTransformer(0, 224, 224, False, 0, 0, 0, 0, True, False, True, False)'
        assert isinstance(new_transformer, ImageTransformer)

    def test_creation_augmentation(self):
        pass

    # TODO: Test for invalid arguments
