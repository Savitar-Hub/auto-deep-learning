import torchvision.transforms as transforms

from auto_deep_learning.utils import ImageTransformer

class TestImageAugmentation():
    def test_instantiation(self):
        new_transformer = ImageTransformer()

        assert str(new_transformer) == """ImageTransformer(
            0,
            224,
            224,
            False,
            0,
            0,
            0,
            0,
            True,
            False,
            True,
            True
        )"""
        assert repr(new_transformer) == """ImageTransformer(
            0,
            224,
            224,
            False,
            0,
            0,
            0,
            0,
            True,
            False,
            True,
            True
        )"""
        assert isinstance(new_transformer, ImageTransformer)

    def test_creation_augmentation(self):
        new_transformer_created = ImageTransformer(
            color_jitter_enabled=False
        ).create()

        assert isinstance(new_transformer_created, transforms.Compose)
        assert str(new_transformer_created.transforms) == str(transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ]).transforms)

    # TODO: Test for invalid arguments
