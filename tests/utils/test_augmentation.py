import pytest
import torchvision.transforms as transforms
from pydantic import ValidationError

from auto_deep_learning.utils import ImageTransformer
from auto_deep_learning.exceptions.utils import (
    InvalidArgumentType,
    InconsistentInput
)


@pytest.fixture()
def get_test_numerical_types():
    return [
        'False',
        True,
        {'dummy': 'dummy'},
        ('dummy', ),
        {'dummy'}
    ]


@pytest.fixture()
def get_numerical_fields():
    return [
        'rotation', 
        'resize', 
        'resized_crop', 
        'color_jitter_brightness', 
        'color_jitter_saturation', 
        'color_jitter_contrast', 
        'color_jitter_hue'
    ]


@pytest.fixture()
def get_test_boolean_types():
    return [
        'False',
        0,
        0.0,
        {'dummy': 'dummy'},
        ('dummy', ),
        {'dummy'}
    ]


@pytest.fixture()
def get_boolean_fields():
    return [
        'horizontal_flip',
        'normalize',
        'resize_enabled',
        'resized_crop_enabled',
        'color_jitter_enabled'
    ]


class TestImageAugmentation:

    def test_instantiation(self):
        new_transformer = ImageTransformer()

        assert str(new_transformer) == """ImageTransformer(0.0, 224.0, 224.0, False, 0.0, 0.0, 0.0, 0.0, True, True, False, False)"""
        assert repr(new_transformer) == """ImageTransformer(0.0, 224.0, 224.0, False, 0.0, 0.0, 0.0, 0.0, True, True, False, False)"""
        assert isinstance(new_transformer, ImageTransformer)
    

    def test_creation_augmentation(self):
        new_transformer_created = ImageTransformer(
            color_jitter_enabled=False
        ).create()

        assert isinstance(new_transformer_created, transforms.Compose)
        assert str(new_transformer_created.transforms) == str(transforms.Compose([
            transforms.Resize((224.0, 224.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ]).transforms)


    def test_inconsistent_input(self):
        try:
            ImageTransformer(
                color_jitter_brightness=10.0,
                color_jitter_enabled=False
            )
        
        except InconsistentInput:
            assert True


    @pytest.mark.parametrize("idx_test, ", range(7))
    def test_invalid_numerical_fields(
        self,
        get_numerical_fields, 
        get_test_numerical_types, 
        idx_test
    ):

        locals()['test_field'] = get_numerical_fields[idx_test]

        for test_numerical_type in get_test_numerical_types:
            numerical_mapping = {
                locals()['test_field']: test_numerical_type,
                'input_validation': False
            }
            
            try:
                ImageTransformer(**numerical_mapping)

            except InvalidArgumentType:
                assert True
    

    @pytest.mark.parametrize("idx_test, ", range(5))
    def test_invalid_boolean_fields(
        self,
        get_boolean_fields, 
        get_test_boolean_types, 
        idx_test
    ):

        locals()['test_field'] = get_boolean_fields[idx_test]

        for test_boolean_type in get_test_boolean_types:
            boolean_mapping = {
                locals()['test_field']: test_boolean_type,
                'input_validation': False
            }
            
            try:
                ImageTransformer(**boolean_mapping)
                assert False

            except InvalidArgumentType:
                assert True
        

     