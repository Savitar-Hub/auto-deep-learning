from typing import Optional

from pydantic import validate_arguments
import torchvision.transforms as transforms

from auto_deep_learning.utils.constants import MEAN_CONSTANTS, STD_CONSTANTS
from auto_deep_learning.utils.exceptions.data_handler.img import InvalidArgumentType


class ImageTransformer(object):
    def __init__(
        self, 
        rotation: Optional[int] = 0,
        resize: Optional[int] = 224,
        resized_crop: Optional[int] = 224,
        horizontal_flip: Optional[bool] = False,
        color_jitter_brightness: Optional[int] = 0,
        color_jitter_saturation: Optional[int] = 0,
        color_jitter_contrast: Optional[int] = 0,
        color_jitter_hue: Optional[int] = 0,
        normalize: Optional[bool] = True,
        resize_enabled: Optional[bool] = False,
        resized_crop_enabled: Optional[bool] = True,
        color_jitter_enabled: Optional[bool] = False
    ) -> None:

        # TODO: Refactor this assertions with new function --> function.py utilities
        # Assertion of the type hints
        for numerical_field in [
            'rotation', 
            'resize', 
            'resized_crop', 
            'color_jitter_brightness', 
            'color_jitter_saturation', 
            'color_jitter_contrast', 
            'color_jitter_hue'
        ]:
            if not isinstance(locals()[numerical_field], int):
                raise InvalidArgumentType(
                    variable_name=numerical_field,
                    variable_expected_type='int'
                )
        
        for boolean_field in [
            'horizontal_flip',
            'normalize',
            'resize_enabled',
            'resized_crop_enabled',
            'color_jitter_enabled'
        ]:
            if not isinstance(locals()[boolean_field], bool):
                raise InvalidArgumentType(
                    variable_name=boolean_field,
                    variable_expected_type='bool'
                )

        self.rotation = rotation
        self.resize = resize
        self.resized_crop = resized_crop
        self.horizontal_flip = horizontal_flip
        self.color_jitter_brightness = color_jitter_brightness
        self.color_jitter_saturation = color_jitter_saturation
        self.color_jitter_contrast = color_jitter_contrast
        self.color_jitter_hue = color_jitter_hue
        self.normalize = normalize
        self.resize_enabled = resize_enabled
        self.resized_crop_enabled = resized_crop_enabled
        self.color_jitter_enabled = color_jitter_enabled


    def __str__(self):
        str_image_tramsformer = f"""ImageTransformer({self.rotation}, {self.resize}, {self.resized_crop}, {self.horizontal_flip}, {self.color_jitter_brightness}, {self.color_jitter_saturation}, {self.color_jitter_contrast}, {self.color_jitter_hue}, {self.normalize}, {self.resize_enabled}, {self.resized_crop_enabled}, {self.color_jitter_enabled})"""

        return str_image_tramsformer

    def __repr__(self) -> str:
        repr_image_transformer = f"""ImageTransformer({self.rotation}, {self.resize}, {self.resized_crop}, {self.horizontal_flip}, {self.color_jitter_brightness}, {self.color_jitter_saturation}, {self.color_jitter_contrast}, {self.color_jitter_hue}, {self.normalize}, {self.resize_enabled}, {self.resized_crop_enabled}, {self.color_jitter_enabled})"""

        return repr_image_transformer
    
    def create(self) -> transforms.Compose:
        transformations = [transforms.RandomRotation(self.rotation)] if self.rotation else []
        transformations += [transforms.RandomResizedCrop(224)] if self.resize_enabled else []
        transformations += [transforms.RandomHorizontalFlip()] if self.horizontal_flip else []
        if self.color_jitter_enabled:
            transformations += [
                transforms.ColorJitter(
                    saturation=self.color_jitter_saturation, 
                    contrast=self.color_jitter_contrast, 
                    hue=self.color_jitter_hue, 
                    brightness=self.color_jitter_brightness
                )
            ]
        transformations += [transforms.ToTensor()]
        transformations += [
            transforms.Normalize(
                MEAN_CONSTANTS, 
                STD_CONSTANTS
            )
        ]
            
        return transforms.Compose(transformations)