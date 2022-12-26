from typing import Optional, List, Any

from pydantic import validate_arguments
import torchvision.transforms as transforms

from auto_deep_learning.utils.constants import MEAN_CONSTANTS, STD_CONSTANTS
from auto_deep_learning.exceptions.data_handler.img import (
    InvalidArgumentType,
    InconsistentInput
)


class ImageTransformer(object):
    def __init__(
        self, 
        rotation: Optional[float] = 0.0,
        resize: Optional[float] = 224.0,
        resized_crop: Optional[float] = 224.0,
        horizontal_flip: Optional[bool] = False,
        color_jitter_brightness: Optional[float] = 0.0,
        color_jitter_saturation: Optional[float] = 0.0,
        color_jitter_contrast: Optional[float] = 0.0,
        color_jitter_hue: Optional[float] = 0.0,
        normalize: Optional[bool] = True,
        resize_enabled: Optional[bool] = False,
        resized_crop_enabled: Optional[bool] = True,
        color_jitter_enabled: Optional[bool] = False,
        type_validations: Optional[bool] = True,
        input_validation: Optional[bool] = True
    ) -> None:


        # Assertion of the type hints
        if type_validations:
            for numerical_field in [
                'rotation', 
                'resize', 
                'resized_crop', 
                'color_jitter_brightness', 
                'color_jitter_saturation', 
                'color_jitter_contrast', 
                'color_jitter_hue'
            ]:
                if not isinstance(locals()[numerical_field], float) and \
                    not isinstance(locals()[numerical_field], int):

                    raise InvalidArgumentType(
                        variable_name=numerical_field,
                        variable_expected_type='float'
                    )
            
            # TODO: Pass boolean fields into columns
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


        # Assertion of consistencies between the values
        if input_validation:
            if not resized_crop_enabled and resized_crop != 224.0:
                raise InconsistentInput(
                    msg=f'Resize Crop is not enabled, and has provided a Resized Crop dimension ({resized_crop})'
                )

            if not resize_enabled and resize != 224.0:
                raise InconsistentInput(
                    msg=f'Resize is not enabled, and has provided a Resized dimension ({resize_enabled})'
                ) 
        
            if (not color_jitter_enabled) and \
                (
                    not color_jitter_brightness == 0.0 or \
                    not color_jitter_contrast == 0.0 or \
                    not color_jitter_saturation == 0.0 or \
                    not color_jitter_hue == 0.0
                ):
                
                raise InconsistentInput(
                    msg=f'Color Jitter is not enabled, and has provided a Color Jitter dimension ({resize_enabled})'
                )

        self.rotation: float = rotation
        self.resize: float = resize
        self.resized_crop: float = resized_crop
        self.horizontal_flip: bool = horizontal_flip
        self.color_jitter_brightness: float = color_jitter_brightness
        self.color_jitter_saturation: float = color_jitter_saturation
        self.color_jitter_contrast: float = color_jitter_contrast
        self.color_jitter_hue: float = color_jitter_hue
        self.normalize: bool = normalize
        self.resize_enabled: bool = resize_enabled
        self.resized_crop_enabled: bool = resized_crop_enabled
        self.color_jitter_enabled: bool = color_jitter_enabled


    def __str__(self) -> str:
        str_image_tramsformer: str = f"""ImageTransformer({self.rotation}, {self.resize}, {self.resized_crop}, {self.horizontal_flip}, {self.color_jitter_brightness}, {self.color_jitter_saturation}, {self.color_jitter_contrast}, {self.color_jitter_hue}, {self.normalize}, {self.resize_enabled}, {self.resized_crop_enabled}, {self.color_jitter_enabled})"""

        return str_image_tramsformer

    def __repr__(self) -> str:
        repr_image_transformer: str = f"""ImageTransformer({self.rotation}, {self.resize}, {self.resized_crop}, {self.horizontal_flip}, {self.color_jitter_brightness}, {self.color_jitter_saturation}, {self.color_jitter_contrast}, {self.color_jitter_hue}, {self.normalize}, {self.resize_enabled}, {self.resized_crop_enabled}, {self.color_jitter_enabled})"""

        return repr_image_transformer
    
    def create(self) -> transforms.Compose:
        transformations: List[Any] = [transforms.RandomRotation(self.rotation)] if self.rotation else []
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