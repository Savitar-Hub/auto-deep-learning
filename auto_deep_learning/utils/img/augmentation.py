from typing import Optional

import torchvision.transforms as transforms

from auto_deep_learning.utils.constants import MEAN_CONSTANTS, STD_CONSTANTS


class ImageTransformer(type):
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
        color_jitter_enabled: Optional[bool] = True
    ):

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
        str_image_tramsformer = f"""ImageTransformer(
            {self.rotation},
            {self.resize},
            {self.resized_crop},
            {self.horizontal_flip},
            {self.color_jitter_brightness},
            {self.color_jitter_saturation},
            {self.color_jitter_contrast},
            {self.color_jitter_hue},
            {self.normalize},
            {self.resize_enabled},
            {self.resized_crop_enabled},
            {self.color_jitter_enabled}
        )"""

        return str_image_tramsformer

    def __repr__(self):
        repr_image_transformer = f"""ImageTransformer(
            {self.rotation},
            {self.resize},
            {self.resized_crop},
            {self.horizontal_flip},
            {self.color_jitter_brightness},
            {self.color_jitter_saturation},
            {self.color_jitter_contrast},
            {self.color_jitter_hue},
            {self.normalize},
            {self.resize_enabled},
            {self.resized_crop_enabled},
            {self.color_jitter_enabled}
        )"""

        return repr_image_transformer
    
    def create(self):
        transformations = [transforms.RandomRotation(self.rotation)] if self.rotation else []
        transformations += [transforms.RandomResizedCrop(224)] if self.resize_enabled else []
        transformations += [transforms.RandomHorizontalFlip()] if self.horizontal_flip else []
        transformations += [
            transforms.ColorJitter(
                saturation=self.color_jitter_saturation, 
                contrast=self.color_jitter_contrast, 
                hue=self.color_jitter_hue, 
                brightness=self.color_jitter_brightness
            )
        ] if self.color_jitter_enabled else []
        transformations += [transforms.ToTensor()]
        transformations += [
            transforms.Normalize(
                MEAN_CONSTANTS, 
                STD_CONSTANTS
            )
        ]
            
        return transforms.Compose(transformations)