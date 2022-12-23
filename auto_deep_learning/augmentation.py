import torchvision.transforms as transforms


class ImageTransformer(type):
    def __init__(
        self, 
        rotation: int = 0,
        resize: int = 224,
        resized_crop: int = 224,
        horizontal_flip: int = 0,
        color_jitter_brightness: int = 0,
        color_jitter_saturation: int = 0,
        color_jitter_contrast: int = 0,
        color_jitter_hue: int = 0,
        normalize: bool = True,
        resize_enabled: bool = False,
        resized_crop_enabled: bool = True
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


    def __repr__(self):
        repr = f"""ImageTransformer(
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
            {self.resized_crop_enabled}
        )"""

        return repr
