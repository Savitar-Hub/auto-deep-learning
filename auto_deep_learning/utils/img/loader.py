from torchvision import datasets


class ImageLoaders():
    def __init__(
        self,
        transformation,
        image_folder_path: str,
        image_folder: bool = True,
    ):
        
        self.transformation = transformation
        self.image_folder_path = image_folder_path
        self.image_folder = image_folder


    def get(self):

        # If the path is on the form of /train/class/.img
        if self.image_folder:
            return datasets.ImageFolder(
                self.image_folder_path,
                transform=self.transformation
            )

        # If not, we need to define which is the class label associated with each image