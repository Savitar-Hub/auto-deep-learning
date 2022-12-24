from torchvision import datasets


# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class Loader():
    def __init__(
        self,
        transformation,
        csv_data_path: str,
    ):
        
        self.transformation = transformation
        self.csv_data_path = csv_data_path


    """
    If in form of imagefolder, custom one knowing that the folder name is the class.
        hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
                                           transform=data_transform)
        dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)
    If not, should create the custom one, returning also the classes
    """