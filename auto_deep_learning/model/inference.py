import cv2
from PIL import Image

import torch
import torchvision.transforms as transforms

from auto_deep_learning.model import Model
from auto_deep_learning.utils.constants import MEAN_CONSTANTS, STD_CONSTANTS

## TODO: the class names can be accessed at the `classes` attribute
## of your dataset object (e.g., `train_dataset.classes`)

def inference(
    model: Model,
    img_path: str, 
    top_k: int = 3,
    use_cuda: bool = torch.cuda.is_available()
):

    # Open the image
    image = Image.open(img_path)
    image = image.convert("RGB")
    
    image_transform = transforms.Compose(
        [
            transforms.Resize((224,224)), 
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CONSTANTS, STD_CONSTANTS)
        ]
    )

    # Convert to float and then to torch
    image = image_transform(image).float()
    image = torch.unsqueeze(image, 0)
    
    # Moving to cuda in the case there is GPU
    if use_cuda:
        image = image.cuda()
    
    model.eval()

    # TODO: This output can be multi-dimensional
    output = model(image) # Pass the image as the input
    values, index = output.topk(top_k) # Obtain the top k values
    
    top_output = []
    index = index.tolist()[0] # As it returns an array inside of an array
    # For the range of the index list

    # And we get through the answers
    for i in range(0, len(index)): 

        # TODO: Will append the class which corresponds ot the value of index at position i
        # top_output.append(splitted_classes[index[i]])
        pass
    
    # Get a Dict of output of class_group: class_type: prob.
    return top_output 