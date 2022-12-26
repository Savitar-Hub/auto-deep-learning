class ModelObjective(Enum):
    """Define which is the model objective
    If you want to make research, it is recommended to use for accuracy.
    If you are creating a startup or is for a company project, where speed is important, use throughput.

    Args:
        Enum
    """

    accuracy = 'accuracy'
    throughput = 'throughput' 


class ModelName(Enum):
    """
    Specify which is the model name that you want to use.
    The actual supported models are the following ones.
    """

    vit = 'vit'
    dit = 'dit'
    swin = 'swin'
    beit = 'beit'
    levit = 'levit'
    convex = 'convex'
    resnet_50 = 'resnet-50'