class OptimizerType(Enum):
    SGD = 'sgd'
    ADAM = 'adam'
    
    

class ModelObjective(Enum):
    """Define which is the model objective
    If you want to make research, it is recommended to use for accuracy.
    If you are creating a startup or is for a company project, where speed is important, use throughput.

    Args:
        Enum
    """

    ACCURACY = 'accuracy'
    THROUGHPUT = 'throughput' 


class ModelName(Enum):
    """
    Specify which is the model name that you want to use.
    The actual supported models are the following ones.
    """

    VIT = 'vit'
    DIT = 'dit'
    SWIN = 'swin'
    BEIT = 'beit'
    LEVIT = 'levit'
    CONVEX = 'convex'
    RESTNET_50 = 'resnet-50'