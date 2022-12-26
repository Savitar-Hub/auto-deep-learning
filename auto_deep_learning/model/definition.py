from typing import Optional
from auto_deep_learning.enum import (
    ModelObjective,
    ModelName
)
from auto_deep_learning.utils import Loader


def define_model(
    data: Loader,
    description: Optional[str] = '',
    objective: Optional[ModelObjective] = 'throughput',
    model_name: Optional[ModelName] = '',
    model_version: Optional[str] = ''
):

    if model_name:
        if model_version:
            model = ... # TODO: Get the model
            # Final layers architecture

            return model

    # Get amount of records that do we have
    if len(data) < 10.000 and data.class_group_num < 40:
        # Some simple model

        return model

    description_similarity: float = 0.0

    if description_similarity > .9 and \
        objective == 'throughput':

        # Use vit
        # Freede intermediate
        # Create final layers
        return model
    
    elif description_similarity > 0.5:
        if objective == 'throughput':
            # Use the beit or swin (not the largest one)
            # Without freezing
        

            return model
        
        if objective == 'accuracy':
            # Use the beit or swin (the largest one)
            # Without freezing
    

    # Do now freeze intermediate layers
    