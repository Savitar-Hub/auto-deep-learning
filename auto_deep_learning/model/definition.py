from typing import Optional, List, Dict, Tuple

from sentence_transformers import SentenceTransformer, util

from auto_deep_learning.enum import (
    ModelObjective,
    ModelName
)
from auto_deep_learning.utils import DatasetSampler
from auto_deep_learning.utils.functions import check_numerical_value
from auto_deep_learning.exceptions.model import (
    IncorrectCategoryType
)
from .arch.convolution import SimpleConvNet


def get_category_similarity(
    category_type: str
) -> float:
    """Get the category similarity between purpose of Imagenet (classify objects), with the new dataset (f.e. classify fashion).

    Args:
        category_type (str): what is the category of the dataset.

    Returns:
        cosine_scores: the degree of similarity between the two words
    """

    words: List[str, str] = ["Objects", category_type]

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    objects_embedding, category_embedding = model.encode(words)

    cosine_scores: float = util.pytorch_cos_sim(
        objects_embedding, 
        category_embedding
    )

    return cosine_scores


def define_model(
    data: DatasetSampler,  # TODO: Model adapts to this: need to know number of group classes & class values
    category_type: Optional[str] = '',
    objective: Optional[ModelObjective] = 'throughput',
    model_name: Optional[ModelName] = '',
    model_version: Optional[str] = '',
    input_shape: Optional[Tuple[int]] = (224, 224) 
):
    """Definition of which will be the final model

    Args:
        data (DatasetSampler): the loader of the data that we are going to use
        category_type (Optional[str], optional): which category of objects tried to be classified. Defaults to ''.
        objective (Optional[ModelObjective], optional): how is this model going to be used. Defaults to 'throughput'.
        model_name (Optional[ModelName], optional): if we want to choose the model name. Defaults to ''.
        model_version (Optional[str], optional): if we want to specify the model version (related to model name). Defaults to ''.

    Returns:
        model: deep learning architecture
    """

    map_class_name_length: Dict[str, int] = {key: len(values) for key, values in data.dict_mapping_idx_class}


    if model_name:
        if model_version:
            model = ... # TODO: Get the model
            # Final layers architecture
            

            return model

    # Get amount of records that do we have
    if len(data) < 5000 and data.class_group_num < 3:
        # Some simple model
        model = SimpleConvNet(
            input_shape,
            map_class_name_length,
        )

        return model

    if len(category_type.split(' ')) > 1:
        raise IncorrectCategoryType(
            category_type=category_type,
            msg='Should be only one word'
        )
    
    if check_numerical_value(category_type=category_type):
        raise IncorrectCategoryType(
            category_type=category_type,
            msg='Category type should not contain numerical values'
        )

    category_similarity: float = get_category_similarity(
        category_type=category_type
    )

    if category_similarity > .7 and \
        objective == 'throughput':

        # Use vit
        # Freede intermediate
        # Create final layers
        return model
    
    elif category_similarity > 0.3:
        if objective == 'throughput':
            # Use the beit or swin (not the largest one)
            # Without freezing
        

            return model
        
        if objective == 'accuracy':
            # Use the beit or swin (the largest one)
            # Without freezing

            return model
    
    # Do now freeze intermediate layers
    pass