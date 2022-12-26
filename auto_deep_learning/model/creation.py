"""
Things we take into consideration for creating the model:
- Complexity of the task: how many images and how many classes we want to infer
- Objective of the model: if a model if for production for a company, we might want to optimize tradeoff between throughtput and accuracy

The training of the model:
- Similarity with previous dataset: most of the models are trained with Imagenet and then we apply Transfer Learning. So depending on similarity, we make some warmup of the weights of the whole feature selection for some epochs or we only train the last layers.
- Type of the model: each model supported has the HP for which it was trained
"""

from typing import Optional
from auto_deep_learning.enum import (
    ModelObjective,
    ModelName
)
from auto_deep_learning.utils import Loader
from auto_deep_learning.model.creation import define_model


class Model:
    def __init__(
        self,
        data: Loader,
        description: Optional[str] = '',
        objective: Optional[ModelObjective] = 'throughput',
        model_name: Optional[ModelName] = '',
        model_version: Optional[str] = ''
    ):
        """Instance of the Neural Network model.

        Args:
            data (Loader): the loader of the data that will be used
            description (Optional[str], optional): short description of which task do you want to do. Defaults to None.
            objective (Optional[ModelObjective], optional): definition of which is the objective. Defaults to throughput, as are the simpler and optimized for production environments.
            model_name (Optional[ModelName], optional): definition of which is the model name (from HuggingFace). 
            model_version (Optional[str], optional): definition of which is the model version (from HuggingFace).
        """
        
        self.data = data
        self.objective = objective
        self.model_name = model_name
        self.description = description
        self.model_version = model_version

        self.model = define_model(
            data=self.data,
            description=self.description,
            objective=self.objective,
            model_name=self.model_name,
            model_version=self.model_version
        )

        