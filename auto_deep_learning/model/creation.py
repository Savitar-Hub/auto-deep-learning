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
    ModelName,
    ModelVersion
)
from auto_deep_learning.utils import Loader


class Model:
    def __init__(
        self,
        data: Loader,
        objective: Optional[ModelObjective] = None,
        model_name: Optional[ModelName] = None,
        model_version: Optional[str] = None
    ):
        pass