from typing import List
from pydantic import BaseModel


class ModelPredictionClassUnit(BaseModel):
    """
    The prediction made for a class inside of a group of classes.
    """

    class_name: str
    class_prediction: float


class ModelPredictionUnit(BaseModel):
    """
    Multiple predictions made related to a class group.
    """
    prediction_unit: List[ModelPredictionClassUnit]


class ModelPrediction(BaseModel):
    """
    Predictions of the model made for one or more class groups.
    """
    
    prediction: List[ModelPredictionUnit]
