from pydantic import BaseModel


class ModelPredictionClass(BaseModel):
    class_name: str
    class_prediction: float
