from typing import List, Dict, Any
from pydantic import BaseModel, Field
from app.schemas.common import MetaData 

class MlTrainRequest(BaseModel):
    """
    Requête pour entraîner un modèle
    """
    meta: MetaData
    params: Dict[str, Any] = Field(
        ..., 
        description="Contient 'model_type' (logreg, rf)"
    )


class MlPredictRequest(BaseModel):
    """
    Requête pour faire des prédictions
    """
    meta: MetaData
    data: List[Dict[str, Any]] = Field(..., description="Données sans target")
    params: Dict[str, Any] = Field(..., description="Contient 'model_id'")