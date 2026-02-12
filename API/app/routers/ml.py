"""
Router pour Machine Learning baseline (TP4 - ML)
Endpoints : /ml/train, /ml/metrics, /ml/predict, /ml/model-info
"""

import pandas as pd

from fastapi import APIRouter, HTTPException
from app.schemas.common import MetaData, StandardResponse
from app.schemas.ml import MlTrainRequest, MlPredictRequest
from app.services.dataset_generator import DatasetGenerator, store_model, get_model
from app.services.ml_service import MlService
from app.repositories.dataset_store import dataset_store


router = APIRouter()

@router.post("/train", response_model=StandardResponse)
def ml_train(dataset_id: str, model_type: str = "logreg"):
    """
    Entraîne un modèle de classification binaire
    
    **Paramètres** :
    - `model_type` : Type de modèle (logreg, rf)
    
    **Retour** :
    - model_id
    - Métriques train/test (accuracy, precision, recall, f1, AUC)
    - Features utilisées
    
    **Exemple** :
    ```json
    {
        "meta": {
            "dataset_id": "ml_42_1000"
        },
        "params": {
            "model_type": "logreg"
        }
    }
    ```
    """
    try:

        if not dataset_store.exists(dataset_id):raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")

        # Récupérer le dataset
        df = dataset_store.get(dataset_id).df
        
        # Extraire paramètres
        model_type = model_type or "logreg"
        
        # Entraîner le modèle
        model_id, model_data = MlService.train(df, model_type)
        
        # Stocker le modèle
        store_model(model_id, model_data)
        
        # Préparer la réponse
        meta = MetaData(
            dataset_id=dataset_id,
            schema_version="1.0"
        )
        
        result = {
            "model_id": model_id,
            "model_type": model_type,
            "features": model_data["features"],
            "n_features": len(model_data["features"])
        }
        
        report = {
            "status": "training_completed",
            "metrics_train": model_data["metrics"]["train"],
            "metrics_test": model_data["metrics"]["test"]
        }
        
        return StandardResponse(
            meta=meta,
            result=result,
            report=report
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/metrics/{model_id}", response_model=StandardResponse)
def ml_metrics(model_id: str):
    """
    Récupère les métriques d'un modèle entraîné
    
    **Paramètre** : model_id (dans l'URL)
    
    **Retour** :
    - Accuracy, Precision, Recall, F1-Score, AUC
    - Matrice de confusion
    - Pour train et test
    
    **Exemple** : `GET /ml/metrics/model_logreg_20260210_143022`
    """
    try:
        # Récupérer le modèle
        model_data = get_model(model_id)
        
        # Préparer la réponse
        meta = MetaData(
            dataset_id="N/A",
            schema_version="1.0"
        )
        
        return StandardResponse(
            meta=meta,
            result={
                "model_id": model_id,
                "model_type": model_data["model_type"]
            },
            report={
                "metrics_train": model_data["metrics"]["train"],
                "metrics_test": model_data["metrics"]["test"]
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/predict", response_model=StandardResponse)
def ml_predict(request: MlPredictRequest):
    """
    Fait des prédictions avec un modèle entraîné
    
    **Paramètres** :
    - `model_id` : ID du modèle
    - `data` : Données à prédire (sans target)
    
    **Retour** :
    - Prédictions (0 ou 1)
    - Probabilités (si disponible)
    
    **Exemple** :
    ```json
    {
        "meta": {
            "dataset_id": "ml_42_1000"
        },
        "data": [
            {"x1": 100, "x2": 50, "x3": 30, "segment": "A"},
            {"x1": 120, "x2": 55, "x3": 35, "segment": "B"}
        ],
        "params": {
            "model_id": "model_logreg_20260210_143022"
        }
    }
    ```
    """
    try:
        # Extraire paramètres
        model_id = request.params.get("model_id")
        
        if not model_id:
            raise ValueError("model_id manquant dans params")
        
        # Récupérer le modèle
        model_data = get_model(model_id)
        
        # Convertir data en DataFrame
        X = pd.DataFrame(request.data)
        
        # Faire les prédictions
        predictions_result = MlService.predict(model_data, X)
        
        # Préparer la réponse
        meta = MetaData(
            dataset_id=request.meta.dataset_id,
            schema_version="1.0"
        )
        
        return StandardResponse(
            meta=meta,
            result=predictions_result,
            report={"status": "predictions_completed"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/model-info/{model_id}", response_model=StandardResponse)
def ml_model_info(model_id: str):
    """
    Récupère les informations sur un modèle
    
    **Paramètre** : model_id (dans l'URL)
    
    **Retour** :
    - Type de modèle
    - Hyperparamètres
    - Features utilisées
    - Date de création
    
    **Exemple** : `GET /ml/model-info/model_logreg_20260210_143022`
    """
    try:
        # Récupérer le modèle
        model_data = get_model(model_id)
        
        # Extraire les infos
        model_info = MlService.get_model_info(model_data)
        
        # Préparer la réponse
        meta = MetaData(
            dataset_id="N/A",
            schema_version="1.0"
        )
        
        return StandardResponse(
            meta=meta,
            result=model_info,
            report={"status": "info_retrieved"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
