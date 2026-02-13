"""
Router ML Trading – /trading_ml
Entraînement avec split temporel strict 2022/2023/2024.
Versioning : v1=logreg, v2=rf, v3=gbm.
Seul le meilleur modèle est exposé pour la prédiction en production.
"""

from __future__ import annotations

import traceback
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.schemas.dataset import MLTrainParams
from app.repositories.dataset_store import dataset_store
from app.services.trading_ml_service import (
    TradingMLService,
    get_best_model,
    list_models,
    get_model,
)

router = APIRouter(prefix="/trading_ml", tags=["ML Trading (Split Temporel)"])
ml_service = TradingMLService()


@router.post("/train")
def train_model(request: MLTrainParams):
    """
    Entraîne un modèle ML pour le trading GBP/USD.

    **Split temporel strict obligatoire** :
    - `dataset_train_id` = 2022 (entraînement)
    - `dataset_val_id`   = 2023 (validation / sélection modèle)
    - `dataset_test_id`  = 2024 (test final, jamais utilisé pour entraîner)

    **Modèles disponibles** :
    - `logreg` → version v1
    - `rf`     → version v2 (recommandé)
    - `gbm`    → version v3

    **Métriques retournées** : stat (accuracy, F1, AUC) + financières (Sharpe, MDD, profit factor)
    """
    try:
        dataset_train_id = request.dataset_train_id
        dataset_val_id = request.dataset_val_id
        dataset_test_id = request.dataset_test_id
        model_type = request.model_type
        optimize = request.optimize
        
        for did in [dataset_train_id, dataset_val_id]:
            if not dataset_store.exists(did):
                raise HTTPException(status_code=404, detail=f"Dataset introuvable: {did}")

        if dataset_test_id and not dataset_store.exists(dataset_test_id):
            raise HTTPException(status_code=404, detail=f"Dataset test introuvable: {dataset_test_id}")

        df_train = dataset_store.get(dataset_train_id).df
        df_val = dataset_store.get(dataset_val_id).df
        df_test = dataset_store.get(dataset_test_id).df if dataset_test_id else None

        result = ml_service.train(df_train, df_val, df_test, model_type=model_type, optimize=optimize)
        return result

    except HTTPException:
        raise
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"ML training failed: {repr(e)}")


@router.get("/models")
def list_all_models():
    """
    Liste tous les modèles entraînés avec leur version, métriques, et statut 'best'.
    """
    return {"models": list_models()}


@router.get("/best_model")
def get_best():
    """
    Retourne le meilleur modèle actuel (Sharpe validation 2023 le plus élevé).
    C'est CE modèle qui est utilisé par l'endpoint /predict.
    """
    best = get_best_model()
    if best is None:
        raise HTTPException(
            status_code=404,
            detail="Aucun modèle entraîné. Appeler POST /trading_ml/train d'abord."
        )
    return best


@router.get("/model/{model_id}")
def get_model_info(model_id: str):
    """Informations complètes sur un modèle (métriques, features, hyperparams)."""
    try:
        data = get_model(model_id)
        # Retourner sans objets sklearn
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Modèle introuvable: {model_id}")


@router.post("/predict/{dataset_id}")
def predict(
    dataset_id: str,
    model_id: Optional[str] = Query(None, description="ID du modèle (défaut: meilleur modèle)"),
):
    """
    Prédit les signaux BUY/SELL sur un dataset M15 avec features.

    Si `model_id` n'est pas précisé, utilise le **meilleur modèle** entraîné.
    """
    try:
        if not dataset_store.exists(dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")

        df = dataset_store.get(dataset_id).df

        if model_id is None:
            best = get_best_model()
            if best is None:
                raise HTTPException(status_code=404, detail="Aucun modèle entraîné.")
            model_id = best["model_id"]

        result = ml_service.predict(model_id, df)
        return result

    except HTTPException:
        raise
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {repr(e)}")


@router.get("/predict_latest/{dataset_id}")
def predict_latest(
    dataset_id: str,
    model_id: Optional[str] = Query(None, description="ID du modèle (défaut: meilleur modèle)"),
):
    """
    Retourne uniquement le signal de la DERNIÈRE bougie M15 disponible.
    Utile pour un usage en production en temps réel.
    """
    try:
        if not dataset_store.exists(dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")

        df = dataset_store.get(dataset_id).df

        if model_id is None:
            best = get_best_model()
            if best is None:
                raise HTTPException(status_code=404, detail="Aucun modèle entraîné.")
            model_id = best["model_id"]

        result = ml_service.predict_latest(model_id, df)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Latest predict failed: {repr(e)}")
