"""
Router Évaluation Finale – /evaluate
Compare Random / Règles / ML / RL sur 2024.
Métriques obligatoires + stress tests trimestriels.
"""

from __future__ import annotations

import traceback
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.repositories.dataset_store import dataset_store
from app.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/evaluate", tags=["Évaluation Finale"])
eval_service = EvaluationService()


@router.get("/compare/{dataset_id}")
def compare_all(
    dataset_id: str,
    ml_model_id: Optional[str] = Query(None, description="ID modèle ML (défaut: meilleur)"),
    rl_model_id: Optional[str] = Query(None, description="ID modèle RL (défaut: meilleur)"),
    seed: int = Query(42),
):
    """
    Comparaison finale de toutes les stratégies :
    - Random
    - Buy & Hold
    - Règles fixes
    - ML (meilleur modèle)
    - RL (meilleur agent)

    **Classement par Sharpe ratio.**
    Un modèle est valide uniquement s'il est robuste sur 2024.
    """
    try:
        if not dataset_store.exists(dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")
        df = dataset_store.get(dataset_id).df
        result = eval_service.compare_all(
            df,
            ml_model_id=ml_model_id,
            rl_model_id=rl_model_id,
            seed=seed,
        )
        return {"dataset_id": dataset_id, **result}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {repr(e)}")


@router.get("/stress_test/{dataset_id}")
def stress_test(
    dataset_id: str,
    ml_model_id: Optional[str] = Query(None),
    seed: int = Query(42),
):
    """
    Stress test trimestriel sur 2024 (Q1, Q2, Q3, Q4).
    Identifie les sous-périodes où les stratégies sont défaillantes.
    Requiert la colonne 'timestamp' dans le dataset.
    """
    try:
        if not dataset_store.exists(dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")
        df = dataset_store.get(dataset_id).df
        result = eval_service.stress_test_quarterly(df, ml_model_id=ml_model_id, seed=seed)
        return {"dataset_id": dataset_id, **result}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Stress test failed: {repr(e)}")


@router.get("/robustness/{model_id}")
def model_robustness(model_id: str):
    """
    Rapport de robustesse d'un modèle ML :
    compare train (2022) / val (2023) / test (2024).
    Détecte l'overfitting temporel.
    """
    try:
        result = eval_service.model_robustness_report(model_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))
