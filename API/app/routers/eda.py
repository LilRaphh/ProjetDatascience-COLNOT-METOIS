"""
Router EDA – /eda
Analyse exploratoire obligatoire : rendements, volatilité, analyse horaire,
autocorrélation, test ADF.
"""

from __future__ import annotations

import traceback
from fastapi import APIRouter, HTTPException, Query

from app.repositories.dataset_store import dataset_store
from app.services.eda_service import EDAService

router = APIRouter(prefix="/eda", tags=["Analyse Exploratoire (EDA)"])
eda_service = EDAService()


@router.get("/full_report/{dataset_id}")
def full_eda_report(dataset_id: str):
    """
    Génère l'intégralité du rapport EDA :
    - Distribution des rendements (+ test normalité)
    - Volatilité dans le temps (rolling + mensuelle)
    - Analyse horaire (rendement, range, activité par heure)
    - Autocorrélation des rendements (lags 1..20)
    - Test ADF de stationnarité (prix + rendements)
    """
    try:
        if not dataset_store.exists(dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")

        df = dataset_store.get(dataset_id).df
        report = eda_service.full_report(df)
        return {"dataset_id": dataset_id, "eda_report": report}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"EDA failed: {repr(e)}")


@router.get("/returns/{dataset_id}")
def returns_analysis(dataset_id: str):
    """Distribution des rendements M15."""
    try:
        if not dataset_store.exists(dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")
        df = dataset_store.get(dataset_id).df
        return {"dataset_id": dataset_id, "returns": eda_service.analyse_returns(df)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))


@router.get("/volatility/{dataset_id}")
def volatility_analysis(
    dataset_id: str,
    window: int = Query(96, ge=10, le=500, description="Fenêtre rolling (bougies M15)"),
):
    """Volatilité dans le temps (rolling + mensuelle)."""
    try:
        if not dataset_store.exists(dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")
        df = dataset_store.get(dataset_id).df
        return {
            "dataset_id": dataset_id,
            "volatility": eda_service.analyse_volatility(df, window=window),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))


@router.get("/hourly/{dataset_id}")
def hourly_analysis(dataset_id: str):
    """Analyse par heure de la journée (rendement moyen, range, activité)."""
    try:
        if not dataset_store.exists(dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")
        df = dataset_store.get(dataset_id).df
        return {"dataset_id": dataset_id, "hourly": eda_service.analyse_hourly(df)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))


@router.get("/autocorrelation/{dataset_id}")
def autocorrelation_analysis(
    dataset_id: str,
    max_lags: int = Query(20, ge=1, le=100),
):
    """Autocorrélation des rendements (lags 1..max_lags) + bornes IC 95%."""
    try:
        if not dataset_store.exists(dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")
        df = dataset_store.get(dataset_id).df
        return {
            "dataset_id": dataset_id,
            "autocorrelation": eda_service.analyse_autocorrelation(df, max_lags=max_lags),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))


@router.get("/adf/{dataset_id}")
def adf_test(dataset_id: str):
    """Test ADF (Augmented Dickey-Fuller) de stationnarité sur prix et rendements."""
    try:
        if not dataset_store.exists(dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")
        df = dataset_store.get(dataset_id).df
        return {"dataset_id": dataset_id, "adf_test": eda_service.test_adf(df)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))
