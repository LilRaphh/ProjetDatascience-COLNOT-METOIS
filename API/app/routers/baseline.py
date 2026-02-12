"""
Router Baseline – /baseline
Stratégies de référence : Random, Buy & Hold, Règles fixes.
"""

from __future__ import annotations

import traceback
from fastapi import APIRouter, HTTPException, Query

from app.repositories.dataset_store import dataset_store
from app.services.baseline_service import BaselineService

router = APIRouter(prefix="/baseline", tags=["Baseline Strategies"])
baseline_service = BaselineService()


@router.get("/compare/{dataset_id}")
def compare_all_baselines(
    dataset_id: str,
    seed: int = Query(42, description="Seed pour la stratégie aléatoire"),
):
    """
    Compare les 3 stratégies baseline sur un dataset M15 :
    - **random** : signaux aléatoires (BUY/SELL/HOLD)
    - **buy_and_hold** : position long permanente
    - **fixed_rules** : croisement EMA20/EMA50 filtré RSI

    Métriques : total_return_pct, max_drawdown_pct, sharpe, profit_factor, n_trades
    """
    try:
        if not dataset_store.exists(dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")
        df = dataset_store.get(dataset_id).df
        results = baseline_service.compare_all(df, seed=seed)
        return {"dataset_id": dataset_id, "baselines": results}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Baseline compare failed: {repr(e)}")


@router.get("/random/{dataset_id}")
def random_strategy(
    dataset_id: str,
    seed: int = Query(42),
    hold_prob: float = Query(0.5, ge=0.0, le=1.0),
):
    """Stratégie aléatoire avec probabilité de HOLD paramétrable."""
    try:
        if not dataset_store.exists(dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")
        df = dataset_store.get(dataset_id).df
        _, metrics = baseline_service.random_strategy(df, seed=seed, hold_prob=hold_prob)
        return {"dataset_id": dataset_id, "strategy": "random", "metrics": metrics}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))


@router.get("/buy_and_hold/{dataset_id}")
def buy_and_hold(dataset_id: str):
    """Buy & Hold : position longue permanente avec coûts de transaction."""
    try:
        if not dataset_store.exists(dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")
        df = dataset_store.get(dataset_id).df
        _, metrics = baseline_service.buy_and_hold(df)
        return {"dataset_id": dataset_id, "strategy": "buy_and_hold", "metrics": metrics}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))


@router.get("/fixed_rules/{dataset_id}")
def fixed_rules(
    dataset_id: str,
    ema_short: int = Query(20, ge=2, le=100),
    ema_long: int = Query(50, ge=5, le=300),
    rsi_period: int = Query(14, ge=2, le=50),
    rsi_overbought: float = Query(70.0, ge=50.0, le=90.0),
    rsi_oversold: float = Query(30.0, ge=10.0, le=50.0),
):
    """
    Stratégie règles fixes paramétrable :
    BUY si EMA_court > EMA_long ET RSI < overbought
    SELL si EMA_court < EMA_long ET RSI > oversold
    """
    try:
        if not dataset_store.exists(dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")
        df = dataset_store.get(dataset_id).df
        _, metrics = baseline_service.fixed_rules_strategy(
            df,
            ema_short=ema_short,
            ema_long=ema_long,
            rsi_period=rsi_period,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
        )
        return {"dataset_id": dataset_id, "strategy": "fixed_rules", "metrics": metrics}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))
