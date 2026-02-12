"""
Router Reinforcement Learning – /rl
Q-Learning tabulaire sur GBP/USD M15.
Walk-forward : train 2022 / val 2023 / test 2024.
"""

from __future__ import annotations

import traceback
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.schemas.dataset import RLTrainParams
from app.repositories.dataset_store import dataset_store
from app.services.rl_service import RLService, _RL_MODELS, _BEST_RL_ID

router = APIRouter(prefix="/rl", tags=["Reinforcement Learning"])
rl_service = RLService()


@router.get("/design")
def rl_design_document():
    """
    Conception obligatoire sur papier (section 9.1 du projet) :
    Problème métier, State, Action, Reward, Environnement,
    Choix d'algorithme, Paramètres clés.
    """
    return rl_service.design_document()


@router.post("/train")
def train_rl(request: RLTrainParams):
    """
    Entraîne un agent Q-Learning pour le trading GBP/USD.

    **Algorithme** : Q-Learning tabulaire avec discrétisation de l'état.
    **State** : features M15 normalisées + position actuelle + PnL + drawdown.
    **Action** : 0=HOLD, 1=BUY (long), 2=SELL (short).
    **Reward** : PnL step – coûts de transaction – pénalité drawdown excessif.

    **n_episodes** : recommandé ≥ 10 (plus = meilleure convergence, ≤ 30 pour rapidité).
    """
    try:
        dataset_train_id = request.dataset_train_id
        dataset_val_id = request.dataset_val_id
        dataset_test_id = request.dataset_test_id
        n_episodes = request.n_episodes
        seed = request.seed
        
        for did in [dataset_train_id, dataset_val_id]:
            if not dataset_store.exists(did):
                raise HTTPException(status_code=404, detail=f"Dataset introuvable: {did}")

        if dataset_test_id and not dataset_store.exists(dataset_test_id):
            raise HTTPException(status_code=404, detail=f"Dataset test introuvable: {dataset_test_id}")

        df_train = dataset_store.get(dataset_train_id).df
        df_val = dataset_store.get(dataset_val_id).df
        df_test = dataset_store.get(dataset_test_id).df if dataset_test_id else None

        result = rl_service.train(
            df_train, df_val, df_test,
            n_episodes=n_episodes,
            seed=seed,
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"RL training failed: {repr(e)}")


@router.get("/models")
def list_rl_models():
    """Liste tous les agents RL entraînés."""
    return {"rl_models": rl_service.list_models()}


@router.get("/best_model")
def get_best_rl():
    """Retourne le meilleur agent RL (Sharpe validation 2023 le plus élevé)."""
    if not _BEST_RL_ID or _BEST_RL_ID not in _RL_MODELS:
        raise HTTPException(
            status_code=404,
            detail="Aucun modèle RL entraîné. Appeler POST /rl/train d'abord."
        )
    data = _RL_MODELS[_BEST_RL_ID]
    return {k: v for k, v in data.items() if not k.startswith("_")}


@router.post("/evaluate/{dataset_id}")
def evaluate_rl(
    dataset_id: str,
    model_id: Optional[str] = Query(None, description="ID du modèle RL (défaut: meilleur)"),
):
    """
    Évalue un agent RL sur un dataset (greedy, pas d'exploration).
    Retourne les métriques financières : Sharpe, MDD, profit factor, courbe d'equity.
    """
    try:
        if not dataset_store.exists(dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")

        df = dataset_store.get(dataset_id).df

        rl_id = model_id or _BEST_RL_ID
        if not rl_id or rl_id not in _RL_MODELS:
            raise HTTPException(
                status_code=404,
                detail="Aucun modèle RL. Appeler POST /rl/train d'abord."
            )

        agent = _RL_MODELS[rl_id]["_agent"]
        metrics = rl_service._evaluate(agent, df)
        return {"dataset_id": dataset_id, "model_id": rl_id, "metrics": metrics}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"RL evaluation failed: {repr(e)}")
