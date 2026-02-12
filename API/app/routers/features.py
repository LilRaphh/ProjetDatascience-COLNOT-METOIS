"""
Router Feature Engineering V2 – /features
"""

from __future__ import annotations

import traceback
from fastapi import APIRouter, HTTPException

from app.schemas.features import ComputeFeaturesParams
from app.repositories.dataset_store import dataset_store
from app.services.feature_service import FeatureService

router = APIRouter(prefix="/features", tags=["Feature Engineering V2"])
feature_service = FeatureService()


@router.post("/compute")
def compute_features(request: ComputeFeaturesParams):
    """
    Calcule les features V2 sur un dataset M15 nettoyé.

    **Prérequis** : appeler POST /m15/aggregate puis POST /m15/clean
    sur votre dataset M1 avant d'appeler cet endpoint.

    **Features calculées** :
    - Court terme : return_1, return_4, ema_20, ema_50, ema_diff, rsi_14,
      rolling_std_20, range_15m, body, upper_wick, lower_wick
    - Contexte & Régime : ema_200, distance_to_ema200, slope_ema50,
      atr_14, rolling_std_100, volatility_ratio, adx_14, macd, macd_signal

    **Target** (si add_target=True) : y = 1 si close_{t+1} > close_t, sinon 0
    """
    try:
        dataset_id = request.dataset_id
        drop_na = request.drop_na
        add_target = request.add_target
        
        if not dataset_store.exists(dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")

        entry = dataset_store.get(dataset_id)
        df = entry.df

        df_feat, report = feature_service.compute(df, drop_na=drop_na)

        if add_target:
            df_feat = feature_service.compute_target(df_feat)
            report["target_added"] = True
            report["n_valid_targets"] = int(df_feat["target"].notna().sum())

        new_id = f"{dataset_id}_features"
        dataset_store.put(
            dataset_id=new_id,
            df=df_feat,
            meta={
                **(entry.meta or {}),
                "phase": "features",
                "feature_report": report,
            },
        )

        return {
            "dataset_id": new_id,
            "n_rows": int(len(df_feat)),
            "n_features": report["n_features"],
            "features": report["features_computed"],
            "report": report,
            "columns": list(df_feat.columns),
        }

    except HTTPException:
        raise
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Feature computation failed: {repr(e)}")


@router.get("/list")
def list_features():
    """Liste toutes les features disponibles dans le Feature Pack V2."""
    return {
        "short_term_features": FeatureService.SHORT_TERM_FEATURES,
        "regime_features": FeatureService.REGIME_FEATURES,
        "all_features": FeatureService.ALL_FEATURES,
        "n_total": len(FeatureService.ALL_FEATURES),
        "warmup_bars_required": FeatureService.WARMUP_BARS,
        "target": "y = 1 si close_{t+1} > close_t, sinon 0",
    }


@router.get("/info/{dataset_id}")
def features_info(dataset_id: str):
    """Résumé statistique des features d'un dataset."""
    if not dataset_store.exists(dataset_id):
        raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")

    entry = dataset_store.get(dataset_id)
    df = entry.df

    feat_cols = [c for c in FeatureService.ALL_FEATURES if c in df.columns]
    if not feat_cols:
        raise HTTPException(
            status_code=400,
            detail="Aucune feature V2 trouvée. Appeler POST /features/compute d'abord.",
        )

    stats = df[feat_cols].describe().round(6).to_dict()
    missing = {c: int(df[c].isna().sum()) for c in feat_cols}

    return {
        "dataset_id": dataset_id,
        "n_rows": int(len(df)),
        "features_present": feat_cols,
        "n_features_present": len(feat_cols),
        "missing_values": missing,
        "statistics": stats,
    }
