"""
Service ML Trading – GBP/USD M15
Split temporel strict : 2022 train / 2023 val / 2024 test
Modèles : LogisticRegression, RandomForest, GradientBoosting
Métriques : stat + financières (Sharpe, MDD, profit factor)
Versioning : v1 (logreg), v2 (rf), v3 (gbm)
"""

from __future__ import annotations

import uuid
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from app.services.feature_service import FeatureService

MODELS_DIR = Path("models")
TRANSACTION_COST = 0.0002


# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}
_BEST_MODEL_ID: Optional[str] = None


def _register_model(model_data: Dict[str, Any]) -> None:
    """Stocke le modèle et met à jour le meilleur si Sharpe amélioré."""
    global _BEST_MODEL_ID
    mid = model_data["model_id"]
    _MODEL_REGISTRY[mid] = model_data

    # Désigner le meilleur modèle (basé sur Sharpe sur validation 2023)
    val_sharpe = model_data.get("metrics", {}).get("val", {}).get("sharpe", -999)
    if _BEST_MODEL_ID is None:
        _BEST_MODEL_ID = mid
    else:
        best_sharpe = (
            _MODEL_REGISTRY[_BEST_MODEL_ID]
            .get("metrics", {}).get("val", {}).get("sharpe", -999)
        )
        if val_sharpe > best_sharpe:
            _BEST_MODEL_ID = mid


def get_model(model_id: str) -> Dict[str, Any]:
    if model_id not in _MODEL_REGISTRY:
        raise KeyError(f"Modèle introuvable: {model_id}")
    return _MODEL_REGISTRY[model_id]


def get_best_model() -> Optional[Dict[str, Any]]:
    if _BEST_MODEL_ID and _BEST_MODEL_ID in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[_BEST_MODEL_ID]
    return None


def list_models() -> List[Dict[str, Any]]:
    return [
        {
            "model_id": v["model_id"],
            "model_type": v["model_type"],
            "version": v["version"],
            "created_at": v["created_at"],
            "is_best": v["model_id"] == _BEST_MODEL_ID,
            "val_sharpe": v.get("metrics", {}).get("val", {}).get("sharpe"),
            "test_sharpe": v.get("metrics", {}).get("test", {}).get("sharpe"),
        }
        for v in _MODEL_REGISTRY.values()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ON PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _financial_backtest(
    close: pd.Series,
    predictions: np.ndarray,
    cost: float = TRANSACTION_COST,
) -> Dict[str, float]:
    """
    Calcule les métriques financières à partir des prédictions ML.
    1 → BUY (long), 0 → SELL (short)
    """
    signals = np.where(predictions == 1, 1, -1)
    n = len(close)
    pnl = np.zeros(n)
    position = 0
    trades = 0

    for i in range(1, n):
        sig = int(signals[i - 1])
        if sig != position:
            pnl[i] -= cost
            trades += 1
            position = sig
        ret = (close.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1]
        pnl[i] += position * ret

    equity = np.cumprod(1 + pnl)
    total_return = float(equity[-1] - 1)

    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = float(drawdown.min())

    pnl_s = pd.Series(pnl[1:])
    sharpe = (
        float(pnl_s.mean() / pnl_s.std() * np.sqrt(6552))
        if pnl_s.std() > 0 else 0.0
    )

    wins = pnl_s[pnl_s > 0].sum()
    losses = pnl_s[pnl_s < 0].abs().sum()
    profit_factor = float(wins / losses) if losses > 0 else float("inf")

    return {
        "total_return_pct": round(total_return * 100, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "sharpe": round(sharpe, 4),
        "profit_factor": round(profit_factor, 4),
        "n_trades": int(trades),
        "final_equity": round(float(equity[-1]), 6),
        "equity_curve": [round(float(v), 6) for v in equity[::10]],
    }


def _stat_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
        except Exception:
            metrics["roc_auc"] = None
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    } if cm.shape == (2, 2) else {}
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# TRADING ML SERVICE
# ─────────────────────────────────────────────────────────────────────────────

class TradingMLService:
    """
    Entraînement + évaluation ML pour le système de trading GBP/USD.
    Split temporel strict : 2022 train / 2023 val / 2024 test.
    """

    FEATURE_COLS = FeatureService.ALL_FEATURES
    TARGET_COL = "target"
    VERSION_MAP = {
        "logreg": "v1",
        "rf": "v2",
        "gbm": "v3",
    }

    def train(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: Optional[pd.DataFrame],
        model_type: str = "rf",
    ) -> Dict[str, Any]:
        """
        Entraîne un modèle avec split temporel strict.

        Parameters
        ----------
        df_train : données 2022 (avec features + target)
        df_val   : données 2023 (validation)
        df_test  : données 2024 (test final – utilisé uniquement en évaluation)
        model_type : logreg | rf | gbm
        """
        if model_type not in self.VERSION_MAP:
            raise ValueError(f"model_type inconnu: {model_type}. Choix: {list(self.VERSION_MAP)}")

        # ── Préparer les datasets ──────────────────────────────────────────────
        X_train, y_train, close_train = self._prepare(df_train)
        X_val, y_val, close_val = self._prepare(df_val)
        X_test, y_test, close_test = (
            self._prepare(df_test) if df_test is not None else (None, None, None)
        )

        if len(X_train) == 0:
            raise ValueError("Dataset train vide après sélection des features.")
        if len(X_val) == 0:
            raise ValueError("Dataset val vide après sélection des features.")

        # ── Scaling ───────────────────────────────────────────────────────────
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test) if X_test is not None else None

        # ── Modèle ────────────────────────────────────────────────────────────
        model = self._build_model(model_type)
        model.fit(X_train_s, y_train)

        # ── Prédictions ───────────────────────────────────────────────────────
        def _predict(X_s, model):
            y_pred = model.predict(X_s)
            try:
                y_proba = model.predict_proba(X_s)[:, 1]
            except Exception:
                y_proba = None
            return y_pred, y_proba

        y_train_pred, y_train_proba = _predict(X_train_s, model)
        y_val_pred, y_val_proba = _predict(X_val_s, model)

        # ── Métriques ─────────────────────────────────────────────────────────
        metrics = {
            "train": {
                **_stat_metrics(y_train, y_train_pred, y_train_proba),
                **_financial_backtest(close_train, y_train_pred),
            },
            "val": {
                **_stat_metrics(y_val, y_val_pred, y_val_proba),
                **_financial_backtest(close_val, y_val_pred),
            },
        }

        if X_test_s is not None and y_test is not None:
            y_test_pred, y_test_proba = _predict(X_test_s, model)
            metrics["test"] = {
                **_stat_metrics(y_test, y_test_pred, y_test_proba),
                **_financial_backtest(close_test, y_test_pred),
            }

        # ── Feature Importance ────────────────────────────────────────────────
        feature_importance: Optional[Dict[str, float]] = None
        if hasattr(model, "feature_importances_"):
            feature_importance = {
                feat: round(float(imp), 6)
                for feat, imp in zip(self.FEATURE_COLS, model.feature_importances_)
            }
        elif hasattr(model, "coef_"):
            feature_importance = {
                feat: round(float(coef), 6)
                for feat, coef in zip(self.FEATURE_COLS, model.coef_[0])
            }

        # ── Construire model_data ─────────────────────────────────────────────
        model_id = f"trading_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        model_data = {
            "model_id": model_id,
            "model_type": model_type,
            "version": self.VERSION_MAP[model_type],
            "created_at": datetime.now().isoformat(),
            "features": self.FEATURE_COLS,
            "n_features": len(self.FEATURE_COLS),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)) if X_test is not None else 0,
            "hyperparams": model.get_params(),
            "metrics": metrics,
            "feature_importance": feature_importance,
            # Objets sklearn (non sérialisables JSON → stockés en mémoire)
            "_model_object": model,
            "_scaler": scaler,
        }

        _register_model(model_data)

        # Retourner une version sérialisable
        return self._serializable(model_data)

    def predict(
        self,
        model_id: str,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Prédit les signaux sur un DataFrame M15 avec features déjà calculées.
        """
        model_data = get_model(model_id)
        model = model_data["_model_object"]
        scaler = model_data["_scaler"]

        X = df[self.FEATURE_COLS].copy()
        X = X.dropna()
        if len(X) == 0:
            raise ValueError("Aucune ligne valide après sélection des features.")

        X_s = scaler.transform(X)
        y_pred = model.predict(X_s)

        try:
            y_proba = model.predict_proba(X_s)[:, 1].tolist()
        except Exception:
            y_proba = None

        signals = ["BUY" if p == 1 else "SELL" for p in y_pred]

        return {
            "model_id": model_id,
            "n_predictions": int(len(y_pred)),
            "signals": signals,
            "predictions_binary": y_pred.tolist(),
            "probabilities": y_proba,
        }

    def predict_latest(self, model_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Prédit le signal pour la DERNIÈRE bougie disponible."""
        result = self.predict(model_id, df)
        last_signal = result["signals"][-1] if result["signals"] else "HOLD"
        last_proba = (
            result["probabilities"][-1]
            if result["probabilities"] is not None
            else None
        )
        return {
            "model_id": model_id,
            "signal": last_signal,
            "probability_buy": last_proba,
            "model_version": get_model(model_id)["version"],
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _prepare(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
        feat_cols = [c for c in self.FEATURE_COLS if c in df.columns]
        valid = df[feat_cols + [self.TARGET_COL]].dropna()
        X = valid[feat_cols].reset_index(drop=True)
        y = valid[self.TARGET_COL].astype(int).values

        close_col = "close_15m" if "close_15m" in df.columns else "close"
        close = df[close_col].dropna().reset_index(drop=True)
        # Aligner close sur les mêmes indices valides
        close_aligned = df.loc[valid.index, close_col].reset_index(drop=True)

        return X, y, close_aligned

    @staticmethod
    def _build_model(model_type: str):
        if model_type == "logreg":
            return LogisticRegression(
                C=0.1, max_iter=1000, random_state=42, class_weight="balanced"
            )
        elif model_type == "rf":
            return RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=50,
                random_state=42, class_weight="balanced", n_jobs=-1,
            )
        elif model_type == "gbm":
            return GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
        raise ValueError(f"Modèle inconnu: {model_type}")

    @staticmethod
    def _serializable(model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Retourne une copie sans les objets sklearn."""
        return {k: v for k, v in model_data.items() if not k.startswith("_")}
