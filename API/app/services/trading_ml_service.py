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

    # ── OPTIMIZATION & INTERPRETABILITY ──────────────────────────────────────

    def _optimize_model(self, X_train: np.ndarray, y_train: np.ndarray, model_type: str) -> Any:
        """Optimise les hyperparamètres avec Grid Search et TimeSeriesSplit (respect temporalité)."""
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

        # TimeSeriesSplit pour ne pas mélanger le futur et le passé
        # 3 splits : train augmente progressivement, val est le segment suivant
        tscv = TimeSeriesSplit(n_splits=3)

        if model_type == "rf":
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [4, 8, 12],
                "min_samples_leaf": [20, 50, 100],
                "class_weight": ["balanced", "balanced_subsample"],
            }
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        elif model_type == "gbm":
            param_grid = {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 4, 5],
                "subsample": [0.8, 1.0],
            }
            base_model = GradientBoostingClassifier(random_state=42)
        else:
            return self._build_model(model_type)  # Pas d'opti pour logreg ou autres

        print(f"🔍 Grid Search ({model_type}) sur {len(X_train)} samples...")
        grid = GridSearchCV(base_model, param_grid, cv=tscv, scoring="f1", n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        
        print(f"✅ Best params: {grid.best_params_} (Score: {grid.best_score_:.4f})")
        return grid.best_estimator_

    def _explain_decision(self, model: any, X_sample: pd.DataFrame, signal: str) -> str:
        """Génère une explication en langage naturel pour une prédiction donnée."""
        if signal == "HOLD":
            return "No strong signal detected based on current market conditions."

        explanation = []
        
        # Récupérer les feature importances si dispo
        importances = {}
        if hasattr(model, "feature_importances_"):
            importances = dict(zip(self.FEATURE_COLS, model.feature_importances_))
        
        # Analyser les features clés (top influence)
        # On regarde juste les valeurs brutes pour créer une narration simple
        
        # 1. Tendance (EMA)
        ema_diff = X_sample.get("ema_diff", 0)
        dist_ema200 = X_sample.get("distance_to_ema200", 0)
        
        if signal == "BUY":
            if dist_ema200 > 0:
                explanation.append("price is above the long-term trend (EMA200)")
            if ema_diff > 0:
                explanation.append("short-term momentum is positive")
        elif signal == "SELL":
            if dist_ema200 < 0:
                explanation.append("price is below the long-term trend (EMA200)")
            if ema_diff < 0:
                explanation.append("short-term momentum is negative")

        # 2. RSI (V2 feature)
        rsi = X_sample.get("rsi_14", 50)
        if rsi < 30 and signal == "BUY":
            explanation.append("market is potentially oversold (RSI < 30)")
        elif rsi > 70 and signal == "SELL":
            explanation.append("market is potentially overbought (RSI > 70)")

        # 3. Volatilité
        vol = X_sample.get("volatility_ratio", 1.0)
        if vol > 1.5:
            explanation.append("volatility is expanding")
        elif vol < 0.8:
            explanation.append("market is consolidating")

        if not explanation:
            return f"Model detects complex pattern favoring {signal}."
            
        return f"Model suggests {signal} because " + ", ".join(explanation) + "."

    # ── MODIFIED TRAIN ───────────────────────────────────────────────────────

    def train(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: Optional[pd.DataFrame],
        model_type: str = "rf",
        optimize: bool = False,
    ) -> Dict[str, Any]:
        """
        Entraîne un modèle avec split temporel strict.
        Optionnel: optimize=True pour Grid Search.
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

        # ── Scaling ───────────────────────────────────────────────────────────
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test) if X_test is not None else None

        # ── Modèle (Optimisation ou Défaut) ───────────────────────────────────
        if optimize:
            # On combine Train + Val pour le Grid Search car TimeSeriesSplit va créer les folds
            # Mais attention, pour respecter la logique stricte "Train 2022", "Val 2023",
            # l'optimisation doit se faire SUR 2022 UNIQUEMENT (avec CV interne), 
            # OU sur 2022+2023 avec CV qui respecte le temps.
            # Ici on optimise sur le train set (2022) pour trouver les hyperparamètres,
            # puis on valide une dernière fois sur 2023.
            model = self._optimize_model(X_train_s, y_train, model_type)
        else:
            model = self._build_model(model_type)
            model.fit(X_train_s, y_train)

        # ── Le reste est identique (Prédictions, Métriques...) ────────────────
        def _predict(X_s, model):
            y_pred = model.predict(X_s)
            try:
                y_proba = model.predict_proba(X_s)[:, 1]
            except Exception:
                y_proba = None
            return y_pred, y_proba

        y_train_pred, y_train_proba = _predict(X_train_s, model)
        y_val_pred, y_val_proba = _predict(X_val_s, model)

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

        # ── Construire model_data ─────────────────────────────────────────────
        model_id = f"trading_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        model_data = {
            "model_id": model_id,
            "model_type": model_type,
            "version": self.VERSION_MAP[model_type] + ("_optimized" if optimize else ""),
            "created_at": datetime.now().isoformat(),
            "features": self.FEATURE_COLS,
            "hyperparams": model.get_params(),
            "metrics": metrics,
            "feature_importance": feature_importance,
            "_model_object": model,
            "_scaler": scaler,
        }

        _register_model(model_data)
        return self._serializable(model_data)

    def predict_latest(self, model_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Prédit le signal pour la DERNIÈRE bougie disponible + EXPLICATION."""
        model_data = get_model(model_id)
        model = model_data["_model_object"]
        scaler = model_data["_scaler"]

        # Prendre la dernière ligne avec features
        X = df[self.FEATURE_COLS].copy() # Pas de dropna, on veut la dernière
        if X.empty: raise ValueError("DataFrame vide.")
        
        last_row = X.iloc[[-1]] # Garder DataFrame
        if last_row.isna().any().any():
            raise ValueError("Features manquantes pour la dernière bougie (calcul features incomplet).")

        X_s = scaler.transform(last_row)
        y_pred = model.predict(X_s)[0]
        try:
            y_proba = model.predict_proba(X_s)[0, 1]
        except:
            y_proba = None
            
        signal = "BUY" if y_pred == 1 else "SELL"
        
        # Générer explication
        # On doit passer une Series ou Dict avec les noms de colonnes pour l'analyse
        explanation = self._explain_decision(model, last_row.iloc[0], signal)

        return {
            "model_id": model_id,
            "signal": signal,
            "probability_buy": y_proba,
            "explanation": explanation,
            "model_version": model_data["version"],
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

