"""
Service Évaluation Finale
Compare : Random / Règles / ML / RL
Métriques : profit cumulé, MDD, Sharpe, profit factor
Un modèle est valide uniquement s'il est robuste sur 2024.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from app.services.baseline_service import BaselineService, _backtest
from app.services.trading_ml_service import TradingMLService, get_best_model, list_models, get_model
from app.services.rl_service import RLService, _RL_MODELS, _BEST_RL_ID


class EvaluationService:
    """
    Évaluation finale et comparaison de toutes les stratégies.
    """

    def __init__(self):
        self.baseline = BaselineService()
        self.ml_service = TradingMLService()
        self.rl_service = RLService()

    def compare_all(
        self,
        df_test: pd.DataFrame,
        ml_model_id: Optional[str] = None,
        rl_model_id: Optional[str] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Lance une comparaison complète de toutes les stratégies sur df_test.
        df_test doit contenir les features + close_15m.
        """
        results: Dict[str, Any] = {}

        # ── 1. Random ─────────────────────────────────────────────────────────
        _, random_m = self.baseline.random_strategy(df_test, seed=seed)
        results["random"] = random_m

        # ── 2. Buy & Hold ─────────────────────────────────────────────────────
        _, bah_m = self.baseline.buy_and_hold(df_test)
        results["buy_and_hold"] = bah_m

        # ── 3. Règles fixes ───────────────────────────────────────────────────
        _, rules_m = self.baseline.fixed_rules_strategy(df_test)
        results["fixed_rules"] = rules_m

        # ── 4. ML (meilleur modèle par défaut) ────────────────────────────────
        if ml_model_id:
            try:
                ml_result = self.ml_service.predict(ml_model_id, df_test)
                preds = np.array(ml_result["predictions_binary"])
                close = self._get_close(df_test)
                sig_mapped = pd.Series(
                    np.where(preds == 1, 1, -1)[:len(close)]
                )
                ml_metrics = _backtest(close, sig_mapped)
                ml_metrics["strategy"] = "ml"
                ml_metrics["model_id"] = ml_model_id
                results["ml"] = ml_metrics
            except Exception as e:
                results["ml"] = {"error": str(e)}
        else:
            best = get_best_model()
            if best:
                try:
                    ml_result = self.ml_service.predict(best["model_id"], df_test)
                    preds = np.array(ml_result["predictions_binary"])
                    close = self._get_close(df_test)
                    sig_mapped = pd.Series(np.where(preds == 1, 1, -1)[:len(close)])
                    ml_metrics = _backtest(close, sig_mapped)
                    ml_metrics["strategy"] = "ml"
                    ml_metrics["model_id"] = best["model_id"]
                    ml_metrics["model_version"] = best.get("version")
                    results["ml"] = ml_metrics
                except Exception as e:
                    results["ml"] = {"error": str(e)}
            else:
                results["ml"] = {"error": "Aucun modèle ML entraîné. Appeler POST /trading_ml/train d'abord."}

        # ── 5. RL (meilleur agent) ─────────────────────────────────────────────
        rl_id = rl_model_id or _BEST_RL_ID
        if rl_id and rl_id in _RL_MODELS:
            try:
                rl_data = _RL_MODELS[rl_id]
                agent = rl_data["_agent"]
                rl_metrics = self.rl_service._evaluate(agent, df_test)
                rl_metrics["strategy"] = "rl"
                rl_metrics["model_id"] = rl_id
                results["rl"] = rl_metrics
            except Exception as e:
                results["rl"] = {"error": str(e)}
        else:
            results["rl"] = {"error": "Aucun modèle RL entraîné. Appeler POST /rl/train d'abord."}

        # ── Classement ───────────────────────────────────────────────────────
        ranking = self._rank(results)

        return {
            "strategies": results,
            "ranking_by_sharpe": ranking,
            "summary": self._summary(results),
        }

    def stress_test_quarterly(
        self,
        df_test_2024: pd.DataFrame,
        ml_model_id: Optional[str] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Stress test trimestriel sur 2024.
        Évalue chaque stratégie sur les 4 trimestres.
        """
        if "timestamp" not in df_test_2024.columns:
            return {"error": "Colonne 'timestamp' manquante pour le stress test."}

        ts = pd.to_datetime(df_test_2024["timestamp"])
        quarters = {
            "Q1": (ts.dt.month.isin([1, 2, 3])),
            "Q2": (ts.dt.month.isin([4, 5, 6])),
            "Q3": (ts.dt.month.isin([7, 8, 9])),
            "Q4": (ts.dt.month.isin([10, 11, 12])),
        }

        quarterly_results: Dict[str, Any] = {}
        for q, mask in quarters.items():
            df_q = df_test_2024[mask].reset_index(drop=True)
            if len(df_q) < 50:
                quarterly_results[q] = {"error": "Trop peu de données pour ce trimestre."}
                continue
            quarterly_results[q] = self.compare_all(
                df_q, ml_model_id=ml_model_id, seed=seed
            )

        return {"quarterly_stress_tests": quarterly_results}

    def model_robustness_report(self, ml_model_id: str) -> Dict[str, Any]:
        """
        Résumé de robustesse d'un modèle ML :
        compare les métriques train / val / test.
        """
        try:
            model_data = get_model(ml_model_id)
        except KeyError as e:
            return {"error": str(e)}

        metrics = model_data.get("metrics", {})
        train_sharpe = metrics.get("train", {}).get("sharpe", None)
        val_sharpe = metrics.get("val", {}).get("sharpe", None)
        test_sharpe = metrics.get("test", {}).get("sharpe", None)

        overfitting_flag = False
        if train_sharpe is not None and val_sharpe is not None:
            if train_sharpe > 0 and val_sharpe < 0:
                overfitting_flag = True
            elif train_sharpe > 0 and val_sharpe / train_sharpe < 0.3:
                overfitting_flag = True

        is_valid = (
            val_sharpe is not None and val_sharpe > 0
            and test_sharpe is not None and test_sharpe > 0
        )

        return {
            "model_id": ml_model_id,
            "model_type": model_data.get("model_type"),
            "version": model_data.get("version"),
            "sharpe": {
                "train_2022": train_sharpe,
                "val_2023": val_sharpe,
                "test_2024": test_sharpe,
            },
            "max_drawdown": {
                "train_2022": metrics.get("train", {}).get("max_drawdown_pct"),
                "val_2023": metrics.get("val", {}).get("max_drawdown_pct"),
                "test_2024": metrics.get("test", {}).get("max_drawdown_pct"),
            },
            "overfitting_detected": overfitting_flag,
            "is_valid_on_2024": is_valid,
            "verdict": (
                "✅ Modèle robuste : Sharpe positif sur train, val ET test." if is_valid
                else "❌ Modèle non valide : peu robuste sur 2024. Revoir features ou hyperparamètres."
            ),
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_close(df: pd.DataFrame) -> pd.Series:
        for col in ["close_15m", "close", "Close"]:
            if col in df.columns:
                return df[col].astype(float).reset_index(drop=True)
        raise KeyError(f"Colonne close manquante. Disponibles : {list(df.columns)}")

    @staticmethod
    def _rank(results: Dict[str, Any]) -> List[Dict]:
        ranking = []
        for name, m in results.items():
            if "error" not in m and isinstance(m, dict):
                ranking.append({
                    "strategy": name,
                    "sharpe": m.get("sharpe", None),
                    "total_return_pct": m.get("total_return_pct", None),
                    "max_drawdown_pct": m.get("max_drawdown_pct", None),
                    "profit_factor": m.get("profit_factor", None),
                })
        # Trier par Sharpe décroissant
        ranking.sort(key=lambda x: (x["sharpe"] is not None, x["sharpe"] or -999), reverse=True)
        for i, r in enumerate(ranking):
            r["rank"] = i + 1
        return ranking

    @staticmethod
    def _summary(results: Dict[str, Any]) -> Dict[str, str]:
        summary = {}
        for name, m in results.items():
            if "error" in m:
                summary[name] = f"Erreur : {m['error']}"
            else:
                sharpe = m.get("sharpe")
                ret = m.get("total_return_pct")
                mdd = m.get("max_drawdown_pct")
                summary[name] = (
                    f"Sharpe={sharpe:.2f} | Return={ret:.2f}% | MDD={mdd:.2f}%"
                    if all(v is not None for v in [sharpe, ret, mdd])
                    else "données insuffisantes"
                )
        return summary
