"""
Service Analyse Exploratoire (EDA) – GBP/USD M15
Obligatoire :
  - Distribution des rendements
  - Volatilité dans le temps
  - Analyse horaire
  - Autocorrélation
  - Test ADF (stationnarité)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List


class EDAService:
    """
    Calcule toutes les statistiques exploratoires sur un DataFrame M15.
    """

    def analyse_returns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Distribution des rendements M15."""
        c = self._get_close(df)
        returns = c.pct_change().dropna()

        # Test de normalité (Shapiro-Wilk sur un sous-ensemble si trop grand)
        sample = returns.sample(min(500, len(returns)), random_state=42) if len(returns) > 500 else returns
        shapiro_stat, shapiro_p = stats.shapiro(sample)

        # Kurtosis & skewness
        kurt = float(returns.kurtosis())
        skew = float(returns.skew())

        percentiles = {
            f"p{p}": float(returns.quantile(p / 100))
            for p in [1, 5, 25, 50, 75, 95, 99]
        }

        return {
            "n_observations": int(len(returns)),
            "mean": float(returns.mean()),
            "std": float(returns.std()),
            "min": float(returns.min()),
            "max": float(returns.max()),
            "skewness": skew,
            "kurtosis_excess": kurt,
            "percentiles": percentiles,
            "sharpe_annualised": float(
                returns.mean() / returns.std() * np.sqrt(252 * 26)
            ) if returns.std() > 0 else 0.0,
            "normality_shapiro": {
                "statistic": float(shapiro_stat),
                "p_value": float(shapiro_p),
                "is_normal_95": bool(shapiro_p > 0.05),
            },
        }

    def analyse_volatility(self, df: pd.DataFrame, window: int = 96) -> Dict[str, Any]:
        """
        Volatilité dans le temps.
        window = 96 bougies M15 ≈ 1 jour de trading.
        """
        c = self._get_close(df)
        returns = c.pct_change()

        rolling_std = returns.rolling(window).std() * np.sqrt(window)  # annualisé empirique

        # Résumé mensuel si timestamp dispo
        monthly: Dict[str, float] = {}
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            monthly_df = pd.DataFrame({"ts": ts, "ret": returns.values})
            monthly_df["month"] = monthly_df["ts"].dt.to_period("M").astype(str)
            monthly = (
                monthly_df.groupby("month")["ret"].std()
                * np.sqrt(window)
            ).dropna().round(6).to_dict()

        return {
            "rolling_window_bars": int(window),
            "mean_volatility": float(rolling_std.mean()) if not rolling_std.isna().all() else 0.0,
            "max_volatility": float(rolling_std.max()) if not rolling_std.isna().all() else 0.0,
            "min_volatility": float(rolling_std.min()) if not rolling_std.isna().all() else 0.0,
            "monthly_volatility": monthly,
        }

    def analyse_hourly(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyse par heure de la journée :
        volume moyen, rendement moyen, volatilité horaire.
        """
        if "timestamp" not in df.columns:
            return {"error": "Colonne 'timestamp' manquante pour l'analyse horaire."}

        c = self._get_close(df)
        returns = c.pct_change()

        ts = pd.to_datetime(df["timestamp"])
        hour = ts.dt.hour

        df_h = pd.DataFrame({
            "hour": hour.values,
            "return": returns.values,
            "range": (df["high_15m"] - df["low_15m"]).values
            if "high_15m" in df.columns and "low_15m" in df.columns
            else np.nan,
        })

        hourly_stats = (
            df_h.groupby("hour")
            .agg(
                mean_return=("return", "mean"),
                std_return=("return", "std"),
                mean_range=("range", "mean"),
                n_bars=("return", "count"),
            )
            .round(8)
        )

        return {
            "hourly_stats": hourly_stats.reset_index().to_dict(orient="records"),
            "best_hour_return": int(hourly_stats["mean_return"].idxmax()),
            "worst_hour_return": int(hourly_stats["mean_return"].idxmin()),
            "most_active_hour": int(hourly_stats["n_bars"].idxmax()),
        }

    def analyse_autocorrelation(
        self, df: pd.DataFrame, max_lags: int = 20
    ) -> Dict[str, Any]:
        """Autocorrélation des rendements (lags 1..max_lags)."""
        c = self._get_close(df)
        returns = c.pct_change().dropna()

        acf_values: List[float] = []
        for lag in range(1, max_lags + 1):
            acf_values.append(float(returns.autocorr(lag=lag)))

        # Borne de confiance 95% (±1.96/√n)
        n = len(returns)
        ci_95 = 1.96 / np.sqrt(n) if n > 0 else 0.0

        return {
            "n_observations": int(n),
            "lags": list(range(1, max_lags + 1)),
            "acf": acf_values,
            "ci_95": float(ci_95),
            "significant_lags": [
                lag
                for lag, acf in zip(range(1, max_lags + 1), acf_values)
                if abs(acf) > ci_95
            ],
        }

    def test_adf(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Test ADF (Augmented Dickey-Fuller) sur la série des clôtures et les rendements.
        H0 : présence d'une racine unitaire (non-stationnaire).
        """
        from scipy.stats import t as t_dist  # fallback

        c = self._get_close(df)
        returns = c.pct_change().dropna()

        def _adf_simple(series: pd.Series, max_lags: int = 5):
            """ADF simplifié par OLS."""
            s = series.dropna().values
            n = len(s)
            if n < max_lags + 10:
                return {"stat": np.nan, "p_value": np.nan, "is_stationary_5pct": False}

            # Régresser Δy_t sur y_{t-1} et Δy_{t-1}, ...Δy_{t-k}
            dy = np.diff(s)
            y_lag = s[:-1]

            # Construire matrice des régresseurs
            X = [y_lag[max_lags:]]
            for k in range(1, max_lags + 1):
                X.append(dy[max_lags - k: -k if k > 0 else None])
            X = np.column_stack(X)
            y = dy[max_lags:]

            # OLS
            try:
                beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
                fitted = X @ beta
                resid = y - fitted
                s2 = np.sum(resid ** 2) / (len(y) - X.shape[1])
                var_beta = s2 * np.linalg.pinv(X.T @ X)[0, 0]
                t_stat = float(beta[0] / np.sqrt(var_beta)) if var_beta > 0 else np.nan

                # Valeurs critiques MacKinnon approchées
                # (sans racine unitaire, seuil 5% ≈ -2.86 pour n grand)
                is_stationary = t_stat < -2.86 if not np.isnan(t_stat) else False

                # p-valeur approximative (interpolation MacKinnon non dispo,
                # on donne une estimation conservative)
                if np.isnan(t_stat):
                    p_value = np.nan
                elif t_stat < -3.75:
                    p_value = 0.01
                elif t_stat < -2.86:
                    p_value = 0.05
                elif t_stat < -2.57:
                    p_value = 0.10
                else:
                    p_value = 0.50  # non stationnaire

                return {
                    "stat": round(float(t_stat), 4),
                    "p_value": float(p_value),
                    "is_stationary_5pct": bool(is_stationary),
                    "critical_value_5pct": -2.86,
                }
            except Exception:
                return {"stat": np.nan, "p_value": np.nan, "is_stationary_5pct": False}

        return {
            "price_series": _adf_simple(c),
            "return_series": _adf_simple(returns),
            "interpretation": (
                "Les rendements sont généralement stationnaires (H0 rejetée). "
                "La série de prix bruts est souvent non-stationnaire (marche aléatoire)."
            ),
        }

    def full_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Lance toutes les analyses et retourne un rapport complet."""
        return {
            "returns": self.analyse_returns(df),
            "volatility": self.analyse_volatility(df),
            "hourly": self.analyse_hourly(df),
            "autocorrelation": self.analyse_autocorrelation(df),
            "adf": self.test_adf(df),
        }

    # ── helper ────────────────────────────────────────────────────────────────
    @staticmethod
    def _get_close(df: pd.DataFrame) -> pd.Series:
        for col in ["close_15m", "close", "Close"]:
            if col in df.columns:
                return df[col].astype(float).reset_index(drop=True)
        raise KeyError(
            f"Aucune colonne 'close' trouvée. Colonnes dispo : {list(df.columns)}"
        )
