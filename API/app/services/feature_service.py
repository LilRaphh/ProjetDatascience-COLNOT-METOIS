"""
Service Feature Engineering – Version 2
Calcul des features M15 pour GBP/USD.
Toutes les features sont calculées UNIQUEMENT à partir du passé (no look-ahead).

Blocs :
  1. Court terme  : return_1, return_4, ema_20/50/diff, rsi_14,
                    rolling_std_20, range_15m, body, upper_wick, lower_wick
  2. Contexte & Régime : ema_200, distance_to_ema200, slope_ema50,
                          atr_14, rolling_std_100, volatility_ratio,
                          adx_14, macd, macd_signal
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS EMA / ROLLING
# ─────────────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Simplified ADX."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    dm_plus = np.where(
        (high - prev_high) > (prev_low - low),
        np.maximum(high - prev_high, 0),
        0,
    )
    dm_minus = np.where(
        (prev_low - low) > (high - prev_high),
        np.maximum(prev_low - low, 0),
        0,
    )

    dm_plus_s = pd.Series(dm_plus, index=close.index).ewm(com=period - 1, adjust=False).mean()
    dm_minus_s = pd.Series(dm_minus, index=close.index).ewm(com=period - 1, adjust=False).mean()
    tr_s = tr.ewm(com=period - 1, adjust=False).mean()

    di_plus = 100 * dm_plus_s / tr_s.replace(0, np.nan)
    di_minus = 100 * dm_minus_s / tr_s.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    return dx.ewm(com=period - 1, adjust=False).mean()


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SERVICE
# ─────────────────────────────────────────────────────────────────────────────

class FeatureService:
    """
    Calcule l'ensemble des features V2 sur un DataFrame M15.

    Colonnes attendues en entrée :
        timestamp, open_15m, high_15m, low_15m, close_15m
        (volume_15m optionnel)

    Colonnes ajoutées :
        return_1, return_4, ema_20, ema_50, ema_diff,
        rsi_14, rolling_std_20, range_15m, body, upper_wick, lower_wick,
        ema_200, distance_to_ema200, slope_ema50,
        atr_14, rolling_std_100, volatility_ratio,
        adx_14, macd, macd_signal
    """

    SHORT_TERM_FEATURES = [
        "return_1", "return_4",
        "ema_20", "ema_50", "ema_diff",
        "rsi_14", "rolling_std_20",
        "range_15m", "body", "upper_wick", "lower_wick",
    ]

    REGIME_FEATURES = [
        "ema_200", "distance_to_ema200", "slope_ema50",
        "atr_14", "rolling_std_100", "volatility_ratio",
        "adx_14", "macd", "macd_signal",
    ]

    ALL_FEATURES = SHORT_TERM_FEATURES + REGIME_FEATURES

    # Minimum de bougies nécessaires pour éviter NaN sur toutes les features
    WARMUP_BARS = 210  # ema_200 + marge

    def compute(
        self,
        df_m15: pd.DataFrame,
        drop_na: bool = True,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Calcule les features sur df_m15.

        Returns
        -------
        df_feat  : DataFrame M15 enrichi des features
        report   : stats de calcul
        """
        df = df_m15.copy()

        # ── Normalisation des colonnes ──────────────────────────────────────
        rename = {}
        for raw, std in [
            ("open", "open_15m"), ("high", "high_15m"),
            ("low", "low_15m"), ("close", "close_15m"),
        ]:
            if raw in df.columns and std not in df.columns:
                rename[raw] = std
        if rename:
            df = df.rename(columns=rename)

        required = ["open_15m", "high_15m", "low_15m", "close_15m"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Colonnes M15 manquantes : {missing}. Disponibles : {list(df.columns)}")

        o = df["open_15m"]
        h = df["high_15m"]
        l = df["low_15m"]
        c = df["close_15m"]

        n_input = len(df)

        # ── BLOC 1 : Court terme ────────────────────────────────────────────
        df["return_1"] = c.pct_change(1)
        df["return_4"] = c.pct_change(4)

        df["ema_20"] = _ema(c, 20)
        df["ema_50"] = _ema(c, 50)
        df["ema_diff"] = df["ema_20"] - df["ema_50"]

        df["rsi_14"] = _rsi(c, 14)

        df["rolling_std_20"] = c.rolling(20).std()

        df["range_15m"] = h - l
        df["body"] = (c - o).abs()
        df["upper_wick"] = h - pd.concat([c, o], axis=1).max(axis=1)
        df["lower_wick"] = pd.concat([c, o], axis=1).min(axis=1) - l

        # ── BLOC 2 : Contexte & Régime ─────────────────────────────────────
        df["ema_200"] = _ema(c, 200)
        df["distance_to_ema200"] = (c - df["ema_200"]) / df["ema_200"].replace(0, np.nan)

        # Pente ema_50 : différence entre la valeur actuelle et celle d'il y a 5 bars
        df["slope_ema50"] = df["ema_50"].diff(5)

        df["atr_14"] = _atr(h, l, c, 14)
        df["rolling_std_100"] = c.rolling(100).std()

        # Ratio de volatilité : atr_14 / rolling_std_100 (court vs long terme)
        df["volatility_ratio"] = df["atr_14"] / df["rolling_std_100"].replace(0, np.nan)

        df["adx_14"] = _adx(h, l, c, 14)

        # MACD = ema_12 - ema_26 / signal = ema_9(macd)
        ema12 = _ema(c, 12)
        ema26 = _ema(c, 26)
        df["macd"] = ema12 - ema26
        df["macd_signal"] = _ema(df["macd"], 9)

        # ── Suppression des NaN (warm-up) ──────────────────────────────────
        n_before_drop = len(df)
        if drop_na:
            df = df.dropna(subset=self.ALL_FEATURES).reset_index(drop=True)
        n_after_drop = len(df)

        report = {
            "n_input": int(n_input),
            "n_output": int(len(df)),
            "n_dropped_warmup": int(n_before_drop - n_after_drop),
            "features_computed": self.ALL_FEATURES,
            "n_features": len(self.ALL_FEATURES),
        }

        return df, report

    def compute_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute la colonne cible binaire :
            y = 1 si close_t+1 > close_t, sinon 0

        La DERNIÈRE ligne aura target = NaN (pas de futur).
        """
        df = df.copy()
        df["target"] = (df["close_15m"].shift(-1) > df["close_15m"]).astype(float)
        df.loc[df.index[-1], "target"] = np.nan  # dernière ligne sans cible
        return df
