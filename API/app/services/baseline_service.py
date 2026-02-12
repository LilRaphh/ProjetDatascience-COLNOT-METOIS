"""
Service Baseline – Stratégies de référence obligatoires
  1. Random          : signal aléatoire (BUY/SELL/HOLD)
  2. Buy & Hold      : BUY permanent
  3. Règles fixes    : croisement EMA20/EMA50 + RSI filtre
  4. Backtest simple avec coûts de transaction et drawdown
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

TRANSACTION_COST_PCT = 0.0002  # 2 pips sur GBP/USD (spread)
SIGNAL_BUY = 1
SIGNAL_SELL = -1
SIGNAL_HOLD = 0


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _backtest(
    close: pd.Series,
    signals: pd.Series,
    transaction_cost: float = TRANSACTION_COST_PCT,
) -> Dict[str, Any]:
    """
    Backtest vectorisé à partir d'une série de signaux (-1, 0, 1).
    - LONG  (+1) : on détient la position
    - SHORT (-1) : on est short
    - HOLD  (0)  : on ferme toute position

    Returns
    -------
    dict avec : equity_curve (liste), métriques financières
    """
    n = len(close)
    equity = np.ones(n)
    position = 0  # position actuelle (-1, 0, 1)
    pnl = np.zeros(n)
    trades = 0

    for i in range(1, n):
        prev_pos = position
        sig = int(signals.iloc[i - 1])  # signal émis à la bougie i-1

        # Changement de position ?
        if sig != prev_pos:
            # Coût de transaction si on ouvre ou ferme
            if sig != SIGNAL_HOLD or prev_pos != SIGNAL_HOLD:
                pnl[i] -= transaction_cost
                trades += 1
            position = sig

        # Rendement de la position
        ret = (close.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1]
        pnl[i] += position * ret

    equity = np.cumprod(1 + pnl)

    # ── Métriques ─────────────────────────────────────────────────────────────
    total_return = float(equity[-1] - 1)

    # Maximum Drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min())

    # Sharpe simplifié (annualisé M15 : 252j × 26 bougies/jour ≈ 6552 bougies/an)
    pnl_series = pd.Series(pnl[1:])
    sharpe = (
        float(pnl_series.mean() / pnl_series.std() * np.sqrt(6552))
        if pnl_series.std() > 0
        else 0.0
    )

    # Profit Factor
    wins = pnl_series[pnl_series > 0].sum()
    losses = pnl_series[pnl_series < 0].abs().sum()
    profit_factor = float(wins / losses) if losses > 0 else float("inf")

    # Win rate
    win_rate = float((pnl_series > 0).mean())

    return {
        "total_return_pct": round(total_return * 100, 4),
        "max_drawdown_pct": round(max_drawdown * 100, 4),
        "sharpe": round(sharpe, 4),
        "profit_factor": round(profit_factor, 4),
        "n_trades": int(trades),
        "win_rate_pct": round(win_rate * 100, 2),
        "final_equity": round(float(equity[-1]), 6),
        "equity_curve": [round(float(v), 6) for v in equity[::10]],  # every 10 bars for size
    }


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE SERVICE
# ─────────────────────────────────────────────────────────────────────────────

class BaselineService:

    # ── 1. Stratégie aléatoire ────────────────────────────────────────────────
    def random_strategy(
        self,
        df: pd.DataFrame,
        seed: int = 42,
        hold_prob: float = 0.5,
    ) -> Tuple[pd.Series, Dict]:
        """
        Génère des signaux aléatoires.
        hold_prob : probabilité d'un signal HOLD.
        Reste réparti équitablement entre BUY et SELL.
        """
        rng = np.random.default_rng(seed)
        n = len(df)
        side_prob = (1 - hold_prob) / 2

        choices = rng.choice(
            [SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD],
            size=n,
            p=[side_prob, side_prob, hold_prob],
        )
        signals = pd.Series(choices, index=df.index, name="signal_random")
        close = self._get_close(df)
        metrics = _backtest(close, signals)
        return signals, {"strategy": "random", "seed": seed, **metrics}

    # ── 2. Buy & Hold ─────────────────────────────────────────────────────────
    def buy_and_hold(self, df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Signal BUY permanent (pas de SELL, pas de HOLD)."""
        n = len(df)
        signals = pd.Series(
            np.ones(n, dtype=int), index=df.index, name="signal_bah"
        )
        close = self._get_close(df)
        metrics = _backtest(close, signals, transaction_cost=0.0)
        # Avec coûts (1 seul trade initial)
        metrics_with_cost = _backtest(close, signals, transaction_cost=TRANSACTION_COST_PCT)
        metrics_with_cost["strategy"] = "buy_and_hold"
        return signals, metrics_with_cost

    # ── 3. Stratégie règles fixes ──────────────────────────────────────────────
    def fixed_rules_strategy(
        self,
        df: pd.DataFrame,
        ema_short: int = 20,
        ema_long: int = 50,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
    ) -> Tuple[pd.Series, Dict]:
        """
        Stratégie à règles fixes :
          - BUY  : EMA_short > EMA_long ET RSI < rsi_overbought
          - SELL : EMA_short < EMA_long ET RSI > rsi_oversold
          - HOLD : sinon
        """
        close = self._get_close(df)

        ema_s = close.ewm(span=ema_short, adjust=False).mean()
        ema_l = close.ewm(span=ema_long, adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0).ewm(com=rsi_period - 1, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=rsi_period - 1, adjust=False).mean()
        rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        signals = pd.Series(SIGNAL_HOLD, index=df.index, name="signal_rules")
        buy_mask = (ema_s > ema_l) & (rsi < rsi_overbought)
        sell_mask = (ema_s < ema_l) & (rsi > rsi_oversold)

        signals[buy_mask] = SIGNAL_BUY
        signals[sell_mask] = SIGNAL_SELL

        metrics = _backtest(close, signals)
        metrics.update({
            "strategy": "fixed_rules",
            "params": {
                "ema_short": ema_short,
                "ema_long": ema_long,
                "rsi_period": rsi_period,
                "rsi_overbought": rsi_overbought,
                "rsi_oversold": rsi_oversold,
            },
            "n_buy_signals": int((signals == SIGNAL_BUY).sum()),
            "n_sell_signals": int((signals == SIGNAL_SELL).sum()),
            "n_hold_signals": int((signals == SIGNAL_HOLD).sum()),
        })
        return signals, metrics

    # ── 4. Comparaison toutes stratégies ──────────────────────────────────────
    def compare_all(
        self,
        df: pd.DataFrame,
        seed: int = 42,
    ) -> Dict[str, Dict]:
        _, random_metrics = self.random_strategy(df, seed=seed)
        _, bah_metrics = self.buy_and_hold(df)
        _, rules_metrics = self.fixed_rules_strategy(df)

        return {
            "random": random_metrics,
            "buy_and_hold": bah_metrics,
            "fixed_rules": rules_metrics,
        }

    # ── helper ────────────────────────────────────────────────────────────────
    @staticmethod
    def _get_close(df: pd.DataFrame) -> pd.Series:
        for col in ["close_15m", "close", "Close"]:
            if col in df.columns:
                return df[col].astype(float).reset_index(drop=True)
        raise KeyError(
            f"Aucune colonne close. Disponibles : {list(df.columns)}"
        )
