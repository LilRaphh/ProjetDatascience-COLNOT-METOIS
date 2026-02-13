"""
Service Reinforcement Learning – GBP/USD M15
Conception obligatoire sur papier → implémentation

Problème métier : agent doit maximiser le PnL cumulé sur GBP/USD M15
Contraintes : coûts de transaction, drawdown limité, horizon épisodique

State   : fenêtre glissante de 20 bougies × (features + position actuelle)
          normalisé z-score sur la fenêtre
Action  : discret {0=HOLD, 1=BUY, 2=SELL}
Reward  : PnL réalisé – coût de transaction – pénalité drawdown
Algo    : Q-Learning tabulaire (features discrétisées) → fonctionne sans GPU
          Walk-forward : train 2022, val 2023, test 2024
"""

from __future__ import annotations

import random
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Iterator


TRANSACTION_COST = 0.0002
MAX_DRAWDOWN_LIMIT = 0.20    # Pénalité si drawdown > 20%
WINDOW_SIZE = 20             # Bougies dans l'état
N_FEATURES = 10              # Nombre de features dans l'état (sous-ensemble compact)
N_ACTIONS = 3                # HOLD=0, BUY=1, SELL=2
GAMMA = 0.95                 # Facteur d'actualisation
ALPHA = 0.01                 # Taux d'apprentissage
EPSILON_START = 1.0          # Exploration initiale
EPSILON_MIN = 0.05           # Exploration minimale
EPSILON_DECAY = 0.998        # Décroissance epsilon


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class TradingEnv:
    """
    Environnement de trading M15 pour le RL.

    State : vecteur de dimension (WINDOW_SIZE × N_FEATURES + 3)
            où les 3 derniers éléments sont : position (-1/0/1), PnL normalisé, drawdown
    Action : 0=HOLD, 1=BUY(long), 2=SELL(short)
    Reward : PnL de la step – transaction_cost – pénalité_drawdown
    """

    ACTION_HOLD = 0
    ACTION_BUY = 1
    ACTION_SELL = 2

    # Features compactes utilisées dans le state (sous-ensemble pertinent)
    STATE_FEATURES = [
        "return_1", "return_4",
        "ema_diff", "rsi_14",
        "rolling_std_20",
        "distance_to_ema200", "slope_ema50",
        "atr_14", "macd", "adx_14",
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.close = self._get_close(df)
        self.n_steps = len(self.close)

        # Vérifier features disponibles
        self.feature_cols = [c for c in self.STATE_FEATURES if c in df.columns]
        if len(self.feature_cols) < 3:
            # Fallback : utiliser return_1 et rolling_std si features manquantes
            c = self.close
            self.df["_ret1"] = c.pct_change().fillna(0)
            self.df["_std20"] = c.rolling(20).std().fillna(0)
            self.feature_cols = ["_ret1", "_std20"]

        self.reset()
    
    def _get_close(self, df: pd.DataFrame) -> pd.Series:
        """Récupère la colonne 'close' existante."""
        candidates = ["close_15m", "close", "Close"]
        for c in candidates:
            if c in df.columns:
                return df[c]
        raise ValueError(f"Colonne 'close' introuvable dans le DataFrame. Colonnes: {list(df.columns)}")

    def reset(self) -> np.ndarray:
        self.t = WINDOW_SIZE
        self.position = 0         # -1=short, 0=flat, 1=long
        self.entry_price = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.total_pnl = 0.0
        self.done = False
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        assert not self.done, "Episode terminé, appeler reset()."

        price_now = float(self.close.iloc[self.t])
        price_prev = float(self.close.iloc[self.t - 1])
        step_return = (price_now - price_prev) / price_prev

        # ── Récompense ────────────────────────────────────────────────────────
        reward = 0.0
        info: Dict[str, Any] = {}

        # Coût de transaction si changement de position
        prev_position = self.position
        new_position = {
            self.ACTION_HOLD: 0,
            self.ACTION_BUY: 1,
            self.ACTION_SELL: -1,
        }[action]

        if new_position != prev_position:
            reward -= TRANSACTION_COST
            info["transaction"] = True
        else:
            info["transaction"] = False

        self.position = new_position

        # Rendement selon position
        pnl = self.position * step_return
        reward += pnl

        # Mise à jour equity
        self.equity *= (1 + pnl)
        self.total_pnl += pnl

        # Drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        drawdown = (self.equity - self.peak_equity) / self.peak_equity

        # Pénalité drawdown excessif
        if drawdown < -MAX_DRAWDOWN_LIMIT:
            reward -= 0.01  # pénalité supplémentaire
            info["drawdown_penalty"] = True

        info.update({
            "step": self.t,
            "position": self.position,
            "pnl": round(pnl, 6),
            "equity": round(self.equity, 6),
            "drawdown": round(drawdown, 4),
        })

        self.t += 1
        self.done = self.t >= self.n_steps

        next_state = self._get_state() if not self.done else np.zeros(self._state_dim())
        return next_state, float(reward), self.done, info

    def _get_state(self) -> np.ndarray:
        """Construit le vecteur d'état."""
        start = self.t - WINDOW_SIZE
        end = self.t

        # Features M15 normalisées sur la fenêtre
        feat_window = self.df[self.feature_cols].iloc[start:end].values.astype(float)

        # Normalisation z-score par feature
        means = np.nanmean(feat_window, axis=0)
        stds = np.nanstd(feat_window, axis=0) + 1e-8
        feat_norm = (feat_window - means) / stds
        feat_norm = np.nan_to_num(feat_norm, 0)

        # Aplatir la fenêtre (dernière bougie uniquement pour limiter la dim)
        last_feat = feat_norm[-1]  # vecteur de taille N_features

        # Informations de position
        price_now = float(self.close.iloc[self.t - 1])
        drawdown = (self.equity - self.peak_equity) / (self.peak_equity + 1e-8)

        state = np.concatenate([
            last_feat,
            [float(self.position), self.total_pnl * 100, drawdown],
        ])
        return state.astype(np.float32)

    def _state_dim(self) -> int:
        return len(self.feature_cols) + 3


# ─────────────────────────────────────────────────────────────────────────────
# Q-LEARNING AGENT (tabulaire via discrétisation)
# ─────────────────────────────────────────────────────────────────────────────

class QLearningAgent:
    """
    Agent Q-Learning avec état discrétisé.
    State : on utilise seulement les signes (+/-) des features → 2^N états.
    Adapté pour fonctionner sans GPU et sans bibliothèque externe.
    """

    def __init__(self, state_dim: int, n_actions: int = N_ACTIONS):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.q_table: Dict[Tuple, np.ndarray] = {}
        self.epsilon = EPSILON_START
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.step_count = 0

    def _discretize(self, state: np.ndarray) -> Tuple:
        """Discrétisation simple : signe de chaque composante."""
        return tuple(np.sign(state).astype(int))

    def get_q(self, state: np.ndarray) -> np.ndarray:
        key = self._discretize(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        return self.q_table[key]

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.get_q(state)))

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        q = self.get_q(state)
        q_next = self.get_q(next_state)
        target = reward + (0 if done else self.gamma * np.max(q_next))
        td_error = target - q[action]
        q[action] += self.alpha * td_error
        self.step_count += 1
        return float(td_error)

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def info(self) -> Dict[str, Any]:
        return {
            "n_states_visited": len(self.q_table),
            "epsilon": round(self.epsilon, 4),
            "step_count": self.step_count,
        }


# ─────────────────────────────────────────────────────────────────────────────
# RL SERVICE
# ─────────────────────────────────────────────────────────────────────────────

_RL_MODELS: Dict[str, Dict[str, Any]] = {}
_BEST_RL_ID: Optional[str] = None


class RLService:
    """
    Entraîne un agent Q-Learning et évalue ses performances.
    Walk-forward : train 2022 / val 2023 / test 2024.
    """

    def train_gen(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: Optional[pd.DataFrame] = None,
        n_episodes: int = 10,
        seed: int = 42,
    ) -> Iterator[Dict[str, Any]]:
        """
        Entraîne l'agent et yield des événements de progression.
        """
        try:
            random.seed(seed)
            np.random.seed(seed)

            env_train = TradingEnv(df_train)
            agent = QLearningAgent(state_dim=env_train._state_dim())

            train_rewards: List[float] = []
            train_equities: List[float] = []

            for ep in range(n_episodes):
                state = env_train.reset()
                ep_reward = 0.0

                while True:
                    action = agent.act(state)
                    next_state, reward, done, _ = env_train.step(action)
                    agent.learn(state, action, reward, next_state, done)
                    ep_reward += reward
                    state = next_state
                    if done:
                        break

                agent.decay_epsilon()
                train_rewards.append(round(ep_reward, 4))
                train_equities.append(round(env_train.equity, 4))

                # Yield progress event
                yield {
                    "type": "progress",
                    "episode": ep + 1,
                    "total_episodes": n_episodes,
                    "reward": round(ep_reward, 4),
                    "equity": round(env_train.equity, 4),
                    "epsilon": round(agent.epsilon, 4),
                }

            # ── Évaluation validation ──────────────────────────────────────────────
            val_metrics = self._evaluate(agent, df_val)
            test_metrics = self._evaluate(agent, df_test) if df_test is not None else {}

            model_id = f"rl_qlearning_{n_episodes}ep_{seed}seed"
            model_data = {
                "model_id": model_id,
                "algorithm": "Q-Learning (tabulaire)",
                "hyperparams": {
                    "gamma": GAMMA,
                    "alpha": ALPHA,
                    "epsilon_start": EPSILON_START,
                    "epsilon_min": EPSILON_MIN,
                    "epsilon_decay": EPSILON_DECAY,
                    "n_episodes": n_episodes,
                    "window_size": WINDOW_SIZE,
                    "seed": seed,
                },
                "agent_info": agent.info(),
                "train_rewards_per_episode": train_rewards,
                "train_final_equities": train_equities,
                "metrics": {
                    "train": {
                        "mean_reward_per_episode": round(float(np.mean(train_rewards)), 4),
                        "final_equity": train_equities[-1] if train_equities else 1.0,
                    },
                    "val": val_metrics,
                    "test": test_metrics,
                },
                "_agent": agent,
            }

            global _BEST_RL_ID
            _RL_MODELS[model_id] = model_data
            if _BEST_RL_ID is None or (
                val_metrics.get("sharpe", -999)
                > _RL_MODELS.get(_BEST_RL_ID, {}).get("metrics", {}).get("val", {}).get("sharpe", -999)
            ):
                _BEST_RL_ID = model_id

            yield {
                "type": "result",
                "payload": self._serializable(model_data)
            }

        except Exception as e:
            yield {
                "type": "error",
                "message": str(e)
            }
            # Re-raise pour logging serveur si besoin, mais le yield error suffit pour le client
            raise e

    def _evaluate(
        self,
        agent: QLearningAgent,
        df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Lance un épisode greedy et calcule les métriques financières."""
        env = TradingEnv(df)
        state = env.reset()
        close = env.close
        signals = []

        while True:
            action = agent.act(state, greedy=True)
            signals.append(action)
            next_state, _, done, _ = env.step(action)
            state = next_state
            if done:
                break

        # Aligner signaux / prix (on a step de 1..n_steps)
        signals_arr = np.array(signals[:len(close) - 1])
        # Convertir 0/1/2 → 0/1/-1 pour le backtest
        sig_mapped = np.where(
            signals_arr == 1, 1, np.where(signals_arr == 2, -1, 0)
        )

        # Backtest
        pnl = np.zeros(len(close))
        position = 0
        trades = 0
        for i in range(1, len(close)):
            if i - 1 < len(sig_mapped):
                new_pos = int(sig_mapped[i - 1])
                if new_pos != position:
                    pnl[i] -= TRANSACTION_COST
                    trades += 1
                position = new_pos
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

    def get_model(self, model_id: str) -> Dict[str, Any]:
        if model_id not in _RL_MODELS:
            raise KeyError(f"Modèle RL introuvable: {model_id}")
        return _RL_MODELS[model_id]

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "model_id": k,
                "algorithm": v["algorithm"],
                "is_best": k == _BEST_RL_ID,
                "val_sharpe": v.get("metrics", {}).get("val", {}).get("sharpe"),
                "n_states_visited": v.get("agent_info", {}).get("n_states_visited"),
            }
            for k, v in _RL_MODELS.items()
        ]

    @staticmethod
    def _serializable(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    @staticmethod
    def design_document() -> Dict[str, Any]:
        """Retourne la conception obligatoire sur papier (section 9.1)."""
        return {
            "1_probleme_metier": {
                "objectif": "Maximiser le PnL cumulé sur GBP/USD M15 sous contraintes réalistes.",
                "contraintes": [
                    "Coûts de transaction : 2 pips par trade",
                    "Drawdown maximum autorisé : 20%",
                    "Horizon : épisodique (une année par episode)",
                    "Décisions : toutes les 15 minutes",
                ],
            },
            "2_donnees": {
                "qualite": "CSV M1 → agrégé M15 → nettoyé → features V2",
                "alignement": "Split strict 2022 train / 2023 val / 2024 test",
                "couts": "Spread 2 pips = 0.02% par transaction",
            },
            "3_state": {
                "features": TradingEnv.STATE_FEATURES,
                "normalisation": "z-score sur fenêtre glissante de 20 bougies",
                "warm_up": f"{WINDOW_SIZE} bougies M15 nécessaires au démarrage",
                "dimension": f"{len(TradingEnv.STATE_FEATURES)} features + 3 (position, pnl, drawdown)",
            },
            "4_action": {
                "espace": "Discret",
                "actions": {"0": "HOLD (position neutre)", "1": "BUY (long)", "2": "SELL (short)"},
            },
            "5_reward": {
                "formule": "PnL_step – transaction_cost – pénalité_drawdown",
                "pnl": "position × (close_t / close_{t-1} - 1)",
                "penalite_drawdown": "−0.01 si drawdown > 20%",
            },
            "6_environnement": {
                "simulateur": "TradingEnv (classe Python, épisodes journée-glissante)",
                "slippage": "non modélisé (données EOD M15)",
                "transaction_cost": f"{TRANSACTION_COST * 100}% par trade",
            },
            "7_algorithme": {
                "choix": "Q-Learning tabulaire",
                "justification": (
                    "Pas de GPU requis. L'espace d'état discrétisé (signes) "
                    "est gérable en mémoire. Convergence prouvée pour MDPs finis. "
                    "Extension naturelle vers DQN si ressources disponibles."
                ),
            },
            "8_parametres_cles": {
                "gamma": GAMMA,
                "learning_rate": ALPHA,
                "epsilon_start": EPSILON_START,
                "epsilon_min": EPSILON_MIN,
                "epsilon_decay": EPSILON_DECAY,
                "window_size": WINDOW_SIZE,
                "batch_size": "N/A (online Q-Learning)",
                "seed": 42,
            },
            "9_evaluation": {
                "split": "Walk-forward 2022→2023→2024",
                "metriques": ["Sharpe", "MDD", "profit_factor", "total_return"],
                "stress_tests": "Évaluation séparée sur chaque trimestre 2024",
            },
        }
