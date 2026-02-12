from __future__ import annotations

from typing import Dict, Tuple
import pandas as pd


class M15CleanService:
    """
    Nettoyage M15
    - drop bougies incomplètes (NaN sur OHLC)
    - contrôle prix négatifs / incohérences OHLC
    - détection gaps anormaux via retours (seuil configurable)
    """

    def clean(
        self,
        df_m15: pd.DataFrame,
        gap_return_threshold: float = 0.02,  # 2% par bougie 15m
        drop_gaps: bool = True,
    ) -> Tuple[pd.DataFrame, Dict]:
        df = df_m15.copy()

        # 1) timestamp
        if "timestamp" not in df.columns:
            raise KeyError("Colonne 'timestamp' absente du M15.")

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        # 2) détecter les colonnes OHLC M15 (open_15m etc.)
        # (supporte aussi open/high/low/close si tu changes plus tard)
        candidates = [
            ("open_15m", "high_15m", "low_15m", "close_15m"),
            ("open", "high", "low", "close"),
        ]

        ohlc = None
        for cols in candidates:
            if all(c in df.columns for c in cols):
                ohlc = cols
                break

        if ohlc is None:
        # cas fréquent : on a passé un M1 (Date/Time/Open/High/Low/Close)
            if ("Date" in df.columns and "Time" in df.columns) and (
                "Open" in df.columns or "Close" in df.columns
            ):
                raise ValueError(
                    "Dataset au format M1 détecté (Date/Time/Open/High/Low/Close). "
                    "Fais d'abord POST /m15/aggregate sur le dataset M1, "
                    "puis appelle POST /m15/clean sur le dataset *_m15."
                )

            raise KeyError(f"Colonnes OHLC introuvables. Colonnes dispo: {list(df.columns)}")

        o_col, h_col, l_col, c_col = ohlc

        report: Dict = {
            "input_rows": int(len(df)),
            "dropped_bad_timestamp": int(df["timestamp"].isna().sum()),
            "dropped_incomplete_ohlc": 0,
            "dropped_negative_prices": 0,
            "dropped_ohlc_incoherence": 0,
            "flagged_gaps": 0,
            "dropped_gaps": 0,
            "gap_return_threshold": float(gap_return_threshold),
        }

        # 3) bougies incomplètes (NaN sur OHLC)
        before = len(df)
        df = df.dropna(subset=[o_col, h_col, l_col, c_col])
        report["dropped_incomplete_ohlc"] = int(before - len(df))

        # 4) prix négatifs / nuls (Forex -> doit être > 0)
        before = len(df)
        mask_pos = (df[[o_col, h_col, l_col, c_col]] > 0).all(axis=1)
        df = df[mask_pos]
        report["dropped_negative_prices"] = int(before - len(df))

        # 5) incohérences OHLC (high >= max(open,close) et low <= min(open,close))
        before = len(df)
        max_oc = df[[o_col, c_col]].max(axis=1)
        min_oc = df[[o_col, c_col]].min(axis=1)
        mask_ok = (df[h_col] >= max_oc) & (df[l_col] <= min_oc) & (df[h_col] >= df[l_col])
        df = df[mask_ok]
        report["dropped_ohlc_incoherence"] = int(before - len(df))

        # 6) gaps anormaux (sur close)
        # return = (close_t / close_{t-1}) - 1
        df["_prev_close"] = df[c_col].shift(1)
        df["_ret"] = (df[c_col] / df["_prev_close"]) - 1.0

        # on ignore la première ligne (NaN)
        gap_mask = df["_ret"].abs() > gap_return_threshold
        report["flagged_gaps"] = int(gap_mask.sum())

        if drop_gaps:
            before = len(df)
            df = df[~gap_mask]
            report["dropped_gaps"] = int(before - len(df))

        df = df.drop(columns=["_prev_close", "_ret"])

        report["output_rows"] = int(len(df))
        report["dropped_total"] = int(report["input_rows"] - report["output_rows"])

        return df.reset_index(drop=True), report


m15_clean_service = M15CleanService()
