from __future__ import annotations

from typing import Dict, Tuple
import pandas as pd


class M15AggregationService:
    def _normalize_colnames(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Nettoyage basique
        df.columns = [c.strip() for c in df.columns]

        # Map de normalisation (MetaTrader / variantes)
        mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "VOLUME": "volume",
        }

        # Applique mapping si la cible n'existe pas déjà
        rename = {}
        for c in df.columns:
            if c in mapping and mapping[c] not in df.columns:
                rename[c] = mapping[c]

        if rename:
            df = df.rename(columns=rename)

        return df

    def _ensure_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            return df

        # fallback Date + Time
        if "Date" in df.columns and "Time" in df.columns:
            ts = df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip()
            df["timestamp"] = pd.to_datetime(ts, errors="coerce")
            return df

        raise KeyError("Pas de colonne 'timestamp' et pas de ('Date','Time') pour reconstruire le temps.")

    def aggregate(self, df_m1: pd.DataFrame, drop_incomplete: bool = True) -> Tuple[pd.DataFrame, Dict]:
        df = df_m1.copy()

        df = self._normalize_colnames(df)
        df = self._ensure_timestamp(df)

        required = ["timestamp", "open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Colonnes manquantes: {missing}. Colonnes dispo: {list(df.columns)}")

        # drop timestamps invalides
        n_before = len(df)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        n_after = len(df)

        df = df.set_index("timestamp")

        agg_map = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in df.columns:
            agg_map["volume"] = "sum"

        df_15 = df.resample("15T").agg(agg_map)

        report: Dict = {
            "input_rows": int(n_before),
            "dropped_bad_timestamp": int(n_before - n_after),
            "resample_rows_before_dropna": int(len(df_15)),
        }

        # Drop bougies vides
        df_15 = df_15.dropna(subset=["open", "high", "low", "close"])
        report["resample_rows_after_dropna"] = int(len(df_15))

        if drop_incomplete:
            # compte de points M1 par tranche 15min
            counts = df["close"].resample("15T").count()
            mask_complete = counts >= 15
            before = len(df_15)
            df_15 = df_15[mask_complete.reindex(df_15.index).fillna(False)]
            report["dropped_incomplete_m15"] = int(before - len(df_15))
        else:
            report["dropped_incomplete_m15"] = 0

        df_15 = df_15.reset_index()

        # Renommer en *_15m si tu veux coller au cahier des charges
        df_15 = df_15.rename(
            columns={
                "open": "open_15m",
                "high": "high_15m",
                "low": "low_15m",
                "close": "close_15m",
            }
        )

        return df_15, report