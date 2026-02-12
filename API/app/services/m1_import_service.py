from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import HTTPException

from app.config import (
    DATA_DIR, EXPECTED_COLS, NUMERIC_COLS,
    DATETIME_FORMAT, REGULARITY_PCT_THRESHOLD
)

from pathlib import Path
from app.config import settings  # si tu as un Settings Pydantic

def resolve_csv_path(year: int) -> Path:
    filename = settings.file_pattern.format(pair=settings.pair, timeframe=settings.timeframe, year=year)
    return Path(settings.data_dir) / filename


class M1ImportService:
    @staticmethod
    def make_dataset_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def resolve_csv_path(year: int) -> Path:
        if not DATA_DIR.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Dossier data introuvable: {DATA_DIR.resolve()}",
            )

        files = sorted(DATA_DIR.glob(f"*{year}*.csv"))
        if not files:
            raise HTTPException(
                status_code=404,
                detail=f"Aucun fichier CSV trouvé dans '{DATA_DIR.resolve()}' pour l'année {year}",
            )
        return files[0]

    @staticmethod
    def load_raw_m1_csv(csv_path: Path) -> pd.DataFrame:
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"Fichier introuvable : {csv_path}")

        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip() for c in df.columns]

        # fallback si pas de header
        if not set(EXPECTED_COLS).issubset(set(df.columns)):
            df = pd.read_csv(csv_path, header=None, names=EXPECTED_COLS)

        df.columns = [str(c).strip() for c in df.columns]

        missing = [c for c in EXPECTED_COLS if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Colonnes manquantes dans le CSV: {missing}. Colonnes trouvées: {list(df.columns)}",
            )

        for c in NUMERIC_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        ts = pd.to_datetime(
            df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
            format=DATETIME_FORMAT,
            errors="coerce",
        )
        df.insert(0, "timestamp", ts)

        return df

    @staticmethod
    def regularity_report(df: pd.DataFrame) -> Dict[str, Any]:
        if "timestamp" not in df.columns:
            return {"has_timestamp": False}

        s = df["timestamp"].dropna().sort_values()
        if len(s) < 2:
            return {"has_timestamp": True, "is_regular_1min": False}

        dt = s.diff().dropna()
        seconds = dt.dt.total_seconds().astype("int64")
        pct_60 = float((seconds == 60).mean())

        return {
            "has_timestamp": True,
            "min_ts": s.min().isoformat(),
            "max_ts": s.max().isoformat(),
            "pct_exact_60s": pct_60,
            "is_regular_1min": bool(pct_60 >= REGULARITY_PCT_THRESHOLD),
        }

m1_import_service = M1ImportService()
