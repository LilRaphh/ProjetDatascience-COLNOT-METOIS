# app/routers/dataset.py
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.schemas.dataset import BaseResponse, LoadParams, LoadResult, Meta

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
EXPECTED_COLS = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]

# ---------------------------------------------------------------------
# Stockage mémoire
# ---------------------------------------------------------------------
_DATASETS: Dict[str, pd.DataFrame] = {}
_DATASETS_META: Dict[str, Dict[str, Any]] = {}


def _make_dataset_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _resolve_csv_path(year: int) -> Path:
    """
    Retourne le premier fichier CSV dans DATA_DIR contenant l'année dans son nom.
    Exemple: *2022*.csv
    """
    if not DATA_DIR.exists():
        raise HTTPException(status_code=404, detail=f"Dossier data introuvable: {DATA_DIR.resolve()}")

    files = sorted(DATA_DIR.glob(f"*{year}*.csv"))

    if not files:
        raise HTTPException(
            status_code=404,
            detail=f"Aucun fichier CSV trouvé dans '{DATA_DIR.resolve()}' pour l'année {year}",
        )

    return files[0]


def _load_raw_m1_csv(csv_path: Path) -> pd.DataFrame:
    """
    Chargement RAW :
    - Aucun nettoyage
    - Gestion automatique si CSV sans header
    - Ajout colonne timestamp
    """
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"Fichier introuvable : {csv_path}")

    # Tentative normale (header présent)
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    # Si pas de header => la 1ère ligne est prise comme colonnes
    if not set(EXPECTED_COLS).issubset(set(df.columns)):
        df = pd.read_csv(csv_path, header=None, names=EXPECTED_COLS)

    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Colonnes manquantes dans le CSV: {missing}. Colonnes trouvées: {list(df.columns)}",
        )

    # Cast numérique (sans suppression de NaN)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ajout timestamp
    ts = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
        format="%Y.%m.%d %H:%M",
        errors="coerce",
    )
    df.insert(0, "timestamp", ts)

    return df


def _regularity_report(df: pd.DataFrame) -> Dict[str, Any]:
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
        "is_regular_1min": bool(pct_60 >= 0.95),
    }