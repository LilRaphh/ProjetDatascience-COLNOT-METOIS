# app/routers/dataset.py
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.schemas.dataset import BaseResponse, LoadParams, LoadResult, Meta

router = APIRouter(prefix="/dataset", tags=["dataset"])

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
FILENAME_PATTERN = "DAT_MT_{pair}_{timeframe}_{year}.csv"

EXPECTED_COLS = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]

# ---------------------------------------------------------------------
# Stockage mémoire
# ---------------------------------------------------------------------
_DATASETS: Dict[str, pd.DataFrame] = {}
_DATASETS_META: Dict[str, Dict[str, Any]] = {}


def _make_dataset_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _resolve_csv_path(pair: str, timeframe: str, year: int) -> Path:
    filename = FILENAME_PATTERN.format(
        pair=pair.upper(),
        timeframe=timeframe.upper(),
        year=year
    )
    return DATA_DIR / filename


def _load_raw_m1_csv(csv_path: Path) -> pd.DataFrame:
    """
    Chargement RAW :
    - Aucun nettoyage
    - Gestion automatique si CSV sans header
    - Ajout colonne timestamp
    """

    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Fichier introuvable : {csv_path}"
        )

    # Tentative normale
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    # Si les colonnes attendues ne sont pas présentes -> CSV sans header
    if not set(EXPECTED_COLS).issubset(set(df.columns)):
        df = pd.read_csv(csv_path, header=None, names=EXPECTED_COLS)

    # Sécurisation noms
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
        df["Date"].astype(str).str.strip()
        + " "
        + df["Time"].astype(str).str.strip(),
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

# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@router.post("/load_m1", response_model=BaseResponse)
def load_m1(request: LoadParams) -> BaseResponse:

    csv_path = _resolve_csv_path(request.pair, request.timeframe, request.year)
    df = _load_raw_m1_csv(csv_path)

    dataset_id = _make_dataset_id(
        f"m1_{request.pair.upper()}_{request.year}"
    )

    _DATASETS[dataset_id] = df
    _DATASETS_META[dataset_id] = {
        "pair": request.pair.upper(),
        "timeframe": request.timeframe.upper(),
        "year": request.year,
        "file_path": str(csv_path),
        "raw": True,
    }

    result = LoadResult(
        file_path=str(csv_path),
        shape=(int(df.shape[0]), int(df.shape[1])),
        columns=list(df.columns),
        sample=df.head(20).to_dict(orient="records"),
        regularity=_regularity_report(df),
    )

    return BaseResponse(
        meta=Meta(dataset_id=dataset_id),
        result=result.model_dump(),
    )


@router.get("/{dataset_id}/info", response_model=BaseResponse)
def dataset_info(dataset_id: str) -> BaseResponse:

    if dataset_id not in _DATASETS:
        raise HTTPException(status_code=404, detail="dataset_id not found")

    df = _DATASETS[dataset_id]

    info = {
        "meta": _DATASETS_META.get(dataset_id, {}),
        "shape": (int(df.shape[0]), int(df.shape[1])),
        "columns": list(df.columns),
        "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
        "timestamp_report": _regularity_report(df),
    }

    return BaseResponse(meta=Meta(dataset_id=dataset_id), result=info)


@router.get("/{dataset_id}/sample", response_model=BaseResponse)
def dataset_sample(
    dataset_id: str,
    n: int = Query(20, ge=1, le=500)
) -> BaseResponse:

    if dataset_id not in _DATASETS:
        raise HTTPException(status_code=404, detail="dataset_id not found")

    df = _DATASETS[dataset_id]

    return BaseResponse(
        meta=Meta(dataset_id=dataset_id),
        result={"n": n, "data": df.head(n).to_dict(orient="records")},
    )
