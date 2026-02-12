# app/routers/dataset.py
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.schemas.dataset import BaseResponse, LoadParams, LoadResult, Meta

from app.services.dataset import _DATASETS, _DATASETS_META, _make_dataset_id, _resolve_csv_path, _load_raw_m1_csv, _regularity_report

router = APIRouter(prefix="/dataset", tags=["dataset"])

# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@router.post("/load_m1", response_model=BaseResponse)
def load_m1(request: LoadParams) -> BaseResponse:
    csv_path = _resolve_csv_path(request.year)
    df = _load_raw_m1_csv(csv_path)

    dataset_id = _make_dataset_id(f"m1_{request.year}")

    _DATASETS[dataset_id] = df
    _DATASETS_META[dataset_id] = {
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

    return BaseResponse(meta=Meta(dataset_id=dataset_id), result=result.model_dump())


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
def dataset_sample(dataset_id: str, n: int = Query(20, ge=1, le=500)) -> BaseResponse:
    if dataset_id not in _DATASETS:
        raise HTTPException(status_code=404, detail="dataset_id not found")

    df = _DATASETS[dataset_id]
    return BaseResponse(meta=Meta(dataset_id=dataset_id), result={"n": n, "data": df.head(n).to_dict(orient="records")})
