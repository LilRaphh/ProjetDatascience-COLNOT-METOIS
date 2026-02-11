# app/schemas/dataset.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field


class Meta(BaseModel):
    dataset_id: str


class LoadParams(BaseModel):
    year: int = Field(
        ...,
        ge=2022,
        le=2025,
        description="Année du fichier à charger (ex: 2022)",
    )
    pair: str = Field("GBPUSD")
    timeframe: str = Field("M1")


class LoadResult(BaseModel):
    file_path: str
    shape: Tuple[int, int]
    columns: List[str]
    sample: List[Dict[str, Any]]
    regularity: Dict[str, Any]


class BaseResponse(BaseModel):
    meta: Meta
    result: Any
