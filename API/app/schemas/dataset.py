# app/schemas/dataset.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class Meta(BaseModel):
    dataset_id: str


class LoadParams(BaseModel):
    year: int = Field(
        ...,
        ge=2022,
        le=2025,
        description="Année du fichier à charger (ex: 2022)"
    )


class LoadResult(BaseModel):
    file_path: str
    shape: Tuple[int, int]
    columns: List[str]
    sample: List[Dict[str, Any]]
    regularity: Dict[str, Any]


class BaseResponse(BaseModel):
    meta: Meta
    result: Any


# Schémas pour M15
class M15AggregateParams(BaseModel):
    dataset_id: str = Field(..., description="ID du dataset M1 à agréger")


class M15CleanParams(BaseModel):
    dataset_id: str = Field(..., description="ID du dataset M15 à nettoyer")
    gap_return_threshold: float = Field(0.02, ge=0.0, le=0.5, description="Seuil de gap en %")
    drop_gaps: bool = Field(True, description="Supprimer les gaps")


# Schémas pour ML Trading
class MLTrainParams(BaseModel):
    dataset_train_id: str = Field(..., description="Dataset 2022 avec features (phase=features)")
    dataset_val_id: str = Field(..., description="Dataset 2023 avec features")
    dataset_test_id: Optional[str] = Field(None, description="Dataset 2024 (optionnel, jamais utilisé pour entraîner)")
    model_type: str = Field("rf", description="logreg | rf | gbm")
    optimize: bool = Field(False, description="Activer Grid Search (plus lent)")


# Schémas pour RL
class RLTrainParams(BaseModel):
    dataset_train_id: str = Field(..., description="Dataset 2022 avec features")
    dataset_val_id: str = Field(..., description="Dataset 2023 pour évaluation")
    dataset_test_id: Optional[str] = Field(None, description="Dataset 2024 (évaluation finale uniquement)")
    n_episodes: int = Field(10, ge=1, le=100, description="Nombre d'épisodes d'entraînement")
    seed: int = Field(42, description="Seed aléatoire")
