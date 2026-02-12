# app/schemas/features.py
from __future__ import annotations

from pydantic import BaseModel, Field


class ComputeFeaturesParams(BaseModel):
    dataset_id: str = Field(..., description="ID du dataset M15 nettoyé")
    drop_na: bool = Field(True, description="Supprimer les lignes de warm-up (NaN)")
    add_target: bool = Field(True, description="Ajouter la colonne target y=1 si close_{t+1}>close_t")
