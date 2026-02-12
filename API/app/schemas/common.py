from __future__ import annotations

from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field


class ResponseMeta(BaseModel):
    dataset_id: str
    status: str = Field(default="success")


class ResponseReport(BaseModel):
    message: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class BaseResponse(BaseModel):
    meta: ResponseMeta
    result: Any
    report: Optional[ResponseReport] = None

# ============================================================================
# SCHÉMAS DE BASE (ENVELOPPE STANDARD)
# ============================================================================

class MetaData(BaseModel):
    """
    Métadonnées présentes dans toutes les requêtes/réponses
    Permet le traçage et la versionnement
    """
    dataset_id: Optional[str] = Field(None, description="Identifiant unique du dataset")
    schema_version: str = Field(default="1.0", description="Version du schéma de données")
    
    class Config:
        json_schema_extra = {
            "example": {
                "dataset_id": "clean_42_1000",
                "schema_version": "1.0"
            }
        }

class StandardRequest(BaseModel):
    """
    Structure standard pour toutes les requêtes
    - meta : métadonnées
    - data : données (optionnel)
    - params : paramètres spécifiques à l'endpoint
    """
    meta: MetaData
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Données d'entrée (records)")
    params: Optional[Dict[str, Any]] = Field(None, description="Paramètres spécifiques")
    
    class Config:
        json_schema_extra = {
            "example": {
                "meta": {
                    "dataset_id": "clean_42_1000",
                    "schema_version": "1.0"
                },
                "params": {
                    "impute_strategy": "mean"
                }
            }
        }

class StandardResponse(BaseModel):
    """
    Structure standard pour toutes les réponses
    - meta : métadonnées
    - result : résultat principal
    - report : rapport/statistiques
    - artifacts : artefacts (graphiques, modèles, etc.)
    """
    meta: MetaData
    result: Optional[Dict[str, Any]] = Field(None, description="Résultat principal")
    report: Optional[Dict[str, Any]] = Field(None, description="Rapport/statistiques")
    artifacts: Optional[Dict[str, Any]] = Field(None, description="Artefacts (graphiques, etc.)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "meta": {
                    "dataset_id": "clean_42_1000",
                    "schema_version": "1.0"
                },
                "result": {
                    "status": "success"
                },
                "report": {
                    "missing_values": 150,
                    "duplicates": 25
                }
            }
        }