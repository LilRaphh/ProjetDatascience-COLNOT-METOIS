"""
Router pour le Nettoyage et la Preparation (phase clean).
Gere l'apprentissage et l'application de pipelines de nettoyage.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.repositories.dataset_store import dataset_store
from app.schemas.common import BaseResponse, ResponseMeta, ResponseReport
from app.schemas.clean import (
    CleanFitParams,
    CleanTransformParams,
    CleanFitResult,
    CleanTransformResult,
)
from app.schemas.common import BaseResponse, ResponseMeta, ResponseReport
from app.services.clean_service import CleanService

router = APIRouter(prefix="/clean", tags=["clean"])

clean_service = CleanService()

@router.post("/fit", response_model=BaseResponse)
async def clean_fit(dataset_id: str, params: CleanFitParams):
    """
    Apprend un pipeline de nettoyage sur les donnees brutes.

    NA = Not Available (valeurs manquantes)
    """
    try:
        entry = dataset_store.get(dataset_id)
        df = entry.df

        cleaner_id, rules, quality_before = clean_service.fit(df, params)

        result = CleanFitResult(
            cleaner_id=cleaner_id,
            dataset_id=dataset_id,
            rules=rules,
            quality_before=quality_before,
        )

        return BaseResponse(
            meta=ResponseMeta(dataset_id=dataset_id, status="success"),
            result=result.model_dump(),
            report=ResponseReport(
                message=f"Pipeline de nettoyage cree avec succes (ID: {cleaner_id})",
                metrics={
                    "n_rows": quality_before.n_rows,
                    "n_duplicates": quality_before.n_duplicates,
                    "total_missing": sum(quality_before.missing_values.values()),
                    "total_outliers": sum(quality_before.outliers_count.values()),
                },
            ),
        )

    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'apprentissage: {str(e)}")


@router.post("/transform", response_model=BaseResponse)
async def clean_transform(dataset_id: str, params: CleanTransformParams):
    """
    Applique un pipeline de nettoyage appris aux donnees.
    """
    try:
        entry = dataset_store.get(dataset_id)
        df = entry.df

        df_clean, counters = clean_service.transform(df, params.cleaner_id)

        processed_dataset_id = f"clean_{params.cleaner_id}_{uuid.uuid4().hex[:8]}"

        # Stockage du dataset nettoye + meta
        dataset_store.put(
            dataset_id=processed_dataset_id,
            df=df_clean,
            meta={
                "phase": "clean",
                "source_dataset_id": dataset_id,
                "cleaner_id": params.cleaner_id,
                "counters": counters,
                "parent_meta": entry.meta,
            },
        )

        sample_size = min(20, len(df_clean))
        data_sample = df_clean.head(sample_size).to_dict(orient="records")

        result = CleanTransformResult(
            processed_dataset_id=processed_dataset_id,
            n_rows_before=len(df),
            n_rows_after=len(df_clean),
            counters=counters,
            data_sample=data_sample,
        )

        return BaseResponse(
            meta=ResponseMeta(dataset_id=dataset_id, status="success"),
            result=result.model_dump(),
            report=ResponseReport(
                message="Nettoyage applique avec succes",
                metrics={
                    "rows_removed": len(df) - len(df_clean),
                    "total_imputations": counters.get("missing_imputed_numeric", 0)
                    + counters.get("missing_imputed_categorical", 0),
                    "duplicates_removed": counters.get("duplicates_removed", 0),
                },
            ),
        )

    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Cleaner {params.cleaner_id} non trouve")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la transformation: {str(e)}")


@router.get("/report/{dataset_id}", response_model=BaseResponse)
async def clean_report(dataset_id: str):
    """
    Genere un rapport de qualite des donnees sans transformation.
    """
    try:
        entry = dataset_store.get(dataset_id)
        df = entry.df

        quality_report = clean_service.generate_quality_report(df)

        warnings = []
        if quality_report.n_duplicates > 0:
            warnings.append(f"Doublons detectes: {quality_report.n_duplicates}")
        total_missing = sum(quality_report.missing_values.values())
        if total_missing > 0:
            warnings.append(f"Valeurs manquantes totales: {total_missing}")
        total_outliers = sum(quality_report.outliers_count.values())
        if total_outliers > 0:
            warnings.append(f"Outliers detectes: {total_outliers}")

        return BaseResponse(
            meta=ResponseMeta(dataset_id=dataset_id, status="success"),
            result=quality_report.model_dump(),
            report=ResponseReport(
                message="Rapport de qualite genere avec succes",
                warnings=warnings,
            ),
        )

    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la generation du rapport: {str(e)}")
