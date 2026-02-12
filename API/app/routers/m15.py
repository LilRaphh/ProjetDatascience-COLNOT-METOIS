import traceback
from fastapi import APIRouter, HTTPException, Query

from app.services.m15_clean_service import M15CleanService


from app.repositories.dataset_store import dataset_store
from app.services.m15_aggregation_service import M15AggregationService

router = APIRouter(prefix="/m15", tags=["M15"])

m15_aggregation_service = M15AggregationService()
m15_clean_service = M15CleanService()

@router.post("/aggregate")
def aggregate_m1_to_m15(dataset_id: str):
    try:
        if not dataset_store.exists(dataset_id):
            raise HTTPException(status_code=404, detail="Dataset introuvable")

        entry = dataset_store.get(dataset_id)
        df_m1 = entry.df

        df_15m, report = m15_aggregation_service.aggregate(df_m1)

        new_id = f"{dataset_id}_m15"
        dataset_store.put(
            dataset_id=new_id,
            df=df_15m,
            meta={**(entry.meta or {}), "phase": "m15", "m15_report": report},
        )

        return {"dataset_id": new_id, "n_rows": int(len(df_15m)), "columns": list(df_15m.columns), "report": report}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"m15 aggregate failed: {repr(e)}")



@router.post("/clean")
def clean_m15(
    dataset_id: str,
    gap_return_threshold: float = Query(0.02, ge=0.0, le=0.5),
    drop_gaps: bool = Query(True),
):
    try:
        if not dataset_store.exists(dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")

        entry = dataset_store.get(dataset_id)
        df = entry.df

        df_clean, report = m15_clean_service.clean(
            df_m15=df,
            gap_return_threshold=gap_return_threshold,
            drop_gaps=drop_gaps,
        )

        new_id = f"{dataset_id}_clean"
        dataset_store.put(
            dataset_id=new_id,
            df=df_clean,
            meta={**(entry.meta or {}), "phase": "m15_clean", "m15_clean_report": report},
        )

        return {
            "dataset_id": new_id,
            "n_rows": int(len(df_clean)),
            "columns": list(df_clean.columns),
            "report": report,
        }

    except HTTPException:
        raise
    except (ValueError, KeyError) as e:
        # Erreur utilisateur (mauvais dataset, colonnes manquantes)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"m15 clean failed: {repr(e)}")



@router.get("/report")
def report_m15(dataset_id: str):
    if not dataset_store.exists(dataset_id):
        raise HTTPException(status_code=404, detail=f"Dataset introuvable: {dataset_id}")

    entry = dataset_store.get(dataset_id)
    return {
        "dataset_id": dataset_id,
        "meta": entry.meta,
    }
