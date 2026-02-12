# app/routers/dataset.py
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

from app.schemas.dataset import BaseResponse, LoadParams, LoadResult, Meta
from app.services.m1_import_service import M1ImportService
from app.repositories.dataset_store import dataset_store

router = APIRouter(prefix="/dataset", tags=["dataset"])

m1_import_service = M1ImportService()

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _get_entry_or_404(dataset_id: str):
    try:
        return dataset_store.get(dataset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"dataset_id not found: {dataset_id}")


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@router.post("/load_m1", response_model=BaseResponse)
def load_m1(request: LoadParams) -> BaseResponse:
    """
    Charge un CSV M1 brut (RAW) + ajoute timestamp.
    Stocke le DataFrame en mémoire via dataset_store.
    """
    csv_path = m1_import_service.resolve_csv_path(request.year)
    df = m1_import_service.load_raw_m1_csv(csv_path)

    dataset_id = m1_import_service.make_dataset_id(prefix=f"m1_{request.year}")
    reg = m1_import_service.regularity_report(df)

    dataset_store.put(
        dataset_id=dataset_id,
        df=df,
        meta={
            "phase": "m1_raw",
            "year": request.year,
            "file_path": str(csv_path),
            "raw": True,
            "regularity": reg,
        },
    )

    result = LoadResult(
        file_path=str(csv_path),
        shape=(int(df.shape[0]), int(df.shape[1])),
        columns=list(df.columns),
        sample=df.head(20).to_dict(orient="records"),
        regularity=reg,
    )
    return BaseResponse(meta=Meta(dataset_id=dataset_id), result=result.model_dump())


@router.get("/{dataset_id}/info", response_model=BaseResponse)
def dataset_info(dataset_id: str) -> BaseResponse:
    """
    Infos sur le dataset : meta, shape, colonnes, null_counts, rapport timestamp.
    """
    entry = _get_entry_or_404(dataset_id)
    df = entry.df

    reg = m1_import_service.regularity_report(df) if "timestamp" in df.columns else {"has_timestamp": False}

    info: Dict[str, Any] = {
        "meta": entry.meta,
        "shape": (int(df.shape[0]), int(df.shape[1])),
        "columns": list(df.columns),
        "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
        "timestamp_report": reg,
    }
    return BaseResponse(meta=Meta(dataset_id=dataset_id), result=info)


@router.get("/{dataset_id}/sample", response_model=BaseResponse)
def dataset_sample(dataset_id: str, n: int = Query(20, ge=1, le=500)) -> BaseResponse:
    """
    Retourne un échantillon des n premières lignes.
    """
    entry = _get_entry_or_404(dataset_id)
    df = entry.df
    return BaseResponse(
        meta=Meta(dataset_id=dataset_id),
        result={"n": int(n), "data": df.head(int(n)).to_dict(orient="records")},
    )


@router.get("/list", response_model=BaseResponse)
def dataset_list(limit: int = Query(50, ge=1, le=500)) -> BaseResponse:
    """
    Liste rapide des datasets en mémoire.
    (utile pour debug)
    """
    # dataset_store n'expose pas forcément list() : on le fait simple.
    # Si tu veux, on ajoute dataset_store.list_ids() proprement.
    try:
        store_dict = dataset_store.__dict__.get("_datasets", {})
        ids = list(store_dict.keys())[: int(limit)]
        summary = []
        for did in ids:
            entry = store_dict[did]
            df = entry.df
            summary.append(
                {
                    "dataset_id": did,
                    "shape": (int(df.shape[0]), int(df.shape[1])),
                    "phase": entry.meta.get("phase"),
                    "year": entry.meta.get("year"),
                }
            )

        return BaseResponse(
            meta=Meta(dataset_id="store"),
            result={"count": len(store_dict), "returned": len(summary), "datasets": summary},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur list store: {str(e)}")
