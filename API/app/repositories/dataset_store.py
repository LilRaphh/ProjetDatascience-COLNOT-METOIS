from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class DatasetEntry:
    df: pd.DataFrame
    meta: Dict[str, Any] = field(default_factory=dict)


class DatasetStore:
    """
    Stockage en mémoire des datasets (DataFrame + meta).
    API attendue par tes routers:
    - put(dataset_id, df, meta)
    - get(dataset_id) -> DatasetEntry
    - exists(dataset_id)
    - delete(dataset_id)
    - list_ids()
    """

    def __init__(self) -> None:
        self._datasets: Dict[str, DatasetEntry] = {}

    def put(self, dataset_id: str, df: pd.DataFrame, meta: Optional[Dict[str, Any]] = None) -> None:
        self._datasets[dataset_id] = DatasetEntry(df=df, meta=meta or {})

    # Alias si tu veux aussi utiliser set()
    def set(self, dataset_id: str, df: pd.DataFrame, meta: Optional[Dict[str, Any]] = None) -> None:
        self.put(dataset_id=dataset_id, df=df, meta=meta)

    def get(self, dataset_id: str) -> DatasetEntry:
        return self._datasets[dataset_id]  # KeyError géré par ton router

    def exists(self, dataset_id: str) -> bool:
        return dataset_id in self._datasets

    def delete(self, dataset_id: str) -> None:
        if dataset_id in self._datasets:
            del self._datasets[dataset_id]

    def list_ids(self) -> list[str]:
        return list(self._datasets.keys())


dataset_store = DatasetStore()
