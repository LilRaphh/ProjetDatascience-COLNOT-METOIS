from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import pandas as pd

@dataclass
class DatasetEntry:
    df: pd.DataFrame
    meta: Dict[str, Any]

class DatasetStore:
    def __init__(self) -> None:
        self._datasets: Dict[str, DatasetEntry] = {}

    def put(self, dataset_id: str, df: pd.DataFrame, meta: Dict[str, Any]) -> None:
        self._datasets[dataset_id] = DatasetEntry(df=df, meta=meta)

    def get(self, dataset_id: str) -> DatasetEntry:
        if dataset_id not in self._datasets:
            raise KeyError(dataset_id)
        return self._datasets[dataset_id]

    def exists(self, dataset_id: str) -> bool:
        return dataset_id in self._datasets

# Instance globale simple (comme tu faisais avec des dicts)
dataset_store = DatasetStore()
