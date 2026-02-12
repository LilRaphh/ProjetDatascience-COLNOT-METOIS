from __future__ import annotations

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


# ============================================================
# ===================== PARAMETRES FIT =======================
# ============================================================

class CleanFitParams(BaseModel):
    """
    Parametres d'apprentissage du pipeline de nettoyage.
    """
    impute_numeric: str = Field(default="median", description="mean | median")
    impute_categorical: str = Field(default="mode", description="mode")
    outlier_strategy: str = Field(default="clip", description="clip | remove")


# ============================================================
# ================= PARAMETRES TRANSFORM =====================
# ============================================================

class CleanTransformParams(BaseModel):
    """
    Parametres pour appliquer un pipeline appris.
    """
    cleaner_id: str


# ============================================================
# ====================== RESULTATS FIT =======================
# ============================================================

class QualityReport(BaseModel):
    n_rows: int
    n_duplicates: int
    missing_values: Dict[str, int]
    outliers_count: Dict[str, int]


class CleanFitResult(BaseModel):
    cleaner_id: str
    dataset_id: str
    rules: Dict[str, Any]
    quality_before: QualityReport


# ============================================================
# ==================== RESULTATS TRANSFORM ===================
# ============================================================

class CleanTransformResult(BaseModel):
    processed_dataset_id: str
    n_rows_before: int
    n_rows_after: int
    counters: Dict[str, int]
    data_sample: list[Dict[str, Any]]
