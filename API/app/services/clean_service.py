from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import uuid

import numpy as np
import pandas as pd

from app.schemas.clean import QualityReport, CleanFitParams


# ============================================================
# ======================= STRUCTURE FIT ======================
# ============================================================

@dataclass
class _FittedCleaner:
    numeric_impute_values: Dict[str, float]
    outlier_bounds: Dict[str, Tuple[float, float]]
    params: Dict[str, Any]


# ============================================================
# ======================= CLEAN SERVICE ======================
# ============================================================

class CleanService:
    """
    Service de nettoyage générique:
    - Rapport qualité
    - Fit: apprend règles d'imputation + bornes IQR
    - Transform: applique suppression doublons, imputation, gestion outliers
    """

    def __init__(self) -> None:
        self._cleaners: Dict[str, _FittedCleaner] = {}

    # ============================================================
    # =================== QUALITY REPORT ==========================
    # ============================================================

    def generate_quality_report(self, df: pd.DataFrame) -> QualityReport:
        if df is None or len(df) == 0:
            return QualityReport(
                n_rows=0,
                n_duplicates=0,
                missing_values={},
                outliers_count={},
            )

        n_rows = int(len(df))
        n_duplicates = int(df.duplicated().sum())
        missing_values = {c: int(df[c].isna().sum()) for c in df.columns}

        outliers_count: Dict[str, int] = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for c in numeric_cols:
            s = df[c].dropna()
            if len(s) < 10:
                outliers_count[c] = 0
                continue

            q1 = float(s.quantile(0.25))
            q3 = float(s.quantile(0.75))
            iqr = q3 - q1

            if iqr == 0:
                outliers_count[c] = 0
                continue

            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            outliers_count[c] = int(((df[c] < low) | (df[c] > high)).sum())

        return QualityReport(
            n_rows=n_rows,
            n_duplicates=n_duplicates,
            missing_values=missing_values,
            outliers_count=outliers_count,
        )

    # ============================================================
    # ============================ FIT ============================
    # ============================================================

    def fit(self, df: pd.DataFrame, params: CleanFitParams):

        if df is None or len(df) == 0:
            raise ValueError("Impossible de fitter un DataFrame vide.")

        quality_before = self.generate_quality_report(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # --- Imputation ---
        numeric_impute_values: Dict[str, float] = {}
        for c in numeric_cols:
            s = df[c].dropna()
            if len(s) == 0:
                numeric_impute_values[c] = 0.0
            elif params.impute_numeric == "mean":
                numeric_impute_values[c] = float(s.mean())
            else:  # median par défaut
                numeric_impute_values[c] = float(s.median())

        # --- Bornes outliers IQR ---
        outlier_bounds: Dict[str, Tuple[float, float]] = {}

        for c in numeric_cols:
            s = df[c].dropna()

            if len(s) < 10:
                outlier_bounds[c] = (-np.inf, np.inf)
                continue

            q1 = float(s.quantile(0.25))
            q3 = float(s.quantile(0.75))
            iqr = q3 - q1

            if iqr == 0:
                outlier_bounds[c] = (-np.inf, np.inf)
            else:
                low = q1 - 1.5 * iqr
                high = q3 + 1.5 * iqr
                outlier_bounds[c] = (low, high)

        cleaner_id = uuid.uuid4().hex[:8]

        self._cleaners[cleaner_id] = _FittedCleaner(
            numeric_impute_values=numeric_impute_values,
            outlier_bounds=outlier_bounds,
            params=params.model_dump(),
        )

        rules = {
            "numeric_impute_values": numeric_impute_values,
            "outlier_bounds": {k: [v[0], v[1]] for k, v in outlier_bounds.items()},
            "params": params.model_dump(),
        }

        return cleaner_id, rules, quality_before

    # ============================================================
    # ========================== TRANSFORM =======================
    # ============================================================

    def transform(self, df: pd.DataFrame, cleaner_id: str):

        if cleaner_id not in self._cleaners:
            raise FileNotFoundError(f"Cleaner introuvable: {cleaner_id}")

        if df is None or len(df) == 0:
            raise ValueError("Impossible de transformer un DataFrame vide.")

        fitted = self._cleaners[cleaner_id]
        params = fitted.params

        df2 = df.copy()
        counters: Dict[str, int] = {}

        # ------------------------------------------------------------
        # 1) Suppression doublons
        # ------------------------------------------------------------
        before = len(df2)
        df2 = df2.drop_duplicates()
        counters["duplicates_removed"] = int(before - len(df2))

        # ------------------------------------------------------------
        # 2) Imputation numérique
        # ------------------------------------------------------------
        numeric_cols = df2.select_dtypes(include=[np.number]).columns.tolist()

        missing_imputed = 0
        for c in numeric_cols:
            na = df2[c].isna().sum()
            if na > 0:
                df2[c] = df2[c].fillna(
                    fitted.numeric_impute_values.get(c, 0.0)
                )
                missing_imputed += int(na)

        counters["missing_imputed_numeric"] = missing_imputed

        # ------------------------------------------------------------
        # 3) Outliers
        # ------------------------------------------------------------
        outlier_strategy = params.get("outlier_strategy", "clip")

        if outlier_strategy == "remove":
            mask = pd.Series(True, index=df2.index)

            for c in numeric_cols:
                low, high = fitted.outlier_bounds.get(c, (-np.inf, np.inf))
                mask &= df2[c].between(low, high) | df2[c].isna()

            removed = int((~mask).sum())
            df2 = df2.loc[mask].copy()

            counters["outliers_removed_rows"] = removed

        else:  # clip par défaut
            clipped = 0
            for c in numeric_cols:
                low, high = fitted.outlier_bounds.get(c, (-np.inf, np.inf))
                before_vals = df2[c].copy()
                df2[c] = df2[c].clip(lower=low, upper=high)
                clipped += int((before_vals != df2[c]).sum())

            counters["outliers_clipped_values"] = clipped

        # ------------------------------------------------------------
        # Rapport qualité après nettoyage
        # ------------------------------------------------------------
        quality_after = self.generate_quality_report(df2)

        return df2.reset_index(drop=True), counters, quality_after


# ============================================================
# ================= INSTANCE GLOBALE =========================
# ============================================================

clean_service = CleanService()
