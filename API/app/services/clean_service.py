from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import uuid

import numpy as np
import pandas as pd

from app.schemas.clean import QualityReport, CleanFitParams


@dataclass
class _FittedCleaner:
    numeric_impute_values: Dict[str, float]
    outlier_bounds: Dict[str, Tuple[float, float]]
    params: Dict[str, Any]


class CleanService:
    """
    Service de nettoyage simple:
    - rapport qualité (NA, doublons, outliers)
    - fit: apprend valeurs d'imputation + bornes outliers
    - transform: applique doublons, imputation, clipping/removal
    """

    def __init__(self) -> None:
        self._cleaners: Dict[str, _FittedCleaner] = {}

    def generate_quality_report(self, df: pd.DataFrame) -> QualityReport:
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

    def fit(self, df: pd.DataFrame, params: CleanFitParams):
        quality_before = self.generate_quality_report(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_impute_values: Dict[str, float] = {}

        for c in numeric_cols:
            s = df[c].dropna()
            if len(s) == 0:
                numeric_impute_values[c] = 0.0
            elif params.impute_numeric == "mean":
                numeric_impute_values[c] = float(s.mean())
            else:  # median
                numeric_impute_values[c] = float(s.median())

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

    def transform(self, df: pd.DataFrame, cleaner_id: str):
        if cleaner_id not in self._cleaners:
            raise FileNotFoundError(cleaner_id)

        fitted = self._cleaners[cleaner_id]
        params = fitted.params

        counters: Dict[str, int] = {}

        # 1) Doublons
        before = len(df)
        df2 = df.drop_duplicates().copy()
        counters["duplicates_removed"] = int(before - len(df2))

        # 2) Imputation numérique
        numeric_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
        missing_imputed = 0
        for c in numeric_cols:
            na = df2[c].isna().sum()
            if na > 0:
                df2[c] = df2[c].fillna(fitted.numeric_impute_values.get(c, 0.0))
                missing_imputed += int(na)
        counters["missing_imputed_numeric"] = missing_imputed

        # 3) Outliers
        outlier_strategy = params.get("outlier_strategy", "clip")
        if outlier_strategy == "remove":
            mask = pd.Series(True, index=df2.index)
            removed = 0
            for c in numeric_cols:
                low, high = fitted.outlier_bounds.get(c, (-np.inf, np.inf))
                m = df2[c].between(low, high) | df2[c].isna()
                mask &= m
            removed = int((~mask).sum())
            df2 = df2.loc[mask].copy()
            counters["outliers_removed_rows"] = removed
        else:  # clip
            clipped = 0
            for c in numeric_cols:
                low, high = fitted.outlier_bounds.get(c, (-np.inf, np.inf))
                before_vals = df2[c].copy()
                df2[c] = df2[c].clip(lower=low, upper=high)
                clipped += int((before_vals != df2[c]).sum())
            counters["outliers_clipped_values"] = clipped

        return df2, counters


clean_service = CleanService()
