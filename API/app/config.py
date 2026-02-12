from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


# =============================================================================
# Paths / Environnement
# =============================================================================

DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data")).resolve()

# (Optionnel) dossier outputs
OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "outputs")).resolve()


# =============================================================================
# Format et validation des CSV M1
# =============================================================================

EXPECTED_COLS: List[str] = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
NUMERIC_COLS: List[str] = ["Open", "High", "Low", "Close", "Volume"]

DATETIME_FORMAT: str = os.getenv("DATETIME_FORMAT", "%Y.%m.%d %H:%M")
REGULARITY_PCT_THRESHOLD: float = float(os.getenv("REGULARITY_PCT_THRESHOLD", "0.95"))
MAX_ALLOWED_GAP_MINUTES: int = int(os.getenv("MAX_ALLOWED_GAP_MINUTES", "5"))


# =============================================================================
# Settings applicatifs
# =============================================================================

@dataclass(frozen=True)
class Settings:
    pair: str = os.getenv("PAIR", "GBPUSD")
    timeframe: str = os.getenv("TIMEFRAME", "M1")

    data_dir: str = os.getenv("DATA_DIR", "data")
    file_pattern: str = os.getenv("FILE_PATTERN", "DAT_MT_{pair}_{timeframe}_{year}.csv")

    sep: str = os.getenv("CSV_SEP", ",")
    encoding: Optional[str] = os.getenv("CSV_ENCODING")  # None si non défini

    date_col: str = os.getenv("DATE_COL", "Date")
    time_col: str = os.getenv("TIME_COL", "Time")

    tz: str = os.getenv("TZ", "UTC")

    sort: bool = os.getenv("SORT", "true").lower() in ("1", "true", "yes", "y")


settings = Settings()


# =============================================================================
# Helpers
# =============================================================================

def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_csv_path(year: int) -> Path:
    filename = settings.file_pattern.format(
        pair=settings.pair,
        timeframe=settings.timeframe,
        year=year,
    )
    return (Path(settings.data_dir) / filename).resolve()
