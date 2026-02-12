from pathlib import Path
import os

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

EXPECTED_COLS = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
NUMERIC_COLS = ["Open", "High", "Low", "Close", "Volume"]

DATETIME_FORMAT = "%Y.%m.%d %H:%M"
REGULARITY_PCT_THRESHOLD = 0.95
