# src/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "ml-latest-small"
DATA_PROC = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

SEED = 42
REL_THRESHOLD = 4.0   # a rating >= 4.0 counts as "relevant" for ranking metrics
TEST_FRAC = 0.20      # last 20% of each user's ratings (by time) -> test
VAL_FRAC = 0.10       # last 10% of the remaining -> validation
TOP_K = 10
RATING_MIN, RATING_MAX = 0.5, 5.0

for d in (DATA_PROC, ARTIFACTS, REPORTS):
    d.mkdir(parents=True, exist_ok=True)
