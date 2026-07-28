"""Central configuration: paths and NCBI credentials, all env-overridable."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("GEOTOOL_DATA_DIR", PROJECT_ROOT / "data"))

GEO_CACHE_DIR = DATA_DIR / "geo_cache"
REPORTS_DIR = DATA_DIR / "reports"
SERIES_DIR = DATA_DIR / "series"

NCBI_EMAIL = os.environ.get("GEOTOOL_NCBI_EMAIL", "kit.iz.179@gmail.com")
NCBI_API_KEY = os.environ.get("GEOTOOL_NCBI_API_KEY")  # None -> 3 req/s throttle

# NCBI E-utilities rate limit: 3 req/s without an API key, 10 req/s with one.
NCBI_REQUESTS_PER_SECOND = 10 if NCBI_API_KEY else 3


def ensure_dirs() -> None:
    for d in (GEO_CACHE_DIR, REPORTS_DIR, SERIES_DIR):
        d.mkdir(parents=True, exist_ok=True)
