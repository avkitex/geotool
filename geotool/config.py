"""Central configuration: paths and NCBI credentials, all env-overridable."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("GEOTOOL_DATA_DIR", PROJECT_ROOT / "data"))

GEO_CACHE_DIR = DATA_DIR / "geo_cache"
REPORTS_DIR = DATA_DIR / "reports"
SERIES_DIR = DATA_DIR / "series"
PLATFORMS_DIR = DATA_DIR / "platforms"
# GENCODE transcript/gene -> HUGO symbol reference tables (see
# data/references/build_gencode_reference.py) -- not created by ensure_dirs,
# since these are curated inputs, not an output directory geotool writes to.
REFERENCES_DIR = DATA_DIR / "references"

NCBI_EMAIL = os.environ.get("GEOTOOL_NCBI_EMAIL", "kit.iz.179@gmail.com")
NCBI_API_KEY = os.environ.get("GEOTOOL_NCBI_API_KEY")  # None -> 3 req/s throttle

# NCBI E-utilities rate limit: 3 req/s without an API key, 10 req/s with one.
NCBI_REQUESTS_PER_SECOND = 10 if NCBI_API_KEY else 3

# LLM annotation (ANTHROPIC_API_KEY is read by the SDK itself, no handling needed here).
LLM_MODEL = os.environ.get("GEOTOOL_LLM_MODEL", "claude-haiku-4-5")
LLM_ESCALATION_MODEL = os.environ.get("GEOTOOL_LLM_ESCALATION_MODEL", "claude-sonnet-5")

# Microarray probes/genes at or above this count are considered "full transcriptome"
# coverage rather than an older, lower-density platform.
COVERAGE_THRESHOLD = int(os.environ.get("GEOTOOL_COVERAGE_THRESHOLD", "12000"))

# Microarray platforms below this many probes/genes are rejected outright at download
# time (too old/low-density to be a usable expression platform, e.g. early spotted-cDNA
# or BAC arrays) -- a separate, lower bar than COVERAGE_THRESHOLD above, which is only an
# informational "full_transcriptome" vs "limited" label, not a hard gate.
MIN_ARRAY_PROBE_COUNT = int(os.environ.get("GEOTOOL_MIN_ARRAY_PROBE_COUNT", "8000"))

# A real full-transcriptome bulk RNA-seq gene-level table has tens of thousands of rows.
# Fewer than this strongly suggests the submitter only published a filtered subset (live
# example: GSE197728's supplementary table, which only reports genes with FPKM > 10 --
# 7833 rows, not a whole transcriptome) rather than every gene actually measured. RNA-seq
# only -- unlike COVERAGE_THRESHOLD above, this isn't a microarray platform-density check.
MIN_EXPECTED_RNASEQ_GENE_COUNT = int(os.environ.get("GEOTOOL_MIN_EXPECTED_RNASEQ_GENE_COUNT", "16000"))


def ensure_dirs() -> None:
    for d in (GEO_CACHE_DIR, REPORTS_DIR, SERIES_DIR, PLATFORMS_DIR):
        d.mkdir(parents=True, exist_ok=True)
