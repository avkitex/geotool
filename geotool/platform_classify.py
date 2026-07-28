"""Deterministic (no-LLM) classification of GEO platforms and assay details.

Species, assay "shape" (seq vs array), array vendor, and array coverage are
all already present in GEO's own metadata fields -- classifying them here in
plain code means zero LLM tokens are spent on questions structured metadata
already answers.
"""
from __future__ import annotations

import re

import pandas as pd

from geotool import config

_RNASEQ_TECH_RE = re.compile(r"high.?throughput sequencing", re.IGNORECASE)
_MICROARRAY_TECH_RE = re.compile(r"oligonucleotide|spotted|in situ", re.IGNORECASE)
_SCRNA_HINT_RE = re.compile(r"single.?cell|single.?nucle", re.IGNORECASE)

_VENDOR_PATTERNS = {
    "affymetrix": re.compile(r"affymetrix", re.IGNORECASE),
    "illumina": re.compile(r"illumina", re.IGNORECASE),
    "agilent": re.compile(r"agilent", re.IGNORECASE),
}

_LIBSEL_POLYA_RE = re.compile(r"poly.?a|oligo.?dt", re.IGNORECASE)
_LIBSEL_TOTAL_RE = re.compile(r"random|ribo.?zero|rrna depletion", re.IGNORECASE)
_LIBSEL_EXOME_RE = re.compile(r"hybrid selection|exome", re.IGNORECASE)

_PROTOCOL_POLYA_RE = re.compile(r"poly-?a", re.IGNORECASE)
_PROTOCOL_TOTAL_RE = re.compile(r"ribo-?zero|rrna depletion|total rna", re.IGNORECASE)
_PROTOCOL_EXOME_RE = re.compile(r"exome", re.IGNORECASE)

_TENX_RE = re.compile(r"10x|chromium|droplet", re.IGNORECASE)
_SMARTSEQ_RE = re.compile(r"smart-?seq|fluidigm|plate-?based", re.IGNORECASE)


def as_str(value) -> str:
    """GEOparse/esummary fields are sometimes a list-of-one, sometimes a bare string."""
    if isinstance(value, list):
        return value[0] if value else ""
    return value or ""


def classify_assay_type(technology: str, title: str = "") -> str:
    if _RNASEQ_TECH_RE.search(technology):
        return "scrnaseq" if _SCRNA_HINT_RE.search(title) else "bulk_rnaseq"
    if _MICROARRAY_TECH_RE.search(technology):
        return "microarray"
    return "other"


def classify_vendor(manufacturer: str, title: str = "") -> str:
    haystack = f"{manufacturer} {title}"
    for vendor, pattern in _VENDOR_PATTERNS.items():
        if pattern.search(haystack):
            return vendor
    return "other"


def classify_coverage(data_row_count) -> str:
    try:
        count = int(data_row_count)
    except (TypeError, ValueError):
        return "unknown"
    if count <= 0:
        return "unknown"
    return "full_transcriptome" if count >= config.COVERAGE_THRESHOLD else "limited"


def classify_platform(gpl_id: str, metadata: dict) -> dict:
    """Classify a platform from either a GEOparse gpl.metadata dict or a
    normalized `db=gpl` esummary docsum -- both carry title/technology under
    slightly different keys.
    """
    title = as_str(metadata.get("title", ""))
    technology = as_str(metadata.get("technology") or metadata.get("ptechtype") or "")
    manufacturer = as_str(metadata.get("manufacturer", ""))

    assay_type = classify_assay_type(technology, title)
    vendor = None
    coverage = None
    if assay_type == "microarray":
        vendor = classify_vendor(manufacturer, title)
        coverage = classify_coverage(as_str(metadata.get("data_row_count")))

    return {"gpl_id": gpl_id, "assay_type": assay_type, "vendor": vendor, "coverage": coverage}


def classify_rnaseq_library(sample_metadata: dict) -> str:
    """polyA / total_rna / exome_capture / other / unknown, from library_selection
    (SRA-standard field) with a protocol-text regex fallback.
    """
    library_selection = as_str(sample_metadata.get("library_selection")).strip()
    if library_selection and library_selection.lower() != "unspecified":
        if _LIBSEL_POLYA_RE.search(library_selection):
            return "polyA"
        if _LIBSEL_EXOME_RE.search(library_selection):
            return "exome_capture"
        if _LIBSEL_TOTAL_RE.search(library_selection):
            return "total_rna"
        return "other"

    protocol_text = " ".join(sample_metadata.get("extract_protocol_ch1", []))
    if _PROTOCOL_EXOME_RE.search(protocol_text):
        return "exome_capture"
    if _PROTOCOL_POLYA_RE.search(protocol_text):
        return "polyA"
    if _PROTOCOL_TOTAL_RE.search(protocol_text):
        return "total_rna"
    return "unknown"


def classify_scrna_platform(series_row: dict, samples: pd.DataFrame) -> str:
    """10x / smartseq / other / unknown, from platform title + free-text fields.

    Best-effort regex over series-level text; left "unknown" rather than
    spending an LLM call to force an answer when nothing matches.
    """
    text_parts = [
        series_row.get("title", ""),
        series_row.get("overall_design", ""),
        series_row.get("summary", ""),
    ]
    if "description" in samples.columns:
        text_parts.extend(samples["description"].astype(str).tolist())
    haystack = " ".join(str(p) for p in text_parts)

    if _TENX_RE.search(haystack):
        return "10x"
    if _SMARTSEQ_RE.search(haystack):
        return "smartseq"
    return "unknown"
