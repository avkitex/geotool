"""Turn a fetched GEOparse GSE object into series.tsv + samples.tsv.

These two flat files are the shared artifact between Phase 1 (search/report),
Phase 2 (download will join against gsm_id/platform), and Phase 3 (harmonize
will read characteristics columns to build a cross-cohort schema).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from geotool import config, platform_classify


def _first(metadata: dict, key: str, default: str = "") -> str:
    values = metadata.get(key, [])
    return values[0] if values else default


def parse_characteristics(raw: list[str]) -> dict[str, str]:
    """Parse GEO's 'key: value' characteristics strings into a dict.

    Duplicate keys (rare, e.g. two "treatment:" lines) are suffixed _2, _3, ...
    so no data is silently dropped.
    """
    parsed: dict[str, str] = {}
    for entry in raw:
        if ":" in entry:
            key, value = entry.split(":", 1)
            key, value = key.strip().lower(), value.strip()
        else:
            key, value = entry.strip().lower(), ""
        if key in parsed:
            n = 2
            while f"{key}_{n}" in parsed:
                n += 1
            key = f"{key}_{n}"
        parsed[key] = value
    return parsed


def _series_organism(gse) -> str:
    # Series-level metadata has no 'organism' field; GEO records it per-sample
    # (organism_ch1). Any sample is representative since GEO series are
    # single-organism in practice.
    for gsm in gse.gsms.values():
        organism = _first(gsm.metadata, "organism_ch1")
        if organism:
            return organism
    return ""


def platform_details(gse) -> list[dict]:
    """classify_platform() result for every GPL in the series (Tier 0, no LLM)."""
    gpls = getattr(gse, "gpls", None) or {}
    return [platform_classify.classify_platform(gpl_id, gpl.metadata) for gpl_id, gpl in gpls.items()]


def series_row(gse) -> dict:
    md = gse.metadata
    return {
        "gse_id": _first(md, "geo_accession", gse.get_accession() if hasattr(gse, "get_accession") else ""),
        "title": _first(md, "title"),
        "summary": " ".join(md.get("summary", [])),
        "overall_design": _first(md, "overall_design"),
        "organism": _series_organism(gse),
        "platforms": ";".join(sorted(gse.gpls.keys())) if getattr(gse, "gpls", None) else "",
        "platform_details": json.dumps(platform_details(gse)),
        "n_samples": len(gse.gsms) if getattr(gse, "gsms", None) else 0,
        "submission_date": _first(md, "submission_date"),
        "pubmed_ids": ";".join(md.get("pubmed_id", [])),
    }


def samples_table(gse) -> pd.DataFrame:
    gse_id = _first(gse.metadata, "geo_accession")
    rows = []
    for gsm_id, gsm in gse.gsms.items():
        md = gsm.metadata
        row = {
            "gsm_id": gsm_id,
            "gse_id": gse_id,
            "title": _first(md, "title"),
            "source_name_ch1": _first(md, "source_name_ch1"),
            "organism_ch1": _first(md, "organism_ch1"),
            "molecule_ch1": _first(md, "molecule_ch1"),
            "platform_id": _first(md, "platform_id"),
            "description": " ".join(md.get("description", [])),
            "library_selection": _first(md, "library_selection"),
            "library_strategy": _first(md, "library_strategy"),
            "data_row_count": _first(md, "data_row_count"),
            "rnaseq_library_type": platform_classify.classify_rnaseq_library(md),
        }
        row.update(parse_characteristics(md.get("characteristics_ch1", [])))
        rows.append(row)
    return pd.DataFrame(rows)


def write_series_files(gse, series_dir: Path | None = None) -> tuple[Path, Path]:
    """Write series.tsv and samples.tsv for a fetched GSE, return their paths."""
    srow = series_row(gse)
    gse_id = srow["gse_id"]
    out_dir = (series_dir or config.SERIES_DIR) / gse_id
    out_dir.mkdir(parents=True, exist_ok=True)

    series_path = out_dir / "series.tsv"
    pd.DataFrame([srow]).to_csv(series_path, sep="\t", index=False)

    samples_path = out_dir / "samples.tsv"
    samples_table(gse).to_csv(samples_path, sep="\t", index=False)

    return series_path, samples_path
