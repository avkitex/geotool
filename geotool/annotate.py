"""Turn a fetched GEOparse GSE object into series.tsv + samples.tsv.

These two flat files are the shared artifact between Phase 1 (search/report),
Phase 2 (download will join against gsm_id/platform), and Phase 3 (harmonize
will read characteristics columns to build a cross-cohort schema).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from geotool import config, platform_classify, probe_mapping


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


def is_human_organism(organism: str) -> bool:
    return organism.strip().lower() == "homo sapiens"


def _series_reference_channel(gse) -> int | None:
    """Series-level reference-channel call (1 or 2) from per-sample ch1/ch2
    metadata text alone (probe_mapping._channel_metadata_hint,
    probe_mapping._MIN_METADATA_AGREEMENT) -- None if the series isn't
    two-channel at all, or no clear majority.

    Reuses probe_mapping's existing metadata-only signal rather than its
    full detect_reference_channel (which also needs per-channel expression
    VALUE columns, via probe_mapping.detect_channel_columns, to compute a
    cross-sample variance signal) -- most two-channel Agilent series never
    publish those, only the precomputed ratio (see download.py's docstring),
    so detect_reference_channel can't run for them at all. Live example this
    was built for: GSE71729 (the Moffitt "Virtual Microdissection of PDAC"
    cohort) and GSE77858, both ratio-only two-channel Agilent series where
    channel 1 is always "Human Reference"/a pooled reference mix and channel
    2 carries the real per-sample tumor/normal/met identity -- entirely
    invisible to samples_table before this, which only ever read
    characteristics_ch1/source_name_ch1/etc, so these series' real sample
    identity was silently dropped (annotate.py never even looked at ch2)
    and clinical_annotate's LLM step, seeing only the constant reference
    channel's metadata, concluded "cell line reference dataset, no clinical
    samples" -- wrong, but consistent with what it was actually shown.
    """
    hints = [
        probe_mapping._channel_metadata_hint(gsm)
        for gsm in gse.gsms.values()
        if platform_classify.as_str(gsm.metadata.get("channel_count")).strip() == "2"
    ]
    clear_hints = [h for h in hints if h is not None]
    if not clear_hints:
        return None
    if clear_hints.count(1) / len(clear_hints) >= probe_mapping._MIN_METADATA_AGREEMENT:
        return 1
    if clear_hints.count(2) / len(clear_hints) >= probe_mapping._MIN_METADATA_AGREEMENT:
        return 2
    return None


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
    reference_channel = _series_reference_channel(gse)
    rows = []
    for gsm_id, gsm in gse.gsms.items():
        md = gsm.metadata
        channel_count = platform_classify.as_str(md.get("channel_count")).strip()
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

        # Two-channel (e.g. Agilent reference-design) samples carry a SECOND,
        # entirely separate set of characteristics/source_name/organism under
        # "_ch2" -- never read above, so a series whose channel 1 is a fixed
        # reference (e.g. "Human Reference" on every single array) silently
        # lost its real per-sample identity, which lives in ch2. Always
        # captured here when present, regardless of whether a confident
        # reference-channel call can be made below, since it's real data
        # either way. Characteristic keys are suffixed "_ch2" (distinct from
        # ch1's un-suffixed keys) rather than merged, since the same key
        # (e.g. "tissue") can carry a different value on each channel.
        if channel_count == "2":
            row["channel_count"] = channel_count
            row["source_name_ch2"] = _first(md, "source_name_ch2")
            row["organism_ch2"] = _first(md, "organism_ch2")
            row["molecule_ch2"] = _first(md, "molecule_ch2")
            for key, value in parse_characteristics(md.get("characteristics_ch2", [])).items():
                row[f"{key}_ch2"] = value
            if reference_channel is not None:
                # Which channel to actually read for "the real sample" is
                # exactly the ambiguity a human (or an LLM given only ch1)
                # can't resolve from column names alone -- this makes the
                # already-computed series-level call explicit and visible
                # on every row, rather than requiring every downstream
                # reader (clinical_annotate's LLM prompt included) to
                # re-derive it.
                row["reference_channel"] = reference_channel

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
