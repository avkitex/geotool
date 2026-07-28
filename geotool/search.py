"""Orchestrates: Entrez title/description search -> optional sample-property filter."""
from __future__ import annotations

from geotool import annotate, entrez, geo_fetch


def parse_property_filter(spec: str) -> tuple[str, str]:
    """'tissue:liver' -> ('tissue', 'liver'). Both sides matched case-insensitively."""
    key, sep, value = spec.partition(":")
    if not sep:
        raise ValueError(f"--sample-property must be 'key:value', got {spec!r}")
    return key.strip().lower(), value.strip().lower()


def _matches(samples, key: str, value: str):
    """Boolean Series: which sample rows match value (substring) for the given key.

    Falls back to searching every column if `key` isn't a column on this
    platform (characteristic keys vary a lot between series).
    """
    if key in samples.columns:
        haystack = samples[key].astype(str).str.lower()
    else:
        haystack = samples.astype(str).apply(lambda r: " | ".join(r.values).lower(), axis=1)
    return haystack.str.contains(value, na=False, regex=False)


def filter_by_sample_properties(
    candidates: list[dict],
    property_filters: list[str],
    write_annotations: bool = True,
) -> list[dict]:
    """Keep only series where every property filter matches at least one sample.

    Fetches each candidate's full SOFT record (slow path, only run when the
    user actually asked for sample-property filtering) and, as a side effect,
    writes series.tsv/samples.tsv for every series it had to fetch.
    """
    parsed_filters = [parse_property_filter(spec) for spec in property_filters]
    kept = []
    for series in candidates:
        gse_id = series["gse_id"]
        try:
            gse = geo_fetch.fetch_series(gse_id)
        except Exception as exc:  # network/parsing failures shouldn't kill the whole search
            series["sample_property_matches"] = f"ERROR: {exc}"
            kept.append(series)
            continue

        samples = annotate.samples_table(gse)
        n_total = len(samples)
        summaries = []
        keep = True
        for key, value in parsed_filters:
            n_matched = int(_matches(samples, key, value).sum()) if n_total else 0
            summaries.append(f"{key}:{value}={n_matched}/{n_total}")
            if n_matched == 0:
                keep = False

        if keep:
            series["sample_property_matches"] = "; ".join(summaries)
            if write_annotations:
                annotate.write_series_files(gse)
            kept.append(series)
    return kept


def search(
    title: str | None = None,
    description: str | None = None,
    organism: str | None = None,
    sample_properties: list[str] | None = None,
    max_results: int = 100,
) -> list[dict]:
    candidates = entrez.search_series(
        title=title, description=description, organism=organism, max_results=max_results
    )
    if sample_properties:
        return filter_by_sample_properties(candidates, sample_properties)
    for c in candidates:
        c["sample_property_matches"] = ""
    return candidates
