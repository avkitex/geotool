"""Orchestrates: Entrez title/description search -> optional sample-property filter."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from geotool import annotate, config, entrez, geo_fetch, llm_annotate, nl_query, platform_classify


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


def _persist_annotated(gse_id: str, series_row: dict, samples: pd.DataFrame) -> None:
    out_dir = config.SERIES_DIR / gse_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([series_row]).to_csv(out_dir / "series.tsv", sep="\t", index=False)
    samples.to_csv(out_dir / "samples.tsv", sep="\t", index=False)


def llm_annotate_candidates(candidates: list[dict], escalate_ambiguous: bool = False) -> list[dict]:
    """Fetch + LLM-annotate every candidate series, writing llm_* columns into
    series.tsv/samples.tsv and adding a diagnosis-breakdown receipt column to
    each candidate dict for the report.
    """
    for series in candidates:
        gse_id = series["gse_id"]
        try:
            gse = geo_fetch.fetch_series(gse_id)
            srow = annotate.series_row(gse)
            samples = annotate.samples_table(gse)
            merged, series_level = llm_annotate.annotate_and_cache(
                gse, srow, samples, escalate_ambiguous=escalate_ambiguous
            )
        except Exception as exc:  # network/parsing/LLM failures shouldn't kill the whole run
            series["llm_diagnosis_breakdown"] = f"ERROR: {exc}"
            continue

        _persist_annotated(gse_id, srow, merged)
        breakdown = merged["llm_diagnosis"].value_counts().to_dict() if "llm_diagnosis" in merged.columns else {}
        series["llm_diagnosis_breakdown"] = "; ".join(f"{k}={v}" for k, v in breakdown.items())
        series["llm_assay_type"] = series_level.assay_type
    return candidates


def search(
    title: str | None = None,
    description: str | None = None,
    organism: str | None = None,
    sample_properties: list[str] | None = None,
    max_results: int = 100,
    llm_annotate_flag: bool = False,
    llm_escalate: bool = False,
) -> list[dict]:
    candidates = entrez.search_series(
        title=title, description=description, organism=organism, max_results=max_results
    )
    if sample_properties:
        candidates = filter_by_sample_properties(candidates, sample_properties)
    else:
        for c in candidates:
            c["sample_property_matches"] = ""

    all_gpl_ids = sorted({gpl for c in candidates for gpl in c.get("platforms", [])})
    gpl_docsums = entrez.esummary_gpl(all_gpl_ids) if all_gpl_ids else {}
    for c in candidates:
        c["array_content"] = platform_classify.summarize_array_content(gpl_docsums, c.get("platforms", []))

    if llm_annotate_flag:
        candidates = llm_annotate_candidates(candidates, escalate_ambiguous=llm_escalate)

    return candidates


def _log_filters(filters: nl_query.QueryFilters) -> None:
    print("Parsed query filters (from Claude):")
    print(f"  diagnosis:         {filters.diagnosis}")
    print(f"  diagnosis synonyms: {', '.join(filters.diagnosis_synonyms) or '(none)'}")
    print(f"  species:           {filters.species}")
    print(f"  sample_type:       {filters.sample_type}")
    print(f"  tissue_class:      {filters.tissue_class}")
    print(f"  assay_type:        {filters.assay_type}")
    print(f"  selection_method:  {filters.selection_method}")
    print(f"  min_samples:       {filters.min_samples if filters.min_samples is not None else '(none)'}")


def run_nl_query(
    text: str, max_results: int = 100, escalate_ambiguous: bool = False, verbose: bool = True
) -> list[dict]:
    """geotool query pipeline:

    1. NL -> QueryFilters (diagnosis + synonyms + filter categories), one Claude call.
    2. Broad Entrez recall using the diagnosis OR each synonym -- no full-record
       fetch, just esummary docsums (title/summary/platforms/n_samples).
    3. One lightweight Claude call per candidate classifying its summary against
       the filters.
    4. Every candidate is returned with its classification as separate columns --
       nothing is dropped here; the report is the filter, the human decides.
    """
    filters = nl_query.parse_query_filters(text)
    if verbose:
        _log_filters(filters)

    terms = [filters.diagnosis] + filters.diagnosis_synonyms
    diagnosis_clause = " OR ".join(f'"{t}"[Title] OR "{t}"[Description]' for t in terms if t)
    clauses = [f"({diagnosis_clause})"]
    if filters.species in ("human", "mouse"):
        organism = "Homo sapiens" if filters.species == "human" else "Mus musculus"
        clauses.append(f'"{organism}"[Organism]')
    term = " AND ".join(clauses) + " AND GSE[ETYP]"

    uids, _total = entrez.esearch_gds(term, retmax=max_results)
    candidates = [entrez.normalize_docsum(d) for d in entrez.esummary_gds(uids)]
    if verbose:
        print(f"\nEntrez search term: {term}")
        print(f"Found {len(candidates)} candidate series.\n")

    all_gpl_ids = sorted({gpl for c in candidates for gpl in c.get("platforms", [])})
    gpl_docsums = entrez.esummary_gpl(all_gpl_ids) if all_gpl_ids else {}

    rows = []
    for i, candidate in enumerate(candidates, start=1):
        gse_id = candidate.get("gse_id", "")
        platform_titles = [
            platform_classify.as_str(gpl_docsums[gpl].get("title", ""))
            for gpl in candidate.get("platforms", [])
            if gpl in gpl_docsums
        ]
        row = dict(candidate)
        row["meets_min_samples"] = (
            "" if filters.min_samples is None else candidate.get("n_samples", 0) >= filters.min_samples
        )
        row["array_content"] = platform_classify.summarize_array_content(gpl_docsums, candidate.get("platforms", []))
        try:
            classification = nl_query.classify_series_with_escalation(
                candidate, filters, platform_titles, escalate=escalate_ambiguous
            )
        except Exception as exc:
            row["notes"] = f"classification failed: {exc}"
            rows.append(row)
            if verbose:
                print(f"  [{i}/{len(candidates)}] {gse_id}: classification FAILED ({exc})")
            continue

        if verbose:
            print(
                f"  [{i}/{len(candidates)}] {gse_id}: matches_diagnosis={classification.matches_diagnosis} "
                f"species={classification.species} sample_type={classification.sample_type} "
                f"tissue_class={classification.tissue_class} assay_type={classification.assay_type} "
                f"selection_method={classification.selection_method} n_samples={candidate.get('n_samples', '')}"
            )
            if classification.diagnosis_detail:
                print(f"      diagnosis_detail: {classification.diagnosis_detail}")
            if classification.notes:
                print(f"      notes: {classification.notes}")

        row.update({
            "matches_diagnosis": classification.matches_diagnosis,
            "diagnosis_detail": classification.diagnosis_detail,
            "species": classification.species,
            "sample_type": classification.sample_type,
            "tissue_class": classification.tissue_class,
            "assay_type": classification.assay_type,
            "selection_method": classification.selection_method,
            "notes": classification.notes,
        })
        rows.append(row)
    return rows
