"""One-row-per-*cohort* summary across already-downloaded cohorts --
companion to harmonize.py's one-row-per-*sample* master table. Together the
two are "the harmonization process": `geotool harmonize` writes both
data/harmonized/<name>/annotation.tsv (sample-level, from harmonize.py) and
data/harmonized/<name>/cohort_annotations.tsv (this module).

Combines each cohort's own series.tsv/annotation.tsv (download outcome,
sample count, expression QC status -- both written by `geotool download`)
with SuperSeries structure (data/series/<gse_id>/superseries.json, written by
geotool.download.resolve_download_targets whenever a requested id turns out
to be a SuperSeries) -- every subseries a requested SuperSeries expanded to
gets its own row here too, including ones never explicitly requested to
begin with (flagged via in_requested_list), with parent_series pointing back
to it, and any data GEO's own SuperSeries record carries that isn't in any
subseries (an "orphan") surfaced on the SuperSeries' own row's
reason_if_not_ready rather than silently dropped.

readiness/expression_file default to expression_status alone (from each
cohort's own annotation.tsv, backfilled by `geotool download`): "ok" means
ready. Pass collection_root to instead check that project's own processed
matrix collection (e.g. data/mtap_prmt5_cohorts, built by a project-specific
pipeline on top of geotool.gene_symbol_mapping) for a
<collection_root>/<gse_id>/<final_matrix_filename> file -- readiness then
means "this specific analysis-ready file exists on disk", a stronger claim
than "geotool resolved *some* raw expression file for this cohort".

sample_id_match_status/sample_id_match_detail likewise only mean something
when collection_root is given: whether geotool.rnaseq_finalize's own
<collection_root>/<gse_id>/sample_id_map.tsv (geotool.sample_id_matching --
expression matrix column -> gsm_id) resolved every sample by name/order
("matched") or left at least one unresolved ("needs_review" -- a human
needs to look at that cohort's sample_id_map.tsv and either supply a
manual mapping or accept the gap).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from geotool import config
from geotool import download as download_mod

DEFAULT_FINAL_MATRIX_FILENAME = "expression_final.tsv.gz"
DEFAULT_SAMPLE_ID_MAP_FILENAME = "sample_id_map.tsv"


def _load_superseries_marker(gse_id: str, series_dir: Path) -> dict | None:
    return download_mod._load_superseries_marker(series_dir / gse_id)


def resolve_effective_cohort_ids(
    gse_ids: list[str], series_dir: Path | None = None
) -> tuple[list[str], dict[str, str]]:
    """Expand gse_ids to include every subseries a requested SuperSeries id
    resolved to. Without this, an accession pulled to disk only as a side
    effect of downloading its SuperSeries parent would never show up in any
    report, even though it's real data sitting in data/series.

    Returns (all_ids, parent_of): all_ids is gse_ids plus every
    auto-discovered subseries, stable order, no duplicates; parent_of maps
    {gse_id: parent_gse_id} for every id that's a subseries of some other id
    in all_ids (whether or not that parent was itself in gse_ids).
    """
    series_dir = series_dir or config.SERIES_DIR
    parent_of: dict[str, str] = {}
    extra_ids: list[str] = []

    # Walk to a fixed point: a newly-discovered subseries could itself be a
    # SuperSeries with its own marker (geo_fetch.resolve_leaf_series_ids
    # already recurses when *resolving*, but marker files are only written
    # at each level it actually visited, so re-check newly added ids too).
    frontier = list(gse_ids)
    seen = set(gse_ids)
    while frontier:
        gse_id = frontier.pop()
        marker = _load_superseries_marker(gse_id, series_dir)
        if not marker:
            continue
        for sub_id in marker.get("subseries", []):
            parent_of.setdefault(sub_id, gse_id)
            if sub_id not in seen:
                seen.add(sub_id)
                extra_ids.append(sub_id)
                frontier.append(sub_id)

    return list(gse_ids) + extra_ids, parent_of


def _orphan_note(marker: dict | None) -> str:
    if not marker:
        return ""
    orphaned_gsms = marker.get("orphaned_gsm_ids") or []
    orphaned_files = marker.get("orphaned_supplementary_files") or []
    if not orphaned_gsms and not orphaned_files:
        return ""
    parts = []
    if orphaned_gsms:
        parts.append(
            f"{len(orphaned_gsms)} sample(s) attached to the SuperSeries record itself, "
            f"in no subseries: {', '.join(orphaned_gsms)}"
        )
    if orphaned_files:
        parts.append(
            f"{len(orphaned_files)} supplementary file(s) on the SuperSeries record itself, "
            f"in no subseries: {', '.join(orphaned_files)}"
        )
    return "SuperSeries carries data not covered by any subseries -- " + "; ".join(parts)


def _sample_id_match_status(collection_root: Path | str | None, gse_id: str, sample_id_map_filename: str) -> tuple[str, str]:
    """(status, detail) for <collection_root>/<gse_id>/<sample_id_map_filename>:
    status is "" if collection_root wasn't given (not checked), "not_available"
    if that cohort has no id map file at all, "matched" if every expression
    column resolved to a gsm_id, or "needs_review" if at least one didn't --
    meaning a human should look at that cohort's own sample_id_map.tsv rather
    than assume expression columns and gsm_ids line up.
    """
    if collection_root is None:
        return "", ""
    map_path = Path(collection_root) / gse_id / sample_id_map_filename
    if not map_path.exists():
        return "not_available", f"no {sample_id_map_filename} in {collection_root}/{gse_id}/"

    id_map = pd.read_csv(map_path, sep="\t")
    if "gsm_id" not in id_map.columns or id_map.empty:
        return "not_available", f"{sample_id_map_filename} has no gsm_id column or is empty"

    n_total = len(id_map)
    n_matched = int(id_map["gsm_id"].notna().sum())
    detail = f"{n_matched}/{n_total} matched"
    if n_matched == n_total:
        return "matched", detail
    return "needs_review", f"{detail} -- see {sample_id_map_filename} for unresolved samples"


def cohort_report_row(
    gse_id: str,
    series_dir: Path,
    parent_of: dict[str, str],
    requested_ids: set[str],
    collection_root: Path | str | None = None,
    final_matrix_filename: str = DEFAULT_FINAL_MATRIX_FILENAME,
    sample_id_map_filename: str = DEFAULT_SAMPLE_ID_MAP_FILENAME,
) -> dict:
    series_row_dir = series_dir / gse_id
    series_path = series_row_dir / "series.tsv"
    annotation_path = series_row_dir / "annotation.tsv"

    row = {
        "gse_id": gse_id,
        "parent_series": parent_of.get(gse_id, ""),
        "in_requested_list": gse_id in requested_ids,
        "downloaded": series_row_dir.exists(),
        "title": "",
        "organism": "",
        "assay_type": "",
        "platforms": "",
        "n_samples": None,
        "expression_status": "",
        "readiness": "not_ready",
        "expression_file": "",
        "reason_if_not_ready": "",
        "sample_id_match_status": "",
        "sample_id_match_detail": "",
    }

    if not series_row_dir.exists():
        row["reason_if_not_ready"] = "not downloaded yet (no data/series/<GSE> dir) -- run `geotool download` first"
        return row

    marker = _load_superseries_marker(gse_id, series_dir)
    orphan_note = _orphan_note(marker)

    if series_path.exists():
        srow = pd.read_csv(series_path, sep="\t").iloc[0]
        row["title"] = srow.get("title", "")
        row["organism"] = srow.get("organism", "")
        try:
            platform_details = json.loads(srow.get("platform_details", "[]"))
        except (json.JSONDecodeError, TypeError):
            platform_details = []
        row["assay_type"] = ";".join(sorted({p.get("assay_type") for p in platform_details if p.get("assay_type")}))
        row["platforms"] = ";".join(sorted({p.get("gpl_id") for p in platform_details if p.get("gpl_id")}))

    if annotation_path.exists():
        adf = pd.read_csv(annotation_path, sep="\t", low_memory=False)
        row["n_samples"] = len(adf)
        if "expression_status" in adf.columns and len(adf):
            row["expression_status"] = adf["expression_status"].iloc[0]

    if marker and not annotation_path.exists():
        # A pure SuperSeries record has no annotation.tsv of its own --
        # download_cohort is never called on it (see
        # download.resolve_download_targets); its real data lives in the
        # subseries rows this function also produces.
        row["reason_if_not_ready"] = (
            "this GSE is a SuperSeries record with no expression data of its own -- its actual "
            "data lives in its SubSeries, which (if in scope) are separate rows in this report"
        )
    elif collection_root is not None:
        candidate = Path(collection_root) / gse_id / final_matrix_filename
        if candidate.exists():
            row["readiness"] = "ready"
            row["expression_file"] = str(candidate.relative_to(collection_root)).replace("\\", "/")
        else:
            row["reason_if_not_ready"] = f"no {final_matrix_filename} in {collection_root}/{gse_id}/"
        row["sample_id_match_status"], row["sample_id_match_detail"] = _sample_id_match_status(
            collection_root, gse_id, sample_id_map_filename,
        )
    elif row["expression_status"] == "ok":
        row["readiness"] = "ready"
    else:
        row["reason_if_not_ready"] = row["expression_status"] or "no expression_status recorded"

    if orphan_note:
        row["reason_if_not_ready"] = (
            f"{row['reason_if_not_ready']}; {orphan_note}" if row["reason_if_not_ready"] else orphan_note
        )

    return row


def build_cohort_report(
    gse_ids: list[str],
    series_dir: Path | None = None,
    collection_root: Path | str | None = None,
    final_matrix_filename: str = DEFAULT_FINAL_MATRIX_FILENAME,
    sample_id_map_filename: str = DEFAULT_SAMPLE_ID_MAP_FILENAME,
) -> pd.DataFrame:
    """One row per cohort across gse_ids plus every SuperSeries-subseries it
    expands to (see resolve_effective_cohort_ids) -- title/organism/assay/
    platforms/n_samples/expression_status from each cohort's own series.tsv/
    annotation.tsv, readiness/expression_file from collection_root's final
    matrix file if given (else from expression_status alone), and
    sample_id_match_status/sample_id_match_detail from collection_root's
    sample_id_map.tsv if given (else left blank -- "" is not the same as
    "needs_review": it means this wasn't checked at all).
    """
    series_dir = series_dir or config.SERIES_DIR
    all_ids, parent_of = resolve_effective_cohort_ids(gse_ids, series_dir=series_dir)
    requested_ids = set(gse_ids)
    rows = [
        cohort_report_row(
            gid, series_dir, parent_of, requested_ids,
            collection_root=collection_root, final_matrix_filename=final_matrix_filename,
            sample_id_map_filename=sample_id_map_filename,
        )
        for gid in all_ids
    ]
    return pd.DataFrame(rows)
