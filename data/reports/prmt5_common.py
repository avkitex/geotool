"""Shared helpers for the PRMT5/MTAP cohort collection scripts.

Resolves the *effective* cohort id list: every id in
mtap_prmt5_selected.tsv, plus any subseries a selected SuperSeries id
expanded to (geotool.download.resolve_download_targets writes a
data/series/<gse_id>/superseries.json marker for this -- see
_write_superseries_marker) that isn't already in the selected list itself.
Without this, an accession like GSE240725 -- pulled to disk as a side effect
of downloading GSE240726, a SuperSeries -- would never show up in any
report, even though it's real data sitting in data/series.
"""
import json
from pathlib import Path

import pandas as pd

SERIES_ROOT = Path("data/series")
SELECTED_LIST = Path("data/reports/mtap_prmt5_selected.tsv")


def load_superseries_marker(gse_id):
    path = SERIES_ROOT / gse_id / "superseries.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def effective_cohort_ids():
    """Returns (all_ids, selected_ids, parent_of) --
    all_ids: selected_ids plus any auto-discovered subseries, in a stable order.
    selected_ids: the original mtap_prmt5_selected.tsv list, unchanged.
    parent_of: {gse_id: parent_gse_id} for every id that's a subseries of some
    other id in all_ids (whether or not that id was itself in the original
    selected list).
    """
    selected_ids = pd.read_csv(SELECTED_LIST, sep="\t")["gse_id"].tolist()
    parent_of: dict[str, str] = {}
    extra_ids: list[str] = []

    # Walk to a fixed point: a newly-discovered subseries could itself be a
    # SuperSeries with its own marker (geo_fetch.resolve_leaf_series_ids
    # already recurses when *resolving*, but marker files are only written
    # at each level it actually visited, so re-check newly added ids too).
    frontier = list(selected_ids)
    seen = set(selected_ids)
    while frontier:
        gse_id = frontier.pop()
        marker = load_superseries_marker(gse_id)
        if not marker:
            continue
        for sub_id in marker.get("subseries", []):
            parent_of.setdefault(sub_id, gse_id)
            if sub_id not in seen:
                seen.add(sub_id)
                extra_ids.append(sub_id)
                frontier.append(sub_id)

    return selected_ids + extra_ids, selected_ids, parent_of
