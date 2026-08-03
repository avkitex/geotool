"""Wraps GEOparse.get_GEO with our on-disk cache directory.

GEOparse already skips re-downloading a SOFT family file that exists on disk,
so this module doesn't need its own bookkeeping beyond pointing it at
config.GEO_CACHE_DIR.
"""
from __future__ import annotations

import re

import GEOparse

from geotool import config

_SUPERSERIES_OF_RE = re.compile(r"^SuperSeries of:\s*(GSE\d+)", re.IGNORECASE)


def fetch_series(gse_id: str):
    """Download (or load from cache) the full SOFT record for a GEO Series.

    Returns a GEOparse.GSE object with .gsms (GSM objects) and .gpls (GPL objects).
    """
    config.ensure_dirs()
    return GEOparse.get_GEO(geo=gse_id, destdir=str(config.GEO_CACHE_DIR), silent=True)


def resolve_leaf_series_ids(gse_id: str, _seen: set[str] | None = None) -> list[str]:
    """Expand gse_id into the leaf (non-SuperSeries) series id(s) to actually
    download: itself, if it isn't a SuperSeries, or every one of its subseries
    -- recursively, in case a subseries is itself a SuperSeries -- if it is.

    A SuperSeries' own fetched record already contains every subseries' samples
    merged into one gse.gsms/gse.gpls (verified live against real GEO records),
    but that merges unrelated platforms/assay types together with no way to tell
    them apart after the fact. Its `relation` metadata ("SuperSeries of: GSEXXXX")
    is a clean list of its direct children, so processing each child as its own
    independent series -- the existing single-series pipeline, just called once
    per leaf -- keeps each assay/platform's own eligibility check meaningful.
    """
    seen = _seen if _seen is not None else set()
    if gse_id in seen:
        return []
    seen.add(gse_id)

    gse = fetch_series(gse_id)
    children = [
        match.group(1)
        for rel in gse.metadata.get("relation", [])
        for match in (_SUPERSERIES_OF_RE.match(rel),)
        if match
    ]
    if not children:
        return [gse_id]

    leaves: list[str] = []
    for child_id in children:
        leaves.extend(resolve_leaf_series_ids(child_id, seen))
    return leaves


def fetch_platform(gpl_id: str):
    """Download (or load from cache) a GEO Platform's own record.

    Returns a GEOparse.GPL object; `.table` is the platform's annotation
    table (probe ID -> gene symbol/Entrez ID/etc, layout varies per platform).
    """
    config.ensure_dirs()
    return GEOparse.get_GEO(geo=gpl_id, destdir=str(config.GEO_CACHE_DIR), silent=True)
