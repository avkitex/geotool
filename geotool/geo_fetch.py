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


def all_supplementary_file_urls(gse) -> set[str]:
    """Every supplementary-file URL a fetched series record makes reachable --
    its own series-level `supplementary_file` metadata, plus every sample's
    own per-GSM `supplementary_file*` keys. This is the exact same "what
    counts as covered" definition download.download_rnaseq_files uses to
    decide what to actually fetch, factored out here so
    find_superseries_orphans checks a SuperSeries parent's files against the
    same definition of "already covered by a subseries" that downloading the
    subseries itself uses -- not a narrower, series-level-only comparison
    that would flag a file as "orphaned" when it's really just published at
    the sample level on a leaf series instead of the series level.
    """
    urls = {f for f in gse.metadata.get("supplementary_file", []) if f and f.strip().upper() != "NONE"}
    for gsm in gse.gsms.values():
        for key, values in gsm.metadata.items():
            if key.startswith("supplementary_file"):
                urls.update(v for v in values if v and v.strip().upper() != "NONE")
    return urls


def find_superseries_orphans(gse_id: str, leaf_ids: list[str]) -> dict:
    """Check whether a SuperSeries' own GEO record carries anything not
    accounted for by its subseries -- resolve_leaf_series_ids' docstring
    notes this is expected *not* to happen for GSMs (verified live against
    real GEO records: a SuperSeries' own gse.gsms is just the union of its
    children's), but that was only checked for samples, not supplementary
    files, and "expected" isn't "guaranteed" for every SuperSeries GEO will
    ever host -- so check both, defensively, rather than assume it can never
    happen:

    1. orphaned_gsm_ids -- GSM ids present in the parent's own record but
       in none of its (recursively resolved) leaf series. If GEO ever
       attaches a sample directly to a SuperSeries without also listing it
       under a subseries, download_cohort would never see it (it's only
       ever called per leaf id).
    2. orphaned_supplementary_files -- supplementary-file URLs on the
       parent's own record (series- or, in principle, sample-level, via
       all_supplementary_file_urls) that don't appear anywhere among the
       leaves' own covered URLs (same function, same definition of
       "covered" download_rnaseq_files itself uses) -- so a file published
       on *both* the parent and a subseries is never double-counted as
       "extra" data; only a URL genuinely absent from every subseries is.

    fetch_series's on-disk cache (see fetch_series) makes re-fetching the
    parent and each already-fetched leaf here a cache hit, not a new
    network call.
    """
    parent = fetch_series(gse_id)
    parent_gsm_ids = set(parent.gsms.keys())
    parent_urls = all_supplementary_file_urls(parent)

    leaf_gsm_ids: set[str] = set()
    leaf_urls: set[str] = set()
    for leaf_id in leaf_ids:
        leaf = fetch_series(leaf_id)
        leaf_gsm_ids.update(leaf.gsms.keys())
        leaf_urls.update(all_supplementary_file_urls(leaf))

    return {
        "orphaned_gsm_ids": sorted(parent_gsm_ids - leaf_gsm_ids),
        "orphaned_supplementary_files": sorted(parent_urls - leaf_urls),
    }


def fetch_platform(gpl_id: str):
    """Download (or load from cache) a GEO Platform's own record.

    Returns a GEOparse.GPL object; `.table` is the platform's annotation
    table (probe ID -> gene symbol/Entrez ID/etc, layout varies per platform).
    """
    config.ensure_dirs()
    return GEOparse.get_GEO(geo=gpl_id, destdir=str(config.GEO_CACHE_DIR), silent=True)
