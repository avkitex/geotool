"""Wraps GEOparse.get_GEO with our on-disk cache directory.

GEOparse already skips re-downloading a SOFT family file that exists on disk,
so this module doesn't need its own bookkeeping beyond pointing it at
config.GEO_CACHE_DIR.
"""
from __future__ import annotations

import GEOparse

from geotool import config


def fetch_series(gse_id: str):
    """Download (or load from cache) the full SOFT record for a GEO Series.

    Returns a GEOparse.GSE object with .gsms (GSM objects) and .gpls (GPL objects).
    """
    config.ensure_dirs()
    return GEOparse.get_GEO(geo=gse_id, destdir=str(config.GEO_CACHE_DIR), silent=True)


def fetch_platform(gpl_id: str):
    """Download (or load from cache) a GEO Platform's own record.

    Returns a GEOparse.GPL object; `.table` is the platform's annotation
    table (probe ID -> gene symbol/Entrez ID/etc, layout varies per platform).
    """
    config.ensure_dirs()
    return GEOparse.get_GEO(geo=gpl_id, destdir=str(config.GEO_CACHE_DIR), silent=True)
