"""NCBI Entrez E-utilities wrappers for searching the GEO DataSets (gds) database.

GEOparse can only fetch a *known* accession's SOFT record; it has no keyword
search. Title/description/organism search goes through esearch+esummary
against db=gds, restricted to GEO Series (entry type GSE) records.
"""
from __future__ import annotations

import time
from typing import Any

import requests

from geotool import config

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

MAX_RETRIES = 5
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_last_request_time = 0.0


def _throttle() -> None:
    """Keep requests under config.NCBI_REQUESTS_PER_SECOND (NCBI usage policy)."""
    global _last_request_time
    min_interval = 1.0 / config.NCBI_REQUESTS_PER_SECOND
    elapsed = time.monotonic() - _last_request_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_request_time = time.monotonic()


def _params(extra: dict[str, Any]) -> dict[str, Any]:
    params = {"db": "gds", "email": config.NCBI_EMAIL, "tool": "geotool"}
    if config.NCBI_API_KEY:
        params["api_key"] = config.NCBI_API_KEY
    params.update(extra)
    return params


def _get(url: str, params: dict[str, Any]) -> requests.Response:
    """GET with throttling plus retry-with-backoff on 429/5xx.

    NCBI rate limits are a normal occurrence under real usage (shared IP,
    transient load), not just test bursts -- a bare raise_for_status() would
    kill an entire search over one transient block.
    """
    delay = 2.0
    resp = None
    for attempt in range(MAX_RETRIES):
        _throttle()
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code in _RETRYABLE_STATUS and attempt < MAX_RETRIES - 1:
            retry_after = resp.headers.get("retry-after")
            time.sleep(float(retry_after) if retry_after else delay)
            delay *= 2
            continue
        break
    resp.raise_for_status()
    return resp


def build_query(
    title: str | None = None,
    description: str | None = None,
    organism: str | None = None,
    entry_type: str = "GSE",
) -> str:
    """Build an Entrez boolean query. Raises ValueError if no terms given."""
    clauses = []
    if title:
        clauses.append(f'"{title}"[Title]')
    if description:
        clauses.append(f'"{description}"[Description]')
    if organism:
        clauses.append(f'"{organism}"[Organism]')
    if not clauses:
        raise ValueError("At least one of title/description/organism is required")
    query = " AND ".join(clauses)
    if entry_type:
        query = f"({query}) AND {entry_type}[ETYP]"
    return query


def esearch_gds(term: str, retmax: int = 100, retstart: int = 0) -> tuple[list[str], int]:
    """Return (uid_list, total_count) for a gds search term."""
    resp = _get(ESEARCH_URL, _params({"term": term, "retmax": retmax, "retstart": retstart, "retmode": "json"}))
    result = resp.json()["esearchresult"]
    if "ERROR" in result:
        raise RuntimeError(f"Entrez esearch error: {result['ERROR']}")
    return result.get("idlist", []), int(result.get("count", 0))


def esummary_gds(uids: list[str]) -> list[dict[str, Any]]:
    """Return raw docsum dicts for a list of gds UIDs."""
    if not uids:
        return []
    resp = _get(ESUMMARY_URL, _params({"id": ",".join(uids), "retmode": "json"}))
    result = resp.json()["result"]
    return [result[uid] for uid in result.get("uids", [])]


def normalize_docsum(docsum: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw gds docsum into the fields the report needs."""
    gpl_ids = [f"GPL{gpl.strip()}" for gpl in str(docsum.get("gpl", "")).split(";") if gpl.strip()]
    return {
        "gse_id": docsum.get("accession", ""),
        "title": docsum.get("title", ""),
        "summary": docsum.get("summary", ""),
        "organism": docsum.get("taxon", ""),
        "platforms": gpl_ids,
        "n_samples": int(docsum.get("n_samples", 0) or 0),
        "submission_date": docsum.get("pdat", ""),
        "pubmed_ids": docsum.get("pubmedids", []),
    }


def esummary_gpl(gpl_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Batched platform lookup: one esearch + one esummary for every GPL accession.

    NCBI has no separate 'gpl' database -- platform records live in db=gds
    like everything else, disambiguated by entry type. Returns
    {gpl_id: docsum} for whichever accessions were found; docsum has
    title/ptechtype fields compatible with platform_classify.classify_platform.
    """
    unique_ids = sorted(set(gpl_ids))
    if not unique_ids:
        return {}
    term = "(" + " OR ".join(f"{gid}[Accession]" for gid in unique_ids) + ") AND GPL[ETYP]"
    uids, _total = esearch_gds(term, retmax=len(unique_ids) * 2)
    docsums = esummary_gds(uids)
    return {ds["accession"]: ds for ds in docsums if ds.get("accession") in unique_ids}


def search_series(
    title: str | None = None,
    description: str | None = None,
    organism: str | None = None,
    max_results: int = 100,
) -> list[dict[str, Any]]:
    """High-level title/description/organism search over GEO Series. No sample-level filtering."""
    term = build_query(title=title, description=description, organism=organism)
    uids, _total = esearch_gds(term, retmax=max_results)
    docsums = esummary_gds(uids)
    return [normalize_docsum(d) for d in docsums]
