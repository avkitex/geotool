"""Detect and merge GEO's split-array "companion chip" platform pairs.

Some early Affymetrix expression arrays split one array's worth of content
across two physically separate chips because a single chip couldn't hold
the whole probe set -- most notably the Human Genome U133 Set, where GPL96
(HG-U133A) and GPL97 (HG-U133B) together cover one biological sample's
transcriptome. GEO records this as two separate GSM records per biological
sample (one per chip) rather than one combined record. Left alone, every
downstream expression matrix double-counts these samples as two, each with
roughly half the gene set the submitter actually measured.

COMPANION_PLATFORMS is intentionally small and evidence-based, not a guess
from adjacent-looking GPL numbers. GPL570 (HG-U133 Plus 2.0, a standalone
single chip that already contains the full A+B content) and GPL571
(HT_HG-U133A, an independent high-throughput variant of just the A array)
might look like a similar pair -- but checked against real GEO series
(SuperSeries bundling unrelated sub-experiments, different
volunteers/timepoints assigned to each platform, disjoint replicate
numbers split across the two platforms), none show genuine per-sample
splitting the way GPL96/GPL97 do. Add a pair here only once a real one is
confirmed the same way.
"""
from __future__ import annotations

import re

import pandas as pd

COMPANION_PLATFORMS: dict[str, str] = {
    "GPL96": "GPL97",
    "GPL97": "GPL96",
}

# Per-platform substrings that appear in a sample's title solely to identify
# which chip of the pair it was run on -- these are specific/long enough
# (e.g. "u133a") to strip anywhere they occur without risking a false match
# inside an unrelated word.
_CHIP_SUBSTRING_TOKENS: dict[str, list[str]] = {
    "GPL96": ["hg-u133a", "hgu133a", "u133a"],
    "GPL97": ["hg-u133b", "hgu133b", "u133b"],
}

# A bare trailing "_A"/"-A"/" A" designator (e.g. "PaCa1_A") is only safe to
# strip when it's the very last token of the title -- a generic single
# letter would otherwise false-match inside ordinary words (" a" inside
# "array", "_a" inside some accession-like token), so this is applied with
# a $-anchored regex rather than a plain substring replace.
_CHIP_SUFFIX_TOKENS: dict[str, str] = {
    "GPL96": r"[\s_-]a$",
    "GPL97": r"[\s_-]b$",
}


def companion_platforms_present(gpl_ids) -> list[tuple[str, str]]:
    """Every (gpl_a, gpl_b) companion pair -- alphabetically ordered,
    deduplicated -- present in `gpl_ids`.
    """
    ids = set(gpl_ids)
    pairs = {tuple(sorted((a, b))) for a, b in COMPANION_PLATFORMS.items() if a in ids and b in ids}
    return sorted(pairs)


def _normalize_title(title: str, gpl_id: str) -> str:
    text = title.lower()
    for token in _CHIP_SUBSTRING_TOKENS.get(gpl_id, []):
        text = text.replace(token, " ")
    text = re.sub(r"\s+", " ", text).strip()
    suffix_pattern = _CHIP_SUFFIX_TOKENS.get(gpl_id)
    if suffix_pattern:
        text = re.sub(suffix_pattern, "", text).strip()
    return text


def match_companion_samples(gse, gpl_a: str, gpl_b: str) -> dict[str, str] | None:
    """Best-effort 1:1 mapping {gsm_a_id: gsm_b_id} of samples that are the
    same physical biological sample split across gpl_a and gpl_b, matched by
    title with each platform's own chip-name token stripped first (e.g.
    "PaCa1_A" / "PaCa1_B" -> both "paca1"; "... U133A array" / "... U133B
    array" -> both "... array").

    Returns None -- not a partial/best-guess mapping -- unless every sample
    on both platforms pairs up cleanly one-to-one. An ambiguous, incomplete,
    or empty-title match means guessing wrong is worse than leaving the
    samples unmerged; callers should fall back to treating them as
    independent samples in that case.
    """
    a_samples = {
        gsm_id: gsm for gsm_id, gsm in gse.gsms.items() if gpl_a in gsm.metadata.get("platform_id", [])
    }
    b_samples = {
        gsm_id: gsm for gsm_id, gsm in gse.gsms.items() if gpl_b in gsm.metadata.get("platform_id", [])
    }
    if not a_samples or not b_samples or len(a_samples) != len(b_samples):
        return None

    def _key(gsm, gpl_id: str) -> str:
        titles = gsm.metadata.get("title", [])
        return _normalize_title(titles[0], gpl_id) if titles else ""

    a_by_key: dict[str, str] = {}
    for gsm_id, gsm in a_samples.items():
        key = _key(gsm, gpl_a)
        if not key or key in a_by_key:
            return None  # empty or duplicate normalized title -- not safe to match on
        a_by_key[key] = gsm_id

    pairing: dict[str, str] = {}
    for gsm_id, gsm in b_samples.items():
        key = _key(gsm, gpl_b)
        if not key or key not in a_by_key:
            return None
        pairing[a_by_key.pop(key)] = gsm_id

    if a_by_key:  # leftover A samples with no B match
        return None
    return pairing


def detect_pairings(gse) -> dict[tuple[str, str], dict[str, str]]:
    """{(gpl_a, gpl_b): {gsm_a_id: gsm_b_id}} for every companion pair
    present in this series that resolves to a clean 1:1 match. A pair
    that's present but doesn't cleanly match (see match_companion_samples)
    is simply absent from the result, not an error.
    """
    platform_ids = {gpl for gsm in gse.gsms.values() for gpl in gsm.metadata.get("platform_id", [])}
    result = {}
    for gpl_a, gpl_b in companion_platforms_present(platform_ids):
        pairing = match_companion_samples(gse, gpl_a, gpl_b)
        if pairing:
            result[(gpl_a, gpl_b)] = pairing
    return result


def combine_paired_probe_columns(probe_matrix: pd.DataFrame, pairing: dict[str, str]) -> pd.DataFrame:
    """From a probes x samples matrix with separate columns for each half of
    every companion pair in `pairing`, return a new probes x samples matrix
    where each pair's two columns are combined into one, named
    "<gsm_a>+<gsm_b>", by unioning their (almost entirely disjoint) probe
    values. The handful of probes both chips happen to share (Affymetrix's
    cross-chip reference/control probes) keep gsm_a's value -- an arbitrary
    but deterministic tie-break, since these are QC probes rather than
    genes of biological interest. Columns not part of any pair pass through
    unchanged.
    """
    if not pairing:
        return probe_matrix
    paired_cols = set(pairing.keys()) | set(pairing.values())
    combined = {col: probe_matrix[col] for col in probe_matrix.columns if col not in paired_cols}
    for gsm_a, gsm_b in pairing.items():
        combined[f"{gsm_a}+{gsm_b}"] = probe_matrix[gsm_a].combine_first(probe_matrix[gsm_b])
    return pd.DataFrame(combined)


def collapse_paired_samples(
    samples: pd.DataFrame, pairings: dict[tuple[str, str], dict[str, str]]
) -> pd.DataFrame:
    """Collapse `samples` (one row per raw GSM, e.g. from
    annotate.samples_table) so each companion-chip pair becomes one row:
    gsm_id "<gsm_a>+<gsm_b>", platform_id "<gpl_a>+<gpl_b>", every other
    column kept from gsm_a's row. Chip-pair rows are near-duplicates of each
    other by construction (same biological sample, same submitter, same
    metadata -- differing only in the chip-identifying text already
    stripped out during matching), so which half's row survives doesn't
    lose information.
    """
    if not pairings:
        return samples
    gsm_a_to_b: dict[str, str] = {}
    combined_platform: dict[str, str] = {}
    for (gpl_a, gpl_b), pairing in pairings.items():
        for gsm_a, gsm_b in pairing.items():
            gsm_a_to_b[gsm_a] = gsm_b
            combined_platform[gsm_a] = f"{gpl_a}+{gpl_b}"
    b_ids = set(gsm_a_to_b.values())

    rows = []
    for _, row in samples.iterrows():
        gsm_id = row["gsm_id"]
        if gsm_id in b_ids:
            continue  # represented by its pair partner's (the "a") row below
        if gsm_id in gsm_a_to_b:
            row = row.copy()
            row["gsm_id"] = f"{gsm_id}+{gsm_a_to_b[gsm_id]}"
            row["platform_id"] = combined_platform[gsm_id]
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)
