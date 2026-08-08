"""Match an RNA-seq expression matrix's sample column names back to GEO
sample (gsm_id) identifiers.

`geotool download` always keys a cohort's own annotation.tsv by gsm_id --
GEO's own stable sample accession. But the expression matrix a submitter
actually publishes (often an arbitrary supplementary .xlsx/.tsv, never GEO's
own SOFT format) almost never uses gsm_id as its column headers -- it uses
whatever short label the submitter used internally (e.g. "DMSO_1",
"D5_EPZ.r1", "dmso1"). Nothing in a GEO record formally links the two
together; the only real anchors are (a) that label usually appears verbatim,
or as a recognizable substring, somewhere in that GSM's own title/
description/characteristics text, and (b) GEO's own SOFT records list
samples in the same order the submitter organized them in, which is
overwhelmingly (not always) also the order columns appear in a
submitter-authored matrix.

match_expression_columns tries several strategies, cheapest/most-certain
first, falling back to sample order only for whatever's left unresolved
by name -- and only when the *count* of what's left matches exactly, so a
genuine mismatch (extra/missing samples) surfaces as unmatched rather than
a wrong guess. Every row is annotated with which strategy resolved it (or
"unmatched"), so a human can see at a glance how much to trust the result --
this is a best-effort matcher, not a guarantee.
"""
from __future__ import annotations

import re

import pandas as pd

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# Structural/identifier columns that are never themselves a sample's
# human-readable label -- excluded from the text columns searched for a
# match (matching "DMSO_1" against a platform_id like "GPL24676" is never
# useful and only risks an accidental substring hit).
_EXCLUDED_TEXT_COLUMNS = {"gsm_id", "gse_id", "platform_id", "expression_status"}

# Confidence scores are relative, not calibrated probabilities -- just a
# way to rank/threshold match quality for a human skimming the output.
_METHOD_CONFIDENCE = {
    "exact_gsm_id": 1.0,
    "exact": 0.95,
    "normalized_exact": 0.9,
    "substring": 0.75,
    "reverse_substring": 0.6,
    "positional_fallback": 0.5,
}


def _normalize(value) -> str:
    return _NON_ALNUM_RE.sub("", str(value).lower())


def _text_columns(annotation: pd.DataFrame) -> list[str]:
    # pandas 3.x gives plain string columns their own "str" dtype rather
    # than "object" -- check is_string_dtype (which covers both) and
    # explicitly exclude numeric columns, since is_string_dtype's docs
    # warn it can also be true for object-dtype columns that merely
    # *could* hold strings but actually hold something else.
    return [
        c for c in annotation.columns
        if c not in _EXCLUDED_TEXT_COLUMNS
        and pd.api.types.is_string_dtype(annotation[c])
        and not pd.api.types.is_numeric_dtype(annotation[c])
    ]


def match_expression_columns(
    expression_columns: list[str], annotation: pd.DataFrame, text_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Match each of expression_columns to a gsm_id in annotation (must have
    a gsm_id column). Returns one row per expression column: expression_id,
    gsm_id (None if unresolved), match_method, confidence.

    text_columns overrides which annotation columns are searched for a
    name-based match (default: every object-dtype column except gsm_id/
    gse_id/platform_id/expression_status). A match is only accepted when it
    identifies exactly one gsm_id -- an ambiguous match (two samples with
    the same or overlapping label) is left unresolved rather than guessed.
    """
    text_columns = text_columns if text_columns is not None else _text_columns(annotation)
    normalized_cache = {col: annotation[col].astype(str).map(_normalize) for col in text_columns}

    rows = []
    used_gsm_ids: set[str] = set()
    for expr_col in expression_columns:
        gsm_id, method = _match_one(expr_col, annotation, text_columns, normalized_cache)
        if gsm_id is not None:
            used_gsm_ids.add(gsm_id)
        rows.append({
            "expression_id": expr_col,
            "gsm_id": gsm_id,
            "match_method": method,
            "confidence": _METHOD_CONFIDENCE.get(method, 0.0),
        })

    _apply_positional_fallback(rows, annotation, used_gsm_ids)
    return pd.DataFrame(rows, columns=["expression_id", "gsm_id", "match_method", "confidence"])


def _match_one(expr_col, annotation, text_columns, normalized_cache):
    if "gsm_id" in annotation.columns:
        mask = annotation["gsm_id"] == expr_col
        if mask.sum() == 1:
            return annotation.loc[mask, "gsm_id"].iloc[0], "exact_gsm_id"

    for col in text_columns:
        mask = annotation[col].astype(str) == expr_col
        if mask.sum() == 1:
            return annotation.loc[mask, "gsm_id"].iloc[0], "exact"

    norm_expr = _normalize(expr_col)
    if not norm_expr:
        return None, "unmatched"

    for col in text_columns:
        mask = normalized_cache[col] == norm_expr
        if mask.sum() == 1:
            return annotation.loc[mask, "gsm_id"].iloc[0], "normalized_exact"

    for col in text_columns:
        mask = normalized_cache[col].str.contains(norm_expr, regex=False)
        if mask.sum() == 1:
            return annotation.loc[mask, "gsm_id"].iloc[0], "substring"

    for col in text_columns:
        mask = normalized_cache[col].apply(lambda v: bool(v) and v in norm_expr)
        if mask.sum() == 1:
            return annotation.loc[mask, "gsm_id"].iloc[0], "reverse_substring"

    return None, "unmatched"


def _apply_positional_fallback(rows: list[dict], annotation: pd.DataFrame, used_gsm_ids: set[str]) -> None:
    """For whatever's still unresolved after name-based matching, fall back
    to GEO submission order -- but only when the count of unresolved
    columns exactly equals the count of not-yet-used gsm_ids, so a genuine
    sample-count mismatch surfaces as unmatched rather than a silently wrong
    positional guess.
    """
    unresolved_idx = [i for i, r in enumerate(rows) if r["gsm_id"] is None]
    if not unresolved_idx:
        return
    remaining_gsm_ids = [g for g in annotation["gsm_id"] if g not in used_gsm_ids]
    if len(remaining_gsm_ids) != len(unresolved_idx):
        return
    for idx, gsm_id in zip(unresolved_idx, remaining_gsm_ids):
        rows[idx]["gsm_id"] = gsm_id
        rows[idx]["match_method"] = "positional_fallback"
        rows[idx]["confidence"] = _METHOD_CONFIDENCE["positional_fallback"]
