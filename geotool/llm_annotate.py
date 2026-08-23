"""Per-series LLM annotation.

One Claude call per series covers every unique sample-characteristics
pattern in it (a "fingerprint group"), not one call per sample -- a series
with 200 samples but 8 distinct annotation patterns costs the same as one
with 8 samples. Results are cached to disk keyed by a hash of the series'
summary/characteristics/prompt-version/model, so re-running a search never
re-pays for a series it already classified.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import anthropic
import pandas as pd

from geotool import config, vocab
from geotool.llm_schema import SeriesLevelAnnotation, SeriesLLMResult

PROMPT_VERSION = "2"  # bumped: excluded-numeric-column / numeric_column_units section added to the prompt

FIXED_SAMPLE_COLUMNS = {
    "gsm_id", "gse_id", "title", "source_name_ch1", "organism_ch1", "molecule_ch1",
    "platform_id", "description", "library_selection", "library_strategy",
    "data_row_count", "rnaseq_library_type",
}

_SELECTION_HINT_RE = re.compile(r"sort|FACS|MACS|laser|microdissect|CD\d+\+?", re.IGNORECASE)

SYSTEM_PROMPT_TEMPLATE = """You are annotating a GEO gene expression series for a bioinformatics search tool.

GEO metadata is user-submitted and can be incomplete, inconsistent, or self-contradictory.
Cross-check the series summary's claims against the actual per-sample characteristics rather
than trusting either blindly. Use "unknown" instead of guessing when the evidence is thin, and
record any contradiction you notice between the summary and the characteristics in
inconsistency_notes (leave it "" if nothing is off).

For diagnosis, pick one of the following categories if it fits, otherwise use "other" (with
specifics in diagnosis_detail) or "unknown" if you cannot tell:
{diagnosis_categories}

Classify every sample group given below. Use each group's fingerprint_id exactly as shown.

If an "excluded numeric column(s)" section is given below, each one has too many distinct values
to group samples by, but is otherwise worth reading -- estimate its time unit (days/months/years)
from its column name and the value range/examples given, and report it in numeric_column_units.
GEO survival/follow-up columns are most often already in days, sometimes months, rarely years --
use "unknown" rather than guessing when genuinely ambiguous."""


# A characteristic column is only worth grouping samples by if it's
# categorical (a handful of repeated values -- grade, stage, tissue). A
# column with more distinct values than a real categorical trait would ever
# have fragments what would otherwise be a handful of real groups into
# nearly one group per sample -- regardless of whether it's a continuous
# numeric measurement (survival time, age), a per-patient identifier
# ("patient id", "sample id" -- a string, not numeric), or a real but
# unusually diverse categorical field (many distinct free-text treatment
# regimens). A raw count alone can't tell those apart from a column that's
# still genuinely categorical (repeated values, just a lot of them) -- live
# example: GSE71729's source_name_ch2 (21 distinct anatomical/metastasis
# sites + "CellLine" across 357 samples, each value repeated many times).
# Excluding it dropped the *only* signal identifying that cohort's 17
# cell-line samples, which then fingerprinted into two catch-all groups
# with nothing but constant/blank characteristics and got classified
# tissue_class=unknown/sample_source=unknown -- not wrong given what the
# LLM was shown, just never shown the column that mattered.
#
# So a column is only treated as high-cardinality when it's *both* over the
# raw count AND close to unique-per-sample (nunique/n_samples) -- the actual
# property that distinguishes an identifier/continuous measurement (values
# essentially never repeat) from a rich-but-real categorical field (values
# repeat constantly, there are just more distinct ones than a small trait
# like sex/grade would have). Three live examples that each crashed a
# harmonize run by overflowing _call_model's max_tokens before this
# existed (see harmonize.get_llm_annotation's own try/except for the other
# half of that fix), all comfortably above the fraction cutoff below --
#   - GSE183795's "survival months" (128 distinct / 244 samples, 52%)
#   - GSE93326's "patient id" (129 distinct / 204 samples, 63% -- removing
#     it collapses 204 fingerprint groups down to 3)
#   - GSE253260's "sample id"/"normpatient id" (397 distinct / 397 samples,
#     100%) and, even after those, still-large "firsttreatment" (64 distinct
#     regimens / 308 samples, 21%)
# -- versus GSE71729's source_name_ch2 at 21/357 = 6%, safely under it.
_MAX_CATEGORICAL_UNIQUE_VALUES = 12
_MAX_CATEGORICAL_UNIQUE_FRACTION = 0.15
_MIN_NUMERIC_FRACTION = 0.9


def _is_high_cardinality(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    n_unique = non_null.nunique()
    if n_unique <= _MAX_CATEGORICAL_UNIQUE_VALUES:
        return False
    return (n_unique / len(non_null)) > _MAX_CATEGORICAL_UNIQUE_FRACTION


def characteristic_columns(samples: pd.DataFrame) -> list[str]:
    return [
        c for c in samples.columns
        if c not in FIXED_SAMPLE_COLUMNS and not _is_high_cardinality(samples[c])
    ]


def _is_mostly_numeric(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    numeric = pd.to_numeric(non_null, errors="coerce")
    return numeric.notna().mean() >= _MIN_NUMERIC_FRACTION


# The subset of characteristic_columns' exclusions worth asking the LLM to
# estimate a time unit for -- something like "age", "firsttreatment", or
# "patient id" is excluded from the fingerprint for the same cardinality
# reason, but has no survival/follow-up time unit to estimate (and isn't
# even numeric, for the latter two).
_SURVIVAL_COLUMN_NAME_RE = re.compile(
    r"\b(os|pfs|dfs|rfs|efs|dss|css)\b|surviv|follow.?up|time.to.(death|event|progression|relapse|recurrence)",
    re.IGNORECASE,
)


def survival_like_numeric_columns(samples: pd.DataFrame) -> list[str]:
    return [
        c for c in samples.columns
        if c not in FIXED_SAMPLE_COLUMNS
        and _is_high_cardinality(samples[c])
        and _is_mostly_numeric(samples[c])
        and _SURVIVAL_COLUMN_NAME_RE.search(str(c))
    ]


def summarize_numeric_column(series: pd.Series) -> dict:
    """Compact summary (not the ~200+ raw values) for the LLM to estimate a
    survival column's time unit from -- name is given separately by the
    caller, this is just the value-range evidence."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    examples = series.dropna().astype(str).unique()[:5].tolist()
    if numeric.empty:
        return {"min": None, "max": None, "examples": examples}
    return {"min": float(numeric.min()), "max": float(numeric.max()), "examples": examples}


def fingerprint_key(characteristics: dict) -> str:
    """Stable hash of a sample's characteristics dict, for grouping identical samples."""
    canonical = json.dumps(characteristics, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _row_characteristics(row: pd.Series, cols: list[str]) -> dict:
    return {c: ("" if pd.isna(row[c]) else str(row[c])) for c in cols}


def group_fingerprints(samples: pd.DataFrame) -> tuple[pd.Series, dict[str, dict]]:
    """Group samples by identical characteristics.

    Returns (per-row fingerprint_id Series, {fingerprint_id: {characteristics, gsm_ids}}).
    """
    cols = characteristic_columns(samples)
    fp_ids = []
    groups: dict[str, dict] = {}
    for _, row in samples.iterrows():
        chars = _row_characteristics(row, cols)
        fp = fingerprint_key(chars)
        fp_ids.append(fp)
        groups.setdefault(fp, {"characteristics": chars, "gsm_ids": []})["gsm_ids"].append(row["gsm_id"])
    return pd.Series(fp_ids, index=samples.index, name="fingerprint_id"), groups


def _protocol_snippet(gse, gsm_id: str) -> str:
    gsm = gse.gsms.get(gsm_id) if gse is not None else None
    if gsm is None:
        return ""
    md = gsm.metadata
    text = " ".join(md.get("extract_protocol_ch1", []) + md.get("treatment_protocol_ch1", []))
    return text if _SELECTION_HINT_RE.search(text) else ""


def build_prompt(
    gse, series_row: dict, groups: dict[str, dict], numeric_columns: dict[str, dict] | None = None
) -> tuple[str, str]:
    categories = vocab.load_diagnosis_categories()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        diagnosis_categories="\n".join(f"- {c}" for c in categories)
    )

    lines = [
        f"Series accession: {series_row.get('gse_id', '')}",
        f"Title: {series_row.get('title', '')}",
        f"Summary: {series_row.get('summary', '')}",
        f"Overall design: {series_row.get('overall_design', '')}",
        "",
        f"{sum(len(g['gsm_ids']) for g in groups.values())} samples total, "
        f"grouped into {len(groups)} unique characteristics pattern(s):",
        "",
    ]
    for fp_id, group in groups.items():
        example_gsm = group["gsm_ids"][0]
        lines.append(f"### fingerprint_id: {fp_id} (n_samples={len(group['gsm_ids'])}, example={example_gsm})")
        lines.append(f"characteristics: {json.dumps(group['characteristics'])}")
        snippet = _protocol_snippet(gse, example_gsm)
        if snippet:
            lines.append(f"protocol notes: {snippet}")
        lines.append("")

    if numeric_columns:
        lines.append("Excluded numeric column(s) (too many distinct values to group by -- estimate each one's time unit instead, in numeric_column_units):")
        for col, info in numeric_columns.items():
            lines.append(f"- {col}: min={info['min']}, max={info['max']}, examples={info['examples']}")
        lines.append("")

    return system_prompt, "\n".join(lines)


def _extract_parsed(response) -> SeriesLLMResult:
    for block in response.content:
        if getattr(block, "type", None) == "text" and getattr(block, "parsed_output", None) is not None:
            return block.parsed_output
    raise ValueError("Claude response contained no parsed structured output")


def _call_model(system_prompt: str, user_prompt: str, model: str) -> SeriesLLMResult:
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=model,
        # A genuinely diverse cohort (many clinical centers/treatment regimens,
        # e.g. GSE253260's 64 distinct first-line therapies -- real signal for
        # prior_therapy, not something characteristic_columns' fingerprint
        # exclusions above should strip out) can still produce a large number
        # of groups even after those exclusions. Doubled from the original
        # 16000 as extra headroom; harmonize.get_llm_annotation's try/except
        # is still the actual safety net for whatever cohort is large enough
        # to blow past even this. Above ~21k, the SDK's own
        # _calculate_nonstreaming_timeout refuses a plain synchronous call
        # ("Streaming is required for operations that may take longer than
        # 10 minutes") *unless* a timeout is given explicitly -- that
        # calculation only runs "if not is_given(timeout)", so passing one
        # here sidesteps it entirely rather than reshaping this whole
        # function around a streamed response.
        max_tokens=32000,
        timeout=1200.0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        output_format=SeriesLLMResult,
    )
    return _extract_parsed(response)


def annotate_series(
    gse, series_row: dict, groups: dict[str, dict], samples: pd.DataFrame | None = None, model: str | None = None
) -> SeriesLLMResult:
    numeric_columns = None
    if samples is not None:
        numeric_cols = survival_like_numeric_columns(samples)
        if numeric_cols:
            numeric_columns = {col: summarize_numeric_column(samples[col]) for col in numeric_cols}
    system_prompt, user_prompt = build_prompt(gse, series_row, groups, numeric_columns)
    return _call_model(system_prompt, user_prompt, model or config.LLM_MODEL)


def _escalate_ambiguous(gse, series_row: dict, groups: dict[str, dict], result: SeriesLLMResult) -> SeriesLLMResult:
    ambiguous_fps = {
        g.fingerprint_id
        for g in result.sample_groups
        if g.diagnosis.strip().lower() == "unknown" or g.diagnosis_source == "ambiguous"
    }
    if not ambiguous_fps:
        return result

    sub_groups = {fp: g for fp, g in groups.items() if fp in ambiguous_fps}
    system_prompt, user_prompt = build_prompt(gse, series_row, sub_groups)
    escalated = _call_model(system_prompt, user_prompt, config.LLM_ESCALATION_MODEL)
    escalated_by_fp = {g.fingerprint_id: g for g in escalated.sample_groups}
    new_groups = [escalated_by_fp.get(g.fingerprint_id, g) for g in result.sample_groups]
    return SeriesLLMResult(series_level=result.series_level, sample_groups=new_groups)


def _normalize_species(organism_ch1: str) -> str:
    value = (organism_ch1 or "").strip().lower()
    if value == "homo sapiens":
        return "human"
    if value == "mus musculus":
        return "mouse"
    return "unknown" if not value else "other"


# Target unit is days, not clinical_annotate.py's months -- this module's
# numeric_column_units estimate comes from a column name + value range, not
# a definitive per-cohort column-mapping analysis, so days keeps the
# conversion a simple, reversible unit change rather than implying the same
# level of confidence as clinical_annotate.py's --clinical-annotate survival
# unification (time_column + event_column, opt-in, one dedicated call).
_TIME_UNIT_TO_DAYS = {"days": 1.0, "months": 30.4375, "years": 365.25}
_NUMBER_RE = re.compile(r"-?\d+\.?\d*")


def _extract_first_number(text) -> float:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return float("nan")
    match = _NUMBER_RE.search(str(text))
    return float(match.group()) if match else float("nan")


def _sanitize_column_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()


def apply_numeric_column_units(samples: pd.DataFrame, units: list) -> pd.DataFrame:
    """Add an llm_<column>_days column for each numeric_column_units entry
    with a resolved (non-"unknown") unit -- additive only, the original raw
    column is left untouched (same "never destroy, only add" convention as
    e.g. this codebase's RMA and two-channel outputs). Named with the same
    "llm_" prefix every other column this module derives uses, since
    harmonize.get_llm_annotation's non-backfill path only ever keeps
    columns matching that prefix when joining back onto a cohort's
    annotation.tsv -- anything else here would be silently dropped there.

    Rounded to whole days (nullable Int64, so a still-missing value stays
    missing rather than becoming 0/NaN-as-float) -- day-level granularity is
    already finer than any submitter's own follow-up precision, so the
    fractional remainder from a months/years->days conversion (e.g.
    56.02185641 months * 30.4375 = 1705.165...) is noise, not real
    precision, and a non-integer day count reads as if it were.
    """
    out = samples.copy()
    for unit_entry in units:
        col = unit_entry.column_name
        if col not in out.columns or unit_entry.unit == "unknown":
            continue
        factor = _TIME_UNIT_TO_DAYS[unit_entry.unit]
        numeric = pd.to_numeric(out[col], errors="coerce")
        needs_fallback = numeric.isna() & out[col].notna()
        if needs_fallback.any():
            numeric.loc[needs_fallback] = out[col][needs_fallback].map(_extract_first_number)
        out[f"llm_{_sanitize_column_name(col)}_days"] = (numeric * factor).round().astype("Int64")
    return out


def merge_annotations(samples: pd.DataFrame, fp_ids: pd.Series, result: SeriesLLMResult) -> pd.DataFrame:
    """Normalize diagnoses and join LLM sample-group annotations back onto `samples`."""
    categories = vocab.load_diagnosis_categories()
    rows = []
    for group in result.sample_groups:
        diagnosis, matched = vocab.normalize_diagnosis(group.diagnosis, categories)
        diagnosis_detail = group.diagnosis_detail
        if not matched:
            diagnosis_detail = f"(unmatched: {group.diagnosis}) {diagnosis_detail}".strip()
        rows.append({
            "fingerprint_id": group.fingerprint_id,
            "llm_sample_source": group.sample_source,
            "llm_tissue_class": group.tissue_class,
            "llm_tissue_detail": group.tissue_detail,
            "llm_selection_method": group.selection_method,
            "llm_selection_detail": group.selection_detail,
            "llm_diagnosis": diagnosis,
            "llm_diagnosis_detail": diagnosis_detail,
            "llm_diagnosis_source": group.diagnosis_source,
            "llm_prior_therapy": group.prior_therapy,
            "llm_prior_therapy_detail": group.prior_therapy_detail,
        })
    annotations = pd.DataFrame(rows)

    out = apply_numeric_column_units(samples, result.series_level.numeric_column_units)
    out["fingerprint_id"] = fp_ids
    out["llm_species"] = (
        out["organism_ch1"].map(_normalize_species) if "organism_ch1" in out.columns else "unknown"
    )
    out = out.merge(annotations, on="fingerprint_id", how="left")

    detail_cols = [c for c in annotations.columns if c.endswith("_detail")]
    category_cols = [c for c in annotations.columns if c not in detail_cols and c != "fingerprint_id"]
    for col in category_cols:
        out[col] = out[col].fillna("unknown")
    for col in detail_cols:
        out[col] = out[col].fillna("")
    return out


def _cache_path(gse_id: str, series_dir: Path | None) -> Path:
    return (series_dir or config.SERIES_DIR) / gse_id / "llm_annotations.json"


def _cache_key(series_row: dict, groups: dict[str, dict], model: str) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "summary": series_row.get("summary", ""),
        "overall_design": series_row.get("overall_design", ""),
        "groups": {fp: g["characteristics"] for fp, g in groups.items()},
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache(path: Path, cache_key: str, result: SeriesLLMResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"cache_key": cache_key, "result": result.model_dump()}, f, indent=2)


def annotate_and_cache(
    gse,
    series_row: dict,
    samples: pd.DataFrame,
    series_dir: Path | None = None,
    model: str | None = None,
    escalate_ambiguous: bool = False,
) -> tuple[pd.DataFrame, SeriesLevelAnnotation]:
    model = model or config.LLM_MODEL
    gse_id = series_row["gse_id"]
    fp_ids, groups = group_fingerprints(samples)
    cache_key = _cache_key(series_row, groups, model)
    cache_file = _cache_path(gse_id, series_dir)

    cached = _load_cache(cache_file)
    if cached is not None and cached.get("cache_key") == cache_key:
        result = SeriesLLMResult.model_validate(cached["result"])
    else:
        result = annotate_series(gse, series_row, groups, samples=samples, model=model)
        if escalate_ambiguous:
            result = _escalate_ambiguous(gse, series_row, groups, result)
        _save_cache(cache_file, cache_key, result)

    merged = merge_annotations(samples, fp_ids, result)
    return merged, result.series_level
