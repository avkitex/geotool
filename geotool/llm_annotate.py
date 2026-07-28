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

PROMPT_VERSION = "1"

FIXED_SAMPLE_COLUMNS = {
    "gsm_id", "title", "source_name_ch1", "organism_ch1", "molecule_ch1",
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

Classify every sample group given below. Use each group's fingerprint_id exactly as shown."""


def characteristic_columns(samples: pd.DataFrame) -> list[str]:
    return [c for c in samples.columns if c not in FIXED_SAMPLE_COLUMNS]


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


def build_prompt(gse, series_row: dict, groups: dict[str, dict]) -> tuple[str, str]:
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
        max_tokens=16000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        output_format=SeriesLLMResult,
    )
    return _extract_parsed(response)


def annotate_series(gse, series_row: dict, groups: dict[str, dict], model: str | None = None) -> SeriesLLMResult:
    system_prompt, user_prompt = build_prompt(gse, series_row, groups)
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

    out = samples.copy()
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
        result = annotate_series(gse, series_row, groups, model=model)
        if escalate_ambiguous:
            result = _escalate_ambiguous(gse, series_row, groups, result)
        _save_cache(cache_file, cache_key, result)

    merged = merge_annotations(samples, fp_ids, result)
    return merged, result.series_level
