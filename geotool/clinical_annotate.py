"""Clean and semantically unify a cohort's per-sample annotation table.

One Claude call per cohort (column headers + a few example values only, not
full per-sample data) plans the cleanup; everything else -- RECIST
normalization, event-value remapping, redundant-column dropping -- is
deterministic code. Same understand-with-LLM/apply-with-code split as
geotool.nl_query / geotool.llm_annotate.
"""
from __future__ import annotations

import re
from typing import Literal

import anthropic
import pandas as pd
from pydantic import BaseModel, Field

from geotool import config

# Identity/provenance columns: never dropped by the constant-column rule,
# even though they're constant within one cohort's file -- they're exactly
# what keeps rows traceable once multiple cohorts' tables get concatenated.
PROTECTED_COLUMNS = {"gsm_id", "gse_id", "platform_id"}

EXAMPLE_VALUES_PER_COLUMN = 5

# Word-enum values for the expression_status column (see
# classify_expression_status) -- a cohort-level status added by download.py
# after apply_column_mapping, not something the LLM plan ever sees. Not
# mutually exclusive: NOT_LOG2_TRANSFORMED and NEGATIVE_VALUES can both apply
# to the same matrix, joined with ";" (see classify_expression_status).
EXPRESSION_STATUS_OK = "ok"
EXPRESSION_STATUS_NO_MATRIX = "no_expression_matrix"
EXPRESSION_STATUS_UNPARSEABLE = "unparseable"
EXPRESSION_STATUS_NOT_LOG2_TRANSFORMED = "not_log2_transformed"
EXPRESSION_STATUS_NEGATIVE_VALUES = "negative_values"
EXPRESSION_STATUS_LOOKS_TRANSPOSED = "looks_transposed"
EXPRESSION_STATUS_LOW_GENE_COUNT = "low_gene_count"


def classify_expression_status(qc_notes: list[str], has_matrix: bool) -> str:
    """Word-enum summary of download.py's expression-matrix QC findings
    (probe_mapping.check_expression_qc / check_rnaseq_expression_qc), for the
    expression_status column in annotation.tsv -- lets a human or downstream
    script filter/flag a cohort with no usable expression matrix at all (e.g.
    GSE108651, which only ever published differential-expression/splicing-
    analysis output, never a raw or normalized matrix), or a matrix with a
    known scale/sign problem, without re-deriving it from qc_notes' free text.

    has_matrix is False whenever download.py never got a usable matrix at
    all -- no file its select_primary_expression_file recognized (RNA-seq),
    no probe->gene mapping available (microarray), or an unhandled assay
    type -- in which case there's nothing left to check and the other
    qc_notes-derived tags don't apply.
    """
    if not has_matrix:
        return EXPRESSION_STATUS_NO_MATRIX
    if any("could not parse" in note for note in qc_notes):
        return EXPRESSION_STATUS_UNPARSEABLE

    tags = []
    if any("not log2-transformed" in note for note in qc_notes):
        tags.append(EXPRESSION_STATUS_NOT_LOG2_TRANSFORMED)
    if any("negative value" in note for note in qc_notes):
        tags.append(EXPRESSION_STATUS_NEGATIVE_VALUES)
    if any("may be transposed" in note for note in qc_notes):
        tags.append(EXPRESSION_STATUS_LOOKS_TRANSPOSED)
    if any("filtered/truncated gene list" in note for note in qc_notes):
        tags.append(EXPRESSION_STATUS_LOW_GENE_COUNT)
    return ";".join(tags) if tags else EXPRESSION_STATUS_OK

SYSTEM_PROMPT = """You are cleaning up a GEO series' per-sample annotation table for a
bioinformatics search tool. You are given every column name and a few example values from
each -- not the full data.

Identify:
- redundant_columns: columns that are constant across all samples, or exact duplicates of
  another column -- safe to drop.
- treatment_columns: column(s) whose content cleanly folds into a single "treatment" value
  (e.g. drug/regimen name).
- treatment_detail_columns: treatment-related columns that do NOT cleanly fold into a single
  value (dosing schedules, free-text notes, multi-line regimens) -- keep these separate rather
  than forcing them into treatment_columns or dropping them.
- response_column: the column with the raw treatment response call, if any (e.g. RECIST
  response, tumor response, clinical response).
- survival: one entry per survival metric actually present (OS, PFS, DFS, RFS, EFS, or any
  other named type -- do not invent ones that aren't there). Each entry MUST have both a
  time_column and an event_column -- a bare time column with no censoring/event information is
  not usable survival data and should not be added. State event_value_meaning in plain
  language from the example values you were given, e.g. "1=death, 0=alive/censored" or
  "Dead=event, Alive=censored".

Only report what's actually present in the given columns -- never invent a column name."""


class SurvivalMapping(BaseModel):
    survival_type: str = Field(description='e.g. "OS", "PFS", "DFS", "RFS" -- whatever is actually present')
    time_column: str | None = None
    time_unit: Literal["days", "months", "years", "unknown"] = "unknown"
    event_column: str | None = None
    event_value_meaning: str = Field(default="", description='e.g. "1=death, 0=censored"')


class ColumnMappingPlan(BaseModel):
    redundant_columns: list[str] = Field(default_factory=list)
    treatment_columns: list[str] = Field(default_factory=list)
    treatment_detail_columns: list[str] = Field(default_factory=list)
    response_column: str | None = None
    survival: list[SurvivalMapping] = Field(default_factory=list)
    notes: str = ""


def _extract_parsed(response, output_type):
    for block in response.content:
        if getattr(block, "type", None) == "text" and getattr(block, "parsed_output", None) is not None:
            return block.parsed_output
    raise ValueError(f"Claude response contained no parsed {output_type.__name__}")


def _column_summary(samples_df: pd.DataFrame) -> str:
    lines = []
    for col in samples_df.columns:
        examples = samples_df[col].dropna().astype(str).unique()[:EXAMPLE_VALUES_PER_COLUMN]
        lines.append(f"- {col}: {', '.join(examples) if len(examples) else '(all empty)'}")
    return "\n".join(lines)


def plan_column_mapping(samples_df: pd.DataFrame, model: str | None = None) -> ColumnMappingPlan:
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=model or config.LLM_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _column_summary(samples_df)}],
        output_format=ColumnMappingPlan,
    )
    return _extract_parsed(response, ColumnMappingPlan)


_RECIST_PATTERNS = [
    (re.compile(r"complete response|^\s*cr\s*$", re.IGNORECASE), "CR"),
    (re.compile(r"partial response|^\s*pr\s*$", re.IGNORECASE), "PR"),
    (re.compile(r"stable disease|^\s*sd\s*$", re.IGNORECASE), "SD"),
    (re.compile(r"progressive disease|progression|^\s*pd\s*$", re.IGNORECASE), "PD"),
    # NE (not evaluable) and NA (not available) mean the same thing here -- one code, not two.
    (re.compile(r"not evaluable|not available|^\s*(ne|na|n/a)\s*$", re.IGNORECASE), "NE"),
]


def normalize_recist(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    for pattern, code in _RECIST_PATTERNS:
        if pattern.search(text):
            return code
    return None


_EVENT_LABEL_RE = re.compile(r"event|death|dead|deceased|progress|relapse|recur", re.IGNORECASE)
_CENSORED_LABEL_RE = re.compile(r"censor|alive|surviv|no event|free", re.IGNORECASE)


def parse_event_mapping(meaning: str) -> dict[str, int]:
    """Parse free text like 'X=event, Y=censored' or '1=death, 0=censored'
    into {lowercased_raw_value: 0_or_1}. Unparseable chunks are skipped."""
    mapping: dict[str, int] = {}
    if not meaning:
        return mapping
    for chunk in re.split(r"[,;]", meaning):
        if "=" not in chunk:
            continue
        raw_value, label = chunk.split("=", 1)
        raw_value = raw_value.strip().strip("\"'").lower()
        label = label.strip()
        if not raw_value:
            continue
        if _EVENT_LABEL_RE.search(label):
            mapping[raw_value] = 1
        elif _CENSORED_LABEL_RE.search(label):
            mapping[raw_value] = 0
    return mapping


def remap_event_column(series: pd.Series, meaning: str) -> pd.Series:
    """Map raw event values to 0/1. Many GEO characteristic values repeat
    their own label (e.g. 'Follow up status: DEAD', from a raw 'clinical
    info_3: Follow up status: DEAD' characteristic line), so an exact match
    against 'dead' would miss it -- fall back to a substring match, longest
    mapped key first so e.g. a hypothetical 'no event' key would be checked
    before 'event'.
    """
    mapping = parse_event_mapping(meaning)
    if not mapping:
        return series  # left as raw; caller records this in notes
    keys_longest_first = sorted(mapping, key=len, reverse=True)

    def _map(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip().lower()
        if text in mapping:
            return mapping[text]
        for key in keys_longest_first:
            if key in text:
                return mapping[key]
        return value  # unmapped values pass through raw

    return series.map(_map)


_TIME_UNIT_TO_MONTHS = {"days": 1 / 30.4375, "months": 1.0, "years": 12.0}
_NUMBER_RE = re.compile(r"-?\d+\.?\d*")


def _extract_first_number(text) -> float:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return float("nan")
    match = _NUMBER_RE.search(str(text))
    return float(match.group()) if match else float("nan")


def convert_time_to_months(series: pd.Series, time_unit: str) -> pd.Series:
    """Numeric-parse the time column, falling back to extracting the first
    number from strings like 'Follow up years: 5.2' (same repeated-label
    pattern as remap_event_column) when direct parsing fails. Uses NaN (not
    None) throughout so the column stays float64 -- pandas 3.x rejects
    assigning None into a float64 slice.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    needs_fallback = numeric.isna() & series.notna()
    if needs_fallback.any():
        numeric.loc[needs_fallback] = series[needs_fallback].map(_extract_first_number)
    factor = _TIME_UNIT_TO_MONTHS.get(time_unit)
    return numeric if factor is None else numeric * factor


_LABEL_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 /_-]{1,40}:\s+")


def strip_repeated_label(value):
    """Some GEO characteristics embed a repeated label in the value itself
    (e.g. value = 'Chemotherapy: CHOP-Like Regimen', from a raw 'clinical
    info_5: Chemotherapy: CHOP-Like Regimen' characteristic line, since
    annotate.parse_characteristics only splits on the first colon). Strip one
    such leading 'Label: ' prefix so the value reads cleanly.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    return _LABEL_PREFIX_RE.sub("", str(value), count=1)


def apply_column_mapping(samples_df: pd.DataFrame, plan: ColumnMappingPlan) -> pd.DataFrame:
    df = samples_df.copy()
    notes: list[str] = [plan.notes] if plan.notes else []

    # 1. Drop redundant + constant (non-protected) columns.
    to_drop = {c for c in plan.redundant_columns if c in df.columns and c not in PROTECTED_COLUMNS}
    for col in df.columns:
        if col in PROTECTED_COLUMNS or col in to_drop:
            continue
        if df[col].nunique(dropna=False) <= 1:
            to_drop.add(col)
    df = df.drop(columns=list(to_drop))

    # 1b. Strip repeated "Label: " prefixes from every remaining non-identity
    #     column's values -- benefits treatment/response/survival built below
    #     as well as any raw characteristic column that isn't otherwise unified.
    for col in df.columns:
        if col in PROTECTED_COLUMNS:
            continue
        df[col] = df[col].map(strip_repeated_label)

    # 2. treatment / treatment_detail -- coalesce what cleanly folds, keep the
    #    rest as free text rather than forcing a bad fit or losing it.
    #    Checking pd.notna() on the *original* values (rather than relying on
    #    .astype(str) turning a missing value into the literal string "nan",
    #    then filtering that out) matters because pandas' nullable string
    #    dtype leaves an actual missing value as a real float NaN even after
    #    .astype(str) -- and float('nan') is truthy, so an `if v` guard alone
    #    doesn't catch it, only to crash moments later on v.lower().
    treatment_cols = [c for c in plan.treatment_columns if c in df.columns]
    if treatment_cols:
        df["treatment"] = df[treatment_cols].apply(
            lambda row: "; ".join(str(v) for v in row if pd.notna(v)), axis=1
        )
        df = df.drop(columns=[c for c in treatment_cols if c != "treatment"])

    detail_cols = [c for c in plan.treatment_detail_columns if c in df.columns]
    if detail_cols:
        df["treatment_detail"] = df[detail_cols].apply(
            lambda row: "; ".join(str(v) for v in row if pd.notna(v)), axis=1
        )
        df = df.drop(columns=[c for c in detail_cols if c != "treatment_detail"])

    # 3. response / recist (RECIST normalized deterministically, not trusted to the model)
    if plan.response_column and plan.response_column in df.columns:
        df["response"] = df[plan.response_column]
        df["recist"] = df["response"].map(normalize_recist)
        if plan.response_column != "response":
            df = df.drop(columns=[plan.response_column])

    # 4. survival -- always a time+event PAIR, never emitted alone.
    for sm in plan.survival:
        if not sm.time_column or not sm.event_column:
            continue
        if sm.time_column not in df.columns or sm.event_column not in df.columns:
            continue
        time_col_name = f"{sm.survival_type}_time"
        event_col_name = f"{sm.survival_type}_event"
        df[time_col_name] = convert_time_to_months(df[sm.time_column], sm.time_unit)
        df[event_col_name] = remap_event_column(df[sm.event_column], sm.event_value_meaning)
        if not parse_event_mapping(sm.event_value_meaning):
            notes.append(
                f"{event_col_name}: could not parse event_value_meaning "
                f"({sm.event_value_meaning!r}); left raw"
            )
        drop_candidates = {sm.time_column, sm.event_column} - {time_col_name, event_col_name}
        df = df.drop(columns=[c for c in drop_candidates if c in df.columns])

    if notes:
        df.attrs["clinical_annotate_notes"] = "; ".join(n for n in notes if n)

    return df
