"""Cross-cohort column-concept matching for phase 3 harmonization.

harmonize.py already gives each cohort's treatment/response/recist/survival/
sex/age/tissue/llm_* columns a stable name *within* that cohort. What's left
after concatenating cohorts is the long tail of raw characteristic columns
that earlier processing didn't touch -- and different cohorts routinely name
the same concept differently (a DLBCL cell-of-origin call might show up as
"COO (Hans algorithm)" in one cohort and "GCB/ABC classifier" in another),
with the values spelled differently too.

Two Claude calls, not one. The first, over the combined table's surviving
columns (names + which cohorts reported them + example values), proposes
clusters of columns that represent the same concept and a canonical name
for each. Live testing showed this same call unreliable at also populating
value_mapping -- it would describe the needed value remap in `notes` and
then leave value_mapping empty, on both the cheap and the escalation model.
So a second, narrower call runs per multi-column cluster: given the
complete, deterministically-enumerated list of raw values for just that one
concept, map every one of them. That's a much smaller, closed task, and the
cheap model handles it reliably. Same understand-with-LLM/apply-with-code
split as clinical_annotate.py and llm_annotate.py either way -- applying the
plan is deterministic and never drops a column or value it doesn't have a
mapping for.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anthropic
import pandas as pd
from pydantic import BaseModel, Field

from geotool import config

PROMPT_VERSION = "4"

MAX_VALUES_PER_CONCEPT = 60

EXAMPLE_VALUES_PER_COLUMN = 8

_DEFAULT_CLINICAL_CONCEPTS_PATH = config.PROJECT_ROOT / "geotool" / "vocab_data" / "clinical_concepts.json"

SYSTEM_PROMPT_TEMPLATE = """You are unifying per-sample clinical/characteristic columns across several GEO
cohorts that have already been combined into one table, for a bioinformatics cohort-harmonization tool.

The combined samples are annotated with the following diagnosis/diagnoses: {diagnosis_context}.

You are given every column that earlier, cheaper processing did NOT already give a stable cross-cohort name
(treatment, response, recist, sex, age, tissue, survival time/event pairs, and LLM-derived tissue/diagnosis
columns are all already unified and are not shown here). For each remaining column you get: its exact name,
which cohort(s) (gse_id) reported it, and its most common values with counts.

Different cohorts routinely name the same underlying concept differently -- e.g. "COO (Hans algorithm)",
"GCB/ABC classifier", and "cell of origin" can all be the same DLBCL biomarker.
{seed_hints}
{master_hints}
For every genuine match you find between two or more columns from DIFFERENT cohorts describing the same
concept, report one cluster with:
- canonical_name: a short snake_case name for the unified column
- source_columns: the exact column names given above (verbatim) to merge into it

Leave value_mapping empty here -- a separate, focused pass unifies each concept's raw value spellings
afterward, so don't spend effort on it in this step.

Only cluster columns that truly represent the same concept -- when in doubt, leave a column out of every
cluster rather than force a bad match. Never invent a column name that wasn't given to you."""


class ColumnCluster(BaseModel):
    canonical_name: str
    source_columns: list[str] = Field(default_factory=list)
    value_mapping: dict[str, str] = Field(
        default_factory=dict, description="raw value (as shown) -> unified canonical value"
    )
    notes: str = ""


class CrossCohortMappingPlan(BaseModel):
    clusters: list[ColumnCluster] = Field(default_factory=list)
    notes: str = ""


class ValueMappingEntry(BaseModel):
    raw_value: str = Field(description="one of the exact raw values given, copied verbatim")
    canonical_value: str = Field(description="the unified canonical label for this raw value")


class ValueUnificationPlan(BaseModel):
    # A list of {raw_value, canonical_value} entries, not a dict[str, str] -- live testing
    # showed models reliably satisfice with an empty dict (trivially schema-valid, since a
    # dict[str, str] field has no required keys) instead of actually populating it, across
    # both the cheap and escalation model, and with or without a sibling `notes` field to
    # dump the real answer into instead. A list of fixed-shape entries doesn't offer that
    # shortcut, and testing confirmed it reliably gets populated.
    mappings: list[ValueMappingEntry] = Field(default_factory=list)

    @property
    def value_mapping(self) -> dict[str, str]:
        return {entry.raw_value: entry.canonical_value for entry in self.mappings}


def load_clinical_concepts(path: Path | None = None) -> dict:
    with open(path or _DEFAULT_CLINICAL_CONCEPTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def clusterable_columns(df: pd.DataFrame, protected: set[str]) -> list[str]:
    return [c for c in df.columns if c not in protected]


def diagnosis_context(df: pd.DataFrame) -> str:
    """Diagnosis/diagnoses this batch is about, from llm_diagnosis(_detail) if
    present -- assumed to be a single diagnosis or a close family of them, so
    one shared string covers the whole matching call rather than per-cohort.
    """
    parts: list[str] = []
    for col in ("llm_diagnosis", "llm_diagnosis_detail"):
        if col not in df.columns:
            continue
        values = sorted({
            str(v).strip() for v in df[col].dropna().unique()
            if str(v).strip() and str(v).strip().lower() != "unknown"
        })
        parts.extend(v for v in values if v not in parts)
    return ", ".join(parts) if parts else "unknown"


def _normalize_diagnosis_key(key: str) -> str:
    return key.strip().lower().replace("_", " ")


def seed_hint_text(diagnosis_ctx: str, concepts: dict | None = None) -> str:
    """Optional prior knowledge for well-known disease-specific concepts
    (COO, IGHV status, PAM50, ...) -- injected only when the batch's
    diagnosis context plausibly matches. The model isn't restricted to these;
    it's free to find and merge other concepts too.
    """
    concepts = concepts if concepts is not None else load_clinical_concepts()
    if not diagnosis_ctx or diagnosis_ctx.strip().lower() == "unknown":
        return ""
    ctx_lower = diagnosis_ctx.lower()
    lines = []
    for diagnosis, fields in concepts.items():
        if _normalize_diagnosis_key(diagnosis) not in ctx_lower and diagnosis.lower() not in ctx_lower:
            continue
        for field_name, values in fields.items():
            lines.append(
                f'- For {diagnosis}, a commonly reported concept is "{field_name}" with canonical values: '
                f'{", ".join(values)}.'
            )
    if not lines:
        return ""
    return "Known disease-specific concepts that may appear under different raw column names:\n" + "\n".join(lines)


def build_column_summary(
    df: pd.DataFrame, columns: list[str], examples_per_column: int = EXAMPLE_VALUES_PER_COLUMN
) -> str:
    has_gse = "gse_id" in df.columns
    lines = []
    for col in columns:
        non_null = df[col].dropna()
        cohorts = sorted(df.loc[non_null.index, "gse_id"].astype(str).unique()) if has_gse else []
        counts = non_null.astype(str).value_counts().head(examples_per_column)
        examples = ", ".join(f"{value!r} (n={n})" for value, n in counts.items())
        cohort_label = ", ".join(cohorts) if cohorts else "unknown"
        lines.append(f"- {col} [cohorts: {cohort_label}]: {examples or '(all empty)'}")
    return "\n".join(lines)


def master_columns_summary(
    master_df: pd.DataFrame | None, protected: set[str], examples_per_column: int = 5
) -> str:
    if master_df is None:
        return ""
    columns = [c for c in master_df.columns if c not in protected]
    if not columns:
        return ""
    return build_column_summary(master_df, columns, examples_per_column=examples_per_column)


def _master_hint_block(master_summary: str) -> str:
    if not master_summary:
        return ""
    return (
        "A master annotation table from prior cohorts already defines these canonical columns -- reuse one "
        "of these exact names and its value vocabulary whenever a new column expresses the same concept, "
        "even if only one of the columns below matches it (report it as a one-column cluster so it renames "
        "onto the existing canonical name); only propose a new canonical_name for a concept genuinely not "
        "covered here:\n" + master_summary + "\n"
    )


def _extract_parsed(response) -> CrossCohortMappingPlan:
    for block in response.content:
        if getattr(block, "type", None) == "text" and getattr(block, "parsed_output", None) is not None:
            return block.parsed_output
    raise ValueError("Claude response contained no parsed CrossCohortMappingPlan")


def plan_column_clusters(
    df: pd.DataFrame,
    columns: list[str],
    diagnosis_ctx: str,
    seed_hints: str = "",
    master_summary: str = "",
    model: str | None = None,
) -> CrossCohortMappingPlan:
    """Defaults to config.LLM_ESCALATION_MODEL, not the cheap per-series
    default -- unlike llm_annotate's classification (one call per series,
    many times over), this is one call per harmonize run, and recognizing
    that two differently-named columns from different cohorts describe the
    same concept is a harder semantic-matching task than per-series
    classification.
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        diagnosis_context=diagnosis_ctx,
        seed_hints=seed_hints,
        master_hints=_master_hint_block(master_summary),
    )
    user_prompt = build_column_summary(df, columns)
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=model or config.LLM_ESCALATION_MODEL,
        # 16000, not clinical_annotate's 2048 -- config.LLM_ESCALATION_MODEL emits an
        # extended-thinking block first, which otherwise eats the whole budget before any
        # text/structured output (found live: stop_reason "max_tokens", zero parsed output).
        max_tokens=16000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        output_format=CrossCohortMappingPlan,
    )
    return _extract_parsed(response)


VALUE_PROMPT_TEMPLATE = """You are unifying the spelling of raw values for one clinical/biomarker concept,
"{canonical_name}", across GEO cohorts that have already been identified as reporting the same underlying
concept (diagnosis context: {diagnosis_context}).
{seed_hints}
Below is every distinct raw value seen for this concept, with its count. Different cohorts often spell the
same value differently -- case, abbreviation vs. spelled-out form (e.g. "GCB" vs "germinal center"), an added
disease-name suffix (e.g. "GCB" vs "GCB DLBCL"), or typos (e.g. "unkown"). Map EVERY value below to one unified
canonical label -- two raw values that mean the same thing MUST map to the identical label. A value that is
genuinely distinct (not a spelling variant of another) can map to itself, cleaned up if needed.

{value_lines}

Return one entry in `mappings` for every raw value listed above -- do not omit any, and do not invent values
that aren't listed. Copy raw_value verbatim from the quoted value shown (without the count)."""


def _extract_value_plan(response) -> ValueUnificationPlan:
    for block in response.content:
        if getattr(block, "type", None) == "text" and getattr(block, "parsed_output", None) is not None:
            return block.parsed_output
    raise ValueError("Claude response contained no parsed ValueUnificationPlan")


def plan_value_mapping(
    canonical_name: str,
    value_counts: pd.Series,
    diagnosis_ctx: str,
    seed_hints: str = "",
    model: str | None = None,
) -> ValueUnificationPlan:
    """One narrow call per concept: given the complete, deterministically-
    enumerated list of raw values for just this one concept, map every one
    of them. The cheap default model is reliable here -- it's a closed
    enumeration task, unlike remembering to also do this mid-way through the
    bigger, open-ended column-clustering call in plan_column_clusters.
    """
    value_lines = "\n".join(f"- {value!r} (n={n})" for value, n in value_counts.head(MAX_VALUES_PER_CONCEPT).items())
    system_prompt = VALUE_PROMPT_TEMPLATE.format(
        canonical_name=canonical_name, diagnosis_context=diagnosis_ctx, seed_hints=seed_hints, value_lines=value_lines,
    )
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=model or config.LLM_MODEL,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": "Return the value_mapping now."}],
        output_format=ValueUnificationPlan,
    )
    return _extract_value_plan(response)


def _cluster_value_counts(df: pd.DataFrame, present_cols: list[str]) -> pd.Series | None:
    """Raw value counts across a cluster's present source columns, or None
    if there's nothing worth unifying (fewer than two distinct values, or
    the cluster is a single-column rename with nothing to merge)."""
    if len(present_cols) < 2:
        return None
    values = pd.concat([df[c] for c in present_cols]).dropna().astype(str)
    if values.nunique() < 2:
        return None
    return values.value_counts()


def _mapping_covers(value_mapping: dict[str, str], value_counts: pd.Series) -> bool:
    if not value_mapping:
        return False
    covered = {str(k).strip().lower() for k in value_mapping}
    return all(str(v).strip().lower() in covered for v in value_counts.index)


def fill_value_mappings(
    df: pd.DataFrame,
    plan: CrossCohortMappingPlan,
    diagnosis_ctx: str,
    seed_hints: str = "",
    model: str | None = None,
) -> CrossCohortMappingPlan:
    """Backfill value_mapping for every multi-column cluster plan_column_clusters
    left incomplete, via a separate plan_value_mapping call per such cluster.
    Clusters that already fully cover their values, or that merge fewer than
    two present source columns (a rename, or a column that vanished), are
    left untouched -- no wasted calls.
    """
    new_clusters = []
    for cluster in plan.clusters:
        present_cols = [c for c in cluster.source_columns if c in df.columns]
        value_counts = _cluster_value_counts(df, present_cols)
        if value_counts is None or _mapping_covers(cluster.value_mapping, value_counts):
            new_clusters.append(cluster)
            continue
        value_plan = plan_value_mapping(cluster.canonical_name, value_counts, diagnosis_ctx, seed_hints, model=model)
        new_clusters.append(cluster.model_copy(update={"value_mapping": value_plan.value_mapping}))
    return plan.model_copy(update={"clusters": new_clusters})


def _cache_key(columns_summary: str, diagnosis_ctx: str, master_summary: str, model: str) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "columns_summary": columns_summary,
        "diagnosis_context": diagnosis_ctx,
        "master_summary": master_summary,
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cache_path(cache_key: str, cache_dir: Path | None = None) -> Path:
    base = cache_dir if cache_dir is not None else (config.DATA_DIR / "harmonized" / "_column_mapping_cache")
    return base / f"{cache_key}.json"


def get_column_mapping_plan(
    df: pd.DataFrame,
    columns: list[str],
    diagnosis_ctx: str,
    seed_hints: str = "",
    master_summary: str = "",
    model: str | None = None,
    cache_dir: Path | None = None,
) -> CrossCohortMappingPlan:
    """Cross-cohort column-matching plan, fully resolved (columns clustered
    *and* their values unified via fill_value_mappings) and cached by a hash
    of the exact column summary shown to the model -- re-running harmonize
    on the same cohort set costs nothing extra.
    """
    model = model or config.LLM_ESCALATION_MODEL
    columns_summary = build_column_summary(df, columns)
    cache_key = _cache_key(columns_summary, diagnosis_ctx, master_summary, model)
    cache_file = _cache_path(cache_key, cache_dir)

    if cache_file.exists():
        try:
            with open(cache_file, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("cache_key") == cache_key:
                return CrossCohortMappingPlan.model_validate(cached["result"])
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    plan = plan_column_clusters(df, columns, diagnosis_ctx, seed_hints, master_summary, model=model)
    plan = fill_value_mappings(df, plan, diagnosis_ctx, seed_hints)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"cache_key": cache_key, "result": plan.model_dump()}, f, indent=2)

    return plan


def apply_column_clusters(df: pd.DataFrame, plan: CrossCohortMappingPlan) -> pd.DataFrame:
    """Deterministically merge each cluster's source_columns into one
    canonical column (first non-null wins per row -- a sample belongs to one
    cohort, so at most one source column should be populated for it) and
    remap values through value_mapping. A canonical name colliding with an
    unrelated existing column is left unmerged rather than clobbered; values
    with no entry in value_mapping pass through raw. Both cases are recorded
    in df.attrs["harmonize_columns_notes"] rather than silently happening.
    """
    df = df.copy()
    notes: list[str] = [plan.notes] if plan.notes else []

    for cluster in plan.clusters:
        cols = [c for c in cluster.source_columns if c in df.columns]
        if not cols:
            continue

        target = cluster.canonical_name
        if target in df.columns and target not in cols:
            notes.append(
                f"{target}: canonical name already exists as an unrelated column; "
                f"left {cols} unmerged to avoid clobbering it"
            )
            continue

        conflict_rows = df[cols].notna().sum(axis=1) > 1
        if conflict_rows.any():
            notes.append(
                f"{target}: {int(conflict_rows.sum())} row(s) had values in more than one source column "
                f"({', '.join(cols)}); kept the first non-null"
            )

        coalesced = df[cols].bfill(axis=1).iloc[:, 0]

        if cluster.value_mapping:
            lowered_map = {str(k).strip().lower(): v for k, v in cluster.value_mapping.items()}
            unmapped: set[str] = set()

            def _map_value(value, _map=lowered_map, _unmapped=unmapped):
                if pd.isna(value):
                    return value
                key = str(value).strip().lower()
                if key in _map:
                    return _map[key]
                _unmapped.add(str(value))
                return value

            coalesced = coalesced.map(_map_value)
            if unmapped:
                notes.append(
                    f"{target}: value(s) not covered by value_mapping, left as-is: {', '.join(sorted(unmapped))}"
                )

        df = df.drop(columns=[c for c in cols if c != target])
        df[target] = coalesced

    if notes:
        df.attrs["harmonize_columns_notes"] = "; ".join(n for n in notes if n)

    return df
