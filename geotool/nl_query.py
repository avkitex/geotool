"""Natural-language cohort query.

Flow:
1. NL query -> QueryFilters (diagnosis + synonyms + filter categories), one Claude call.
2. Broad GEO/Entrez recall using the diagnosis OR each synonym (search.run_nl_query).
3. One lightweight Claude call per candidate series, classifying its title/summary
   (no full-record fetch) against the filters.
4. Every candidate is reported with its classification as separate columns --
   nothing is silently dropped; the human filters/sorts the table themselves.
"""
from __future__ import annotations

from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from geotool import config

SAMPLE_TYPE_VALUES = ("biopsy", "cell_line", "any")
TISSUE_CLASS_VALUES = ("tissue", "blood", "other", "any")
ASSAY_TYPE_VALUES = ("bulk_rnaseq", "microarray", "scrnaseq", "other", "any")
SELECTION_METHOD_VALUES = ("LCM", "FACS_sorting", "MACS_sorting", "none", "any")


class QueryFilters(BaseModel):
    diagnosis: str = Field(description="Primary/canonical diagnosis term from the query")
    diagnosis_synonyms: list[str] = Field(
        default_factory=list,
        description="Alternate names, abbreviations, and closely related terms for the diagnosis, "
        "used to broaden the GEO keyword search",
    )
    species: Literal["human", "mouse", "other", "any"] = "any"
    sample_type: Literal["biopsy", "cell_line", "any"] = "any"
    tissue_class: Literal["tissue", "blood", "other", "any"] = "any"
    assay_type: Literal["bulk_rnaseq", "microarray", "scrnaseq", "other", "any"] = "any"
    selection_method: Literal["LCM", "FACS_sorting", "MACS_sorting", "none", "any"] = "any"
    min_samples: int | None = Field(default=None, description="Minimum series sample size, if the query states one")


class SeriesClassification(BaseModel):
    matches_diagnosis: bool = Field(
        description="Does the summary describe the requested diagnosis (or a named synonym/subtype of it)?"
    )
    diagnosis_detail: str = Field(default="", description="Specific subdiagnosis/subtype mentioned, if any")
    species: Literal["human", "mouse", "other", "unknown"]
    sample_type: Literal["biopsy", "cell_line", "mixed", "unknown"]
    tissue_class: Literal["tissue", "blood", "other", "unknown"]
    assay_type: Literal["bulk_rnaseq", "microarray", "scrnaseq", "other", "unknown"]
    selection_method: Literal["none", "LCM", "FACS_sorting", "MACS_sorting", "other", "unknown"]
    notes: str = Field(default="", description="Anything notable or uncertain about this classification")


QUERY_FILTERS_SYSTEM_PROMPT = """Turn the user's natural-language GEO cohort search request into a
structured filter spec.

`diagnosis` is the primary disease/condition being searched for. `diagnosis_synonyms` should list
alternate names, abbreviations, and closely related terms for that diagnosis (for example, for
"pancreatic cancer": "pancreatic ductal adenocarcinoma", "PDAC", "pancreatic adenocarcinoma",
"pancreatic carcinoma") so a keyword search can find series that use different terminology.

Only set the other fields (species, sample_type, tissue_class, assay_type, selection_method,
min_samples) when the query actually specifies them; leave them at "any" (or null for min_samples)
otherwise."""

CLASSIFY_SYSTEM_PROMPT = """You are screening a GEO series against a cohort search request, using
only its title, summary, and platform name(s) -- you do not have per-sample data.

GEO summaries are user-submitted and can be incomplete or imprecise. Classify each field as best
you can from the text; use "unknown" rather than guessing when the text doesn't say.

The requested diagnosis is: {diagnosis}
Accepted synonyms/subtypes: {synonyms}

Set matches_diagnosis to true only if the summary indicates the series is actually about this
diagnosis (or a named subtype/synonym of it) -- not merely tangentially related, and not a
different disease that happens to share a keyword."""


def _extract_parsed(response, output_type):
    for block in response.content:
        if getattr(block, "type", None) == "text" and getattr(block, "parsed_output", None) is not None:
            return block.parsed_output
    raise ValueError(f"Claude response contained no parsed {output_type.__name__}")


def parse_query_filters(text: str, model: str | None = None) -> QueryFilters:
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=model or config.LLM_MODEL,
        max_tokens=1024,
        system=QUERY_FILTERS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
        output_format=QueryFilters,
    )
    return _extract_parsed(response, QueryFilters)


def build_summary_prompt(candidate: dict, platform_titles: list[str]) -> str:
    lines = [
        f"Series accession: {candidate.get('gse_id', '')}",
        f"Title: {candidate.get('title', '')}",
        f"Organism: {candidate.get('organism', '')}",
        f"Summary: {candidate.get('summary', '')}",
        f"Number of samples: {candidate.get('n_samples', '')}",
    ]
    if platform_titles:
        lines.append(f"Platform(s): {'; '.join(platform_titles)}")
    return "\n".join(lines)


def classify_series(
    candidate: dict,
    filters: QueryFilters,
    platform_titles: list[str] | None = None,
    model: str | None = None,
) -> SeriesClassification:
    system_prompt = CLASSIFY_SYSTEM_PROMPT.format(
        diagnosis=filters.diagnosis,
        synonyms=", ".join(filters.diagnosis_synonyms) or "(none given)",
    )
    user_prompt = build_summary_prompt(candidate, platform_titles or [])
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=model or config.LLM_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        output_format=SeriesClassification,
    )
    return _extract_parsed(response, SeriesClassification)


def _is_ambiguous(result: SeriesClassification) -> bool:
    fields = [result.species, result.sample_type, result.tissue_class, result.assay_type, result.selection_method]
    return fields.count("unknown") >= 2


def classify_series_with_escalation(
    candidate: dict,
    filters: QueryFilters,
    platform_titles: list[str] | None = None,
    escalate: bool = False,
) -> SeriesClassification:
    """classify_series(), optionally re-run on the escalation model when the first
    pass came back mostly "unknown" (i.e. the cheap model couldn't tell from the
    summary alone).
    """
    result = classify_series(candidate, filters, platform_titles)
    if escalate and _is_ambiguous(result):
        result = classify_series(candidate, filters, platform_titles, model=config.LLM_ESCALATION_MODEL)
    return result
