"""Pydantic schema for the per-series LLM annotation call (client.messages.parse).

`diagnosis` is a plain str, not a Literal -- the allowed category list lives
in vocab_data/diagnoses.json (see geotool.vocab) and can grow without a code
change. Every other categorical field here is small and stable, so it stays
a real Literal.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SampleGroupAnnotation(BaseModel):
    fingerprint_id: str = Field(description="Matches the fingerprint id given in the prompt for this group")
    sample_source: Literal["biopsy", "cell_line", "xenograft", "primary_culture", "other", "unknown"]
    tissue_class: Literal["tissue", "blood", "bone_marrow", "other", "unknown"]
    tissue_detail: str = Field(description="Free-text specifics, e.g. 'lymph node biopsy'")
    selection_method: Literal["none", "LCM", "FACS_sorting", "MACS_sorting", "other", "unknown"]
    selection_detail: str = Field(default="", description="Free-text specifics if selection_method is other")
    diagnosis: str = Field(description="High-level diagnosis category; pick from the provided list, or 'other'/'unknown'")
    diagnosis_detail: str = Field(default="", description="Free-text subtype/specifics")
    diagnosis_source: Literal["sample_characteristics", "series_summary_inferred", "ambiguous"]
    prior_therapy: Literal["none", "yes", "unknown"]
    prior_therapy_detail: str = Field(default="", description="Regimen/timing specifics if prior_therapy is yes")


class NumericColumnUnit(BaseModel):
    column_name: str = Field(description="samples.tsv column name for a numeric survival/follow-up time column")
    unit: Literal["days", "months", "years", "unknown"] = Field(
        description="Best estimate of this column's time unit, from its name and the value range given -- "
        "GEO survival columns are most often already in days, months less often, years rarely; use 'unknown' "
        "rather than guessing when genuinely ambiguous"
    )


class SeriesLevelAnnotation(BaseModel):
    assay_type: Literal["bulk_rnaseq", "microarray", "scrnaseq", "other", "unknown"]
    assay_detail: str = Field(default="", description="e.g. library prep or single-cell platform specifics")
    treatment_context: str = Field(default="", description="e.g. 'front-line R-CHOP trial, OS/PFS tracked'")
    has_outcome_data: bool
    outcome_columns: list[str] = Field(default_factory=list, description="samples.tsv column names holding OS/PFS/response, if any")
    numeric_column_units: list[NumericColumnUnit] = Field(
        default_factory=list,
        description="Unit estimate for each numeric survival/follow-up time column listed in the prompt's "
        "'excluded numeric columns' section, if any were given",
    )
    inconsistency_notes: str = Field(default="", description="Contradictions between the summary and per-sample characteristics, or ''")


class SeriesLLMResult(BaseModel):
    series_level: SeriesLevelAnnotation
    sample_groups: list[SampleGroupAnnotation]
