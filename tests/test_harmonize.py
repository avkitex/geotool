import json

import pandas as pd
import pytest

from geotool import harmonize, llm_annotate
from geotool.llm_schema import SampleGroupAnnotation, SeriesLevelAnnotation, SeriesLLMResult


# --- apply_column_aliases ---------------------------------------------------

def test_apply_column_aliases_renames_known_variants():
    df = pd.DataFrame({"gsm_id": ["GSM1"], "Sex": ["F"], "organ": ["liver"]})
    result = harmonize.apply_column_aliases(df, aliases={"Sex": "sex", "organ": "tissue"})
    assert set(result.columns) == {"gsm_id", "sex", "tissue"}
    assert result.iloc[0]["sex"] == "F"


def test_apply_column_aliases_case_insensitive():
    df = pd.DataFrame({"SEX": ["M"]})
    result = harmonize.apply_column_aliases(df, aliases={"sex": "sex"})
    assert "sex" in result.columns


def test_apply_column_aliases_leaves_unknown_columns_untouched():
    df = pd.DataFrame({"some_weird_column": ["x"]})
    result = harmonize.apply_column_aliases(df, aliases={"Sex": "sex"})
    assert list(result.columns) == ["some_weird_column"]


def test_apply_column_aliases_does_not_clobber_existing_canonical_column():
    """If both a raw variant and the canonical name already exist, renaming
    would silently overwrite real data -- must leave the raw column alone."""
    df = pd.DataFrame({"sex": ["F"], "Sex": ["M"]})
    result = harmonize.apply_column_aliases(df, aliases={"Sex": "sex"})
    assert set(result.columns) == {"sex", "Sex"}


def test_load_annotation_aliases_reads_the_real_seed_file():
    aliases = harmonize.load_annotation_aliases()
    assert aliases["Sex"] == "sex"
    assert aliases["organ"] == "tissue"


# --- get_llm_annotation ------------------------------------------------------

def make_samples():
    return pd.DataFrame([
        {
            "gsm_id": "GSM1", "gse_id": "GSE_X", "title": "s1", "source_name_ch1": "biopsy",
            "organism_ch1": "Homo sapiens", "molecule_ch1": "total RNA", "platform_id": "GPL1",
            "description": "", "library_selection": "", "library_strategy": "",
            "data_row_count": "0", "rnaseq_library_type": "",
            "tissue": "lymph node",
        },
        {
            "gsm_id": "GSM2", "gse_id": "GSE_X", "title": "s2", "source_name_ch1": "biopsy",
            "organism_ch1": "Homo sapiens", "molecule_ch1": "total RNA", "platform_id": "GPL1",
            "description": "", "library_selection": "", "library_strategy": "",
            "data_row_count": "0", "rnaseq_library_type": "",
            "tissue": "lymph node",
        },
    ])


def make_llm_result(fp_id):
    return SeriesLLMResult(
        series_level=SeriesLevelAnnotation(assay_type="microarray", has_outcome_data=False),
        sample_groups=[
            SampleGroupAnnotation(
                fingerprint_id=fp_id, sample_source="biopsy", tissue_class="tissue",
                tissue_detail="lymph node biopsy", selection_method="none",
                diagnosis="DLBCL", diagnosis_source="sample_characteristics", prior_therapy="none",
            ),
        ],
    )


def test_get_llm_annotation_returns_none_without_cache_and_no_backfill(tmp_path):
    samples = make_samples()
    series_row = {"gse_id": "GSE_X"}
    result = harmonize.get_llm_annotation("GSE_X", series_row, samples, series_dir=tmp_path, backfill=False)
    assert result is None


def test_get_llm_annotation_reuses_existing_cache_without_backfill(tmp_path, monkeypatch):
    samples = make_samples()
    fp_ids, groups = llm_annotate.group_fingerprints(samples)
    fp_id = list(groups.keys())[0]
    llm_result = make_llm_result(fp_id)

    class _FakeTextBlock:
        def __init__(self, parsed_output):
            self.type = "text"
            self.parsed_output = parsed_output

    class _FakeResponse:
        def __init__(self, parsed_output):
            self.content = [_FakeTextBlock(parsed_output)]

    class _FakeMessages:
        calls = []

        def parse(self, **kwargs):
            _FakeMessages.calls.append(kwargs)
            return _FakeResponse(llm_result)

    class _FakeClient:
        messages = _FakeMessages()

    monkeypatch.setattr(llm_annotate.anthropic, "Anthropic", lambda: _FakeClient())

    series_row = {"gse_id": "GSE_X", "title": "t", "summary": "s", "overall_design": "d"}
    # Prime the cache exactly like `search --llm-annotate` would.
    llm_annotate.annotate_and_cache(None, series_row, samples, series_dir=tmp_path)
    assert len(_FakeMessages.calls) == 1

    result = harmonize.get_llm_annotation("GSE_X", series_row, samples, series_dir=tmp_path, backfill=False)

    assert len(_FakeMessages.calls) == 1  # reused cache, no second LLM call
    assert result is not None
    assert set(result["gsm_id"]) == {"GSM1", "GSM2"}
    assert (result[result["gsm_id"] == "GSM1"]["llm_diagnosis"] == "DLBCL").all()


def test_get_llm_annotation_backfills_when_requested(tmp_path, monkeypatch):
    samples = make_samples()
    fp_ids, groups = llm_annotate.group_fingerprints(samples)
    llm_result = make_llm_result(list(groups.keys())[0])

    class _FakeTextBlock:
        def __init__(self, parsed_output):
            self.type = "text"
            self.parsed_output = parsed_output

    class _FakeResponse:
        def __init__(self, parsed_output):
            self.content = [_FakeTextBlock(parsed_output)]

    class _FakeMessages:
        def parse(self, **kwargs):
            return _FakeResponse(llm_result)

    class _FakeClient:
        messages = _FakeMessages()

    monkeypatch.setattr(llm_annotate.anthropic, "Anthropic", lambda: _FakeClient())

    series_row = {"gse_id": "GSE_X", "title": "t", "summary": "s", "overall_design": "d"}
    result = harmonize.get_llm_annotation("GSE_X", series_row, samples, series_dir=tmp_path, backfill=True)

    assert result is not None
    assert (tmp_path / "GSE_X" / "llm_annotations.json").exists()  # written for future free reuse
    assert set(result["gsm_id"]) == {"GSM1", "GSM2"}


# --- harmonize_cohort / harmonize_cohorts -----------------------------------

def _write_cohort(series_dir, gse_id, annotation_df, series_row=None, samples_df=None):
    out_dir = series_dir / gse_id
    out_dir.mkdir(parents=True, exist_ok=True)
    annotation_df.to_csv(out_dir / "annotation.tsv", sep="\t", index=False)
    if series_row is not None:
        pd.DataFrame([series_row]).to_csv(out_dir / "series.tsv", sep="\t", index=False)
    if samples_df is not None:
        samples_df.to_csv(out_dir / "samples.tsv", sep="\t", index=False)
    return out_dir


def test_harmonize_cohort_returns_none_when_not_downloaded(tmp_path):
    assert harmonize.harmonize_cohort("GSE_MISSING", series_dir=tmp_path) is None


def test_harmonize_cohort_applies_aliases_without_llm_cache(tmp_path):
    annotation = pd.DataFrame({"gsm_id": ["GSM1"], "gse_id": ["GSE_X"], "Sex": ["F"]})
    _write_cohort(tmp_path, "GSE_X", annotation)

    result = harmonize.harmonize_cohort("GSE_X", series_dir=tmp_path)

    assert result is not None
    assert "sex" in result.columns
    assert not any(c.startswith("llm_") for c in result.columns)


def test_harmonize_cohort_enforces_correct_gse_id_even_if_file_lacks_it(tmp_path):
    """Regression: a real older cohort's annotation.tsv had no gse_id column
    at all, which surfaced as NaN in the master table despite gse_id being
    exactly what harmonize_cohorts concatenates/keys on."""
    annotation = pd.DataFrame({"gsm_id": ["GSM1"], "treatment": ["a"]})  # no gse_id column
    _write_cohort(tmp_path, "GSE_NO_ID", annotation)

    result = harmonize.harmonize_cohort("GSE_NO_ID", series_dir=tmp_path)

    assert result is not None
    assert (result["gse_id"] == "GSE_NO_ID").all()


def test_harmonize_cohort_overrides_wrong_gse_id_in_file(tmp_path):
    annotation = pd.DataFrame({"gsm_id": ["GSM1"], "gse_id": ["WRONG"]})
    _write_cohort(tmp_path, "GSE_RIGHT", annotation)

    result = harmonize.harmonize_cohort("GSE_RIGHT", series_dir=tmp_path)

    assert (result["gse_id"] == "GSE_RIGHT").all()


def test_harmonize_cohort_merges_llm_columns_from_cache(tmp_path, monkeypatch):
    annotation = pd.DataFrame({"gsm_id": ["GSM1", "GSM2"], "gse_id": ["GSE_X", "GSE_X"], "treatment": ["a", "a"]})
    samples = make_samples()
    series_row = {"gse_id": "GSE_X", "title": "t", "summary": "s", "overall_design": "d"}
    _write_cohort(tmp_path, "GSE_X", annotation, series_row=series_row, samples_df=samples)

    fp_ids, groups = llm_annotate.group_fingerprints(samples)
    llm_result = make_llm_result(list(groups.keys())[0])

    class _FakeTextBlock:
        def __init__(self, parsed_output):
            self.type = "text"
            self.parsed_output = parsed_output

    class _FakeResponse:
        def __init__(self, parsed_output):
            self.content = [_FakeTextBlock(parsed_output)]

    class _FakeMessages:
        def parse(self, **kwargs):
            return _FakeResponse(llm_result)

    class _FakeClient:
        messages = _FakeMessages()

    monkeypatch.setattr(llm_annotate.anthropic, "Anthropic", lambda: _FakeClient())
    llm_annotate.annotate_and_cache(None, series_row, samples, series_dir=tmp_path)

    result = harmonize.harmonize_cohort("GSE_X", series_dir=tmp_path)

    assert "llm_diagnosis" in result.columns
    assert (result["llm_diagnosis"] == "DLBCL").all()


def test_harmonize_cohorts_skips_missing_and_concatenates_present(tmp_path, capsys):
    annotation_a = pd.DataFrame({"gsm_id": ["GSM1"], "gse_id": ["GSE_A"], "OS_time": [12.0], "OS_event": [1]})
    annotation_b = pd.DataFrame({"gsm_id": ["GSM2"], "gse_id": ["GSE_B"], "PFS_time": [6.0], "PFS_event": [0]})
    _write_cohort(tmp_path, "GSE_A", annotation_a)
    _write_cohort(tmp_path, "GSE_B", annotation_b)

    master = harmonize.harmonize_cohorts(["GSE_A", "GSE_B", "GSE_MISSING"], series_dir=tmp_path)

    assert len(master) == 2
    assert set(master["gse_id"]) == {"GSE_A", "GSE_B"}
    # survival columns unioned across cohorts, NaN where not reported
    assert set(master.columns) >= {"gsm_id", "gse_id", "OS_time", "OS_event", "PFS_time", "PFS_event"}
    gse_a_row = master[master["gse_id"] == "GSE_A"].iloc[0]
    assert pd.isna(gse_a_row["PFS_time"])

    captured = capsys.readouterr()
    assert "GSE_MISSING" in captured.out
    assert "skipped" in captured.out


def test_harmonize_cohorts_returns_empty_dataframe_when_nothing_downloaded(tmp_path):
    result = harmonize.harmonize_cohorts(["GSE_MISSING"], series_dir=tmp_path)
    assert result.empty
