import json

import pandas as pd
import pytest

from geotool import llm_annotate
from geotool.llm_schema import SampleGroupAnnotation, SeriesLevelAnnotation, SeriesLLMResult


def make_samples():
    return pd.DataFrame([
        {
            "gsm_id": "GSM1", "title": "siNC rep1", "source_name_ch1": "MDA-MB-231",
            "organism_ch1": "Homo sapiens", "molecule_ch1": "total RNA", "platform_id": "GPL1",
            "description": "", "library_selection": "cDNA", "library_strategy": "RNA-Seq",
            "data_row_count": "0", "rnaseq_library_type": "other",
            "cell_line": "MDA-MB-231", "treatment": "Negative-control siRNA",
        },
        {
            "gsm_id": "GSM2", "title": "siNC rep2", "source_name_ch1": "MDA-MB-231",
            "organism_ch1": "Homo sapiens", "molecule_ch1": "total RNA", "platform_id": "GPL1",
            "description": "", "library_selection": "cDNA", "library_strategy": "RNA-Seq",
            "data_row_count": "0", "rnaseq_library_type": "other",
            "cell_line": "MDA-MB-231", "treatment": "Negative-control siRNA",
        },
        {
            "gsm_id": "GSM3", "title": "siBYSL rep1", "source_name_ch1": "MDA-MB-231",
            "organism_ch1": "Mus musculus", "molecule_ch1": "total RNA", "platform_id": "GPL1",
            "description": "", "library_selection": "cDNA", "library_strategy": "RNA-Seq",
            "data_row_count": "0", "rnaseq_library_type": "other",
            "cell_line": "MDA-MB-231", "treatment": "siBYSL knockdown",
        },
    ])


def make_result(fp_ids):
    return SeriesLLMResult(
        series_level=SeriesLevelAnnotation(
            assay_type="bulk_rnaseq", has_outcome_data=False, treatment_context="siRNA knockdown study",
        ),
        sample_groups=[
            SampleGroupAnnotation(
                fingerprint_id=fp_ids[0], sample_source="cell_line", tissue_class="other",
                tissue_detail="breast cancer cell line", selection_method="none",
                diagnosis="DLBCL", diagnosis_source="sample_characteristics", prior_therapy="none",
            ),
            SampleGroupAnnotation(
                fingerprint_id=fp_ids[1], sample_source="cell_line", tissue_class="other",
                tissue_detail="breast cancer cell line", selection_method="none",
                diagnosis="Some Made Up Category", diagnosis_source="ambiguous", prior_therapy="unknown",
            ),
        ],
    )


def test_characteristic_columns_excludes_fixed_columns():
    samples = make_samples()
    cols = llm_annotate.characteristic_columns(samples)
    assert set(cols) == {"cell_line", "treatment"}


def make_samples_with_survival_column(n=244, n_survival_values=128):
    """Shaped like the live GSE183795 crash: a real categorical trait
    (tissue, 3 values) alongside a near-continuous numeric survival column
    -- before the fix, the latter alone fragmented 244 samples into 200
    fingerprint groups and overflowed the model's max_tokens."""
    rows = []
    for i in range(n):
        rows.append({
            "gsm_id": f"GSM{i}", "title": f"s{i}", "source_name_ch1": "pancreas",
            "organism_ch1": "Homo sapiens", "molecule_ch1": "total RNA", "platform_id": "GPL1",
            "description": "", "library_selection": "cDNA", "library_strategy": "RNA-Seq",
            "data_row_count": "0", "rnaseq_library_type": "other",
            "tissue": ["tumor", "normal", "metastasis"][i % 3],
            "survival months": round((i % n_survival_values) * 0.7, 1),
        })
    return pd.DataFrame(rows)


def test_characteristic_columns_excludes_high_cardinality_numeric_column():
    samples = make_samples_with_survival_column()
    cols = llm_annotate.characteristic_columns(samples)
    assert cols == ["tissue"]  # survival months excluded -- not a grouping-worthy trait


def test_group_fingerprints_does_not_fragment_on_survival_column():
    samples = make_samples_with_survival_column()
    _fp_ids, groups = llm_annotate.group_fingerprints(samples)
    assert len(groups) == 3  # one per tissue value, not one per (near-)unique sample


def make_samples_with_id_column(n=204, n_patients=129):
    """Shaped like the live GSE93326 crash: multiple samples per patient
    (3 compartments each) but a near-unique 'patient id' string column that,
    included in the fingerprint, fragments what should be ~3 groups into
    nearly one per sample."""
    rows = []
    compartments = ["Epithelium", "Stroma", "Bulk"]
    for i in range(n):
        rows.append({
            "gsm_id": f"GSM{i}", "title": f"s{i}", "source_name_ch1": "pancreas",
            "organism_ch1": "Homo sapiens", "molecule_ch1": "total RNA", "platform_id": "GPL1",
            "description": "", "library_selection": "cDNA", "library_strategy": "RNA-Seq",
            "data_row_count": "0", "rnaseq_library_type": "other",
            "patient id": f"CUMC_{i % n_patients:03d}",
            "compartment": compartments[i % 3],
        })
    return pd.DataFrame(rows)


def test_characteristic_columns_excludes_high_cardinality_identifier_column():
    samples = make_samples_with_id_column()
    cols = llm_annotate.characteristic_columns(samples)
    assert cols == ["compartment"]  # patient id excluded on cardinality alone -- not a characteristic


def test_group_fingerprints_does_not_fragment_on_id_column():
    samples = make_samples_with_id_column()
    _fp_ids, groups = llm_annotate.group_fingerprints(samples)
    assert len(groups) == 3  # one per compartment, not one per (patient, sample)


def test_low_cardinality_id_named_column_is_not_excluded():
    """Cardinality is the only gate -- a column that merely has 'id' in its
    name but few distinct values (e.g. a batch or run id shared by many
    samples) is real, useful grouping signal, not an identifier to drop."""
    samples = make_samples_with_id_column()
    samples["batch id"] = ["batch1" if i < 100 else "batch2" for i in range(len(samples))]
    assert "batch id" in llm_annotate.characteristic_columns(samples)


def test_characteristic_columns_excludes_high_cardinality_categorical_column():
    """Cardinality-only exclusion (no numeric or name-pattern requirement)
    also catches a genuinely diverse *categorical* column -- live example:
    GSE253260's "firsttreatment" (64 distinct free-text regimens, neither
    numeric nor identifier-named), which combined multiplicatively with
    several other modest-cardinality clinical columns to still leave 308
    fingerprint groups even after the numeric/identifier-only exclusions."""
    samples = make_samples_with_id_column()
    samples["treatment"] = [f"REGIMEN_{i % 20}" for i in range(len(samples))]  # 20 distinct free-text values
    cols = llm_annotate.characteristic_columns(samples)
    assert cols == ["compartment"]  # both patient id and treatment excluded on cardinality alone


def test_survival_like_numeric_columns_matches_survival_names_only():
    samples = make_samples_with_survival_column()
    samples["age"] = list(range(len(samples)))  # also high-cardinality numeric, but not survival-named
    assert llm_annotate.survival_like_numeric_columns(samples) == ["survival months"]
    # "age" is excluded from the fingerprint for the same cardinality reason, just not asked about as a time unit
    assert "age" not in llm_annotate.characteristic_columns(samples)


def test_apply_numeric_column_units_converts_to_days_and_skips_unknown():
    from geotool.llm_schema import NumericColumnUnit

    samples = pd.DataFrame({
        "gsm_id": ["GSM1", "GSM2", "GSM3"],
        "os_days": [10, 20, 30],
        "os_months": [1, 2, 3],
        "os_years": [1, 2, 3],
        "unclear_time": [5, 6, 7],
    })
    units = [
        NumericColumnUnit(column_name="os_days", unit="days"),
        NumericColumnUnit(column_name="os_months", unit="months"),
        NumericColumnUnit(column_name="os_years", unit="years"),
        NumericColumnUnit(column_name="unclear_time", unit="unknown"),
    ]

    out = llm_annotate.apply_numeric_column_units(samples, units)

    # llm_ prefix: harmonize.get_llm_annotation only keeps llm_-prefixed
    # columns when joining back onto a cohort's annotation.tsv
    assert out["llm_os_days_days"].tolist() == [10.0, 20.0, 30.0]
    assert out["llm_os_months_days"].tolist() == pytest.approx([30.4375, 60.875, 91.3125])
    assert out["llm_os_years_days"].tolist() == pytest.approx([365.25, 730.5, 1095.75])
    assert "llm_unclear_time_days" not in out.columns  # "unknown" unit -- left unconverted
    assert out["unclear_time"].tolist() == [5, 6, 7]  # raw column untouched either way


def test_group_fingerprints_groups_identical_characteristics():
    samples = make_samples()
    fp_ids, groups = llm_annotate.group_fingerprints(samples)
    # GSM1 and GSM2 share cell_line+treatment -> same fingerprint; GSM3 differs (treatment)
    assert fp_ids.iloc[0] == fp_ids.iloc[1]
    assert fp_ids.iloc[0] != fp_ids.iloc[2]
    assert len(groups) == 2
    fp0 = fp_ids.iloc[0]
    assert set(groups[fp0]["gsm_ids"]) == {"GSM1", "GSM2"}


def test_fingerprint_key_is_stable_and_order_independent():
    a = llm_annotate.fingerprint_key({"tissue": "liver", "sex": "F"})
    b = llm_annotate.fingerprint_key({"sex": "F", "tissue": "liver"})
    c = llm_annotate.fingerprint_key({"tissue": "kidney", "sex": "F"})
    assert a == b
    assert a != c


def test_merge_annotations_joins_by_fingerprint_and_normalizes_diagnosis():
    samples = make_samples()
    fp_ids, groups = llm_annotate.group_fingerprints(samples)
    fp_list = list(groups.keys())
    result = make_result(fp_list)

    merged = llm_annotate.merge_annotations(samples, fp_ids, result)

    # species is deterministic, not from the LLM
    assert merged.loc[merged["gsm_id"] == "GSM1", "llm_species"].iloc[0] == "human"
    assert merged.loc[merged["gsm_id"] == "GSM3", "llm_species"].iloc[0] == "mouse"

    # GSM1/GSM2 share a fingerprint and a recognized diagnosis
    assert merged.loc[merged["gsm_id"] == "GSM1", "llm_diagnosis"].iloc[0] == "DLBCL"
    assert merged.loc[merged["gsm_id"] == "GSM2", "llm_diagnosis"].iloc[0] == "DLBCL"

    # GSM3's group used a diagnosis not in the vocab -> folds to "other", raw value preserved
    gsm3_detail = merged.loc[merged["gsm_id"] == "GSM3", "llm_diagnosis_detail"].iloc[0]
    assert merged.loc[merged["gsm_id"] == "GSM3", "llm_diagnosis"].iloc[0] == "other"
    assert "Some Made Up Category" in gsm3_detail


class _FakeTextBlock:
    def __init__(self, parsed_output):
        self.type = "text"
        self.parsed_output = parsed_output


class _FakeResponse:
    def __init__(self, parsed_output):
        self.content = [_FakeTextBlock(parsed_output)]


class _FakeMessages:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._result)


class _FakeClient:
    def __init__(self, result):
        self.messages = _FakeMessages(result)


def test_annotate_and_cache_calls_model_once_then_reuses_cache(monkeypatch, tmp_path):
    samples = make_samples()
    _fp_ids, groups = llm_annotate.group_fingerprints(samples)
    fp_list = list(groups.keys())
    result = make_result(fp_list)

    fake_client = _FakeClient(result)
    monkeypatch.setattr(llm_annotate.anthropic, "Anthropic", lambda: fake_client)

    series_row = {"gse_id": "GSE_TEST", "title": "t", "summary": "s", "overall_design": "d"}

    class FakeGSE:
        gsms = {}

    merged1, series_level1 = llm_annotate.annotate_and_cache(FakeGSE(), series_row, samples, series_dir=tmp_path)
    assert len(fake_client.messages.calls) == 1
    assert series_level1.assay_type == "bulk_rnaseq"
    assert (tmp_path / "GSE_TEST" / "llm_annotations.json").exists()

    merged2, series_level2 = llm_annotate.annotate_and_cache(FakeGSE(), series_row, samples, series_dir=tmp_path)
    assert len(fake_client.messages.calls) == 1  # no second API call -- cache hit
    pd.testing.assert_frame_equal(merged1, merged2)


def test_annotate_and_cache_recomputes_when_characteristics_change(monkeypatch, tmp_path):
    samples = make_samples()
    _fp_ids, groups = llm_annotate.group_fingerprints(samples)
    result = make_result(list(groups.keys()))
    fake_client = _FakeClient(result)
    monkeypatch.setattr(llm_annotate.anthropic, "Anthropic", lambda: fake_client)

    series_row = {"gse_id": "GSE_TEST2", "title": "t", "summary": "s", "overall_design": "d"}

    class FakeGSE:
        gsms = {}

    llm_annotate.annotate_and_cache(FakeGSE(), series_row, samples, series_dir=tmp_path)
    assert len(fake_client.messages.calls) == 1

    changed_samples = samples.copy()
    changed_samples.loc[0, "treatment"] = "a completely different treatment"
    fp_ids2, groups2 = llm_annotate.group_fingerprints(changed_samples)
    result2 = make_result(list(groups2.keys()))
    fake_client.messages._result = result2

    llm_annotate.annotate_and_cache(FakeGSE(), series_row, changed_samples, series_dir=tmp_path)
    assert len(fake_client.messages.calls) == 2  # cache key changed -> re-called


def test_build_prompt_includes_excluded_numeric_columns_section():
    samples = make_samples_with_survival_column()
    _fp_ids, groups = llm_annotate.group_fingerprints(samples)
    numeric_columns = {
        col: llm_annotate.summarize_numeric_column(samples[col])
        for col in llm_annotate.survival_like_numeric_columns(samples)
    }

    _system_prompt, user_prompt = llm_annotate.build_prompt(None, {"gse_id": "GSE_X"}, groups, numeric_columns)

    assert "Excluded numeric column(s)" in user_prompt
    assert "survival months" in user_prompt
    assert "3 unique characteristics pattern(s)" in user_prompt  # grouped by tissue only, not fragmented


def test_annotate_and_cache_estimates_and_applies_survival_column_units(monkeypatch, tmp_path):
    from geotool.llm_schema import NumericColumnUnit

    samples = make_samples_with_survival_column()
    fp_ids, groups = llm_annotate.group_fingerprints(samples)
    assert len(groups) == 3  # small, thanks to the fingerprint fix -- the real point of this test

    result = SeriesLLMResult(
        series_level=SeriesLevelAnnotation(
            assay_type="bulk_rnaseq", has_outcome_data=True,
            outcome_columns=["survival months"],
            numeric_column_units=[NumericColumnUnit(column_name="survival months", unit="months")],
        ),
        sample_groups=[
            SampleGroupAnnotation(
                fingerprint_id=fp_id, sample_source="biopsy", tissue_class="tissue",
                tissue_detail="pancreas", selection_method="none",
                diagnosis="PDAC", diagnosis_source="sample_characteristics", prior_therapy="unknown",
            )
            for fp_id in groups
        ],
    )
    fake_client = _FakeClient(result)
    monkeypatch.setattr(llm_annotate.anthropic, "Anthropic", lambda: fake_client)

    series_row = {"gse_id": "GSE_SURV", "title": "t", "summary": "s", "overall_design": "d"}

    class FakeGSE:
        gsms = {}

    merged, series_level = llm_annotate.annotate_and_cache(FakeGSE(), series_row, samples, series_dir=tmp_path)

    assert len(fake_client.messages.calls) == 1
    assert series_level.numeric_column_units[0].unit == "months"
    assert "llm_survival_months_days" in merged.columns
    expected = (samples["survival months"] * 30.4375).tolist()
    assert merged["llm_survival_months_days"].tolist() == pytest.approx(expected)
