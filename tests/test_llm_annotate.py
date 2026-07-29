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
