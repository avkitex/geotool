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


# --- harmonize_cohorts cross-cohort column matching --------------------------

class _FakeTextBlock:
    def __init__(self, parsed_output):
        self.type = "text"
        self.parsed_output = parsed_output


class _FakeResponse:
    def __init__(self, parsed_output):
        self.content = [_FakeTextBlock(parsed_output)]


class _FakeMessages:
    def __init__(self, plan):
        self.plan = plan
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self.plan)


class _FakeClient:
    def __init__(self, plan):
        self.messages = _FakeMessages(plan)


def test_harmonize_cohorts_merges_matching_columns_across_cohorts(tmp_path, monkeypatch):
    from geotool import harmonize_columns
    from geotool.harmonize_columns import ColumnCluster, CrossCohortMappingPlan

    annotation_a = pd.DataFrame({
        "gsm_id": ["GSM1"], "gse_id": ["GSE_A"], "coo_hans": ["GCB"],
    })
    annotation_b = pd.DataFrame({
        "gsm_id": ["GSM2"], "gse_id": ["GSE_B"], "cell_of_origin": ["Germinal center B-cell-like"],
    })
    _write_cohort(tmp_path, "GSE_A", annotation_a)
    _write_cohort(tmp_path, "GSE_B", annotation_b)

    plan = CrossCohortMappingPlan(clusters=[
        ColumnCluster(
            canonical_name="coo",
            source_columns=["coo_hans", "cell_of_origin"],
            value_mapping={"GCB": "GCB", "Germinal center B-cell-like": "GCB"},
        )
    ])
    fake_client = _FakeClient(plan)
    monkeypatch.setattr(harmonize_columns.anthropic, "Anthropic", lambda: fake_client)
    monkeypatch.setattr(harmonize.config, "DATA_DIR", tmp_path)

    master = harmonize.harmonize_cohorts(["GSE_A", "GSE_B"], series_dir=tmp_path)

    assert len(fake_client.messages.calls) == 1
    assert "coo" in master.columns
    assert "coo_hans" not in master.columns
    assert "cell_of_origin" not in master.columns
    assert set(master["coo"]) == {"GCB"}


def test_harmonize_cohorts_skips_matching_when_only_protected_columns_present(tmp_path, monkeypatch):
    """Regression guard: a batch with only already-canonical columns (survival
    pairs, identity) must never trigger a live LLM call -- there's nothing to
    cluster, so the matching step should short-circuit before calling out."""
    from geotool import harmonize_columns

    annotation = pd.DataFrame({"gsm_id": ["GSM1"], "gse_id": ["GSE_A"], "OS_time": [1.0], "OS_event": [1]})
    _write_cohort(tmp_path, "GSE_A", annotation)

    def _boom(*args, **kwargs):
        raise AssertionError("should not call the LLM when there are no clusterable columns")

    monkeypatch.setattr(harmonize_columns, "get_column_mapping_plan", _boom)

    result = harmonize.harmonize_cohorts(["GSE_A"], series_dir=tmp_path)
    assert set(result.columns) >= {"gsm_id", "gse_id", "OS_time", "OS_event"}


def test_harmonize_cohorts_match_columns_false_skips_llm_call(tmp_path, monkeypatch):
    from geotool import harmonize_columns

    annotation = pd.DataFrame({"gsm_id": ["GSM1"], "gse_id": ["GSE_A"], "weird_col": ["x"]})
    _write_cohort(tmp_path, "GSE_A", annotation)

    def _boom(*args, **kwargs):
        raise AssertionError("should not call the LLM when match_columns=False")

    monkeypatch.setattr(harmonize_columns, "get_column_mapping_plan", _boom)

    result = harmonize.harmonize_cohorts(["GSE_A"], series_dir=tmp_path, match_columns=False)
    assert "weird_col" in result.columns


# --- harmonize_cohorts master mode -------------------------------------------

def test_harmonize_cohorts_master_path_skips_cohorts_already_in_master(tmp_path, capsys):
    master_path = tmp_path / "master.tsv"
    pd.DataFrame({"gsm_id": ["GSM_OLD"], "gse_id": ["GSE_A"], "coo": ["GCB"]}).to_csv(
        master_path, sep="\t", index=False
    )
    annotation_a = pd.DataFrame({"gsm_id": ["GSM1"], "gse_id": ["GSE_A"], "coo": ["ABC"]})
    _write_cohort(tmp_path, "GSE_A", annotation_a)

    result = harmonize.harmonize_cohorts(["GSE_A"], series_dir=tmp_path, master_path=master_path)

    assert len(result) == 1
    assert set(result["gsm_id"]) == {"GSM_OLD"}
    captured = capsys.readouterr()
    assert "already in master" in captured.out


def test_harmonize_cohorts_master_path_concatenates_new_cohorts(tmp_path):
    master_path = tmp_path / "master.tsv"
    pd.DataFrame({"gsm_id": ["GSM_OLD"], "gse_id": ["GSE_A"], "treatment": ["chemo"]}).to_csv(
        master_path, sep="\t", index=False
    )
    annotation_b = pd.DataFrame({"gsm_id": ["GSM2"], "gse_id": ["GSE_B"], "treatment": ["radio"]})
    _write_cohort(tmp_path, "GSE_B", annotation_b)

    result = harmonize.harmonize_cohorts(["GSE_B"], series_dir=tmp_path, master_path=master_path)

    assert set(result["gsm_id"]) == {"GSM_OLD", "GSM2"}
    assert set(result["gse_id"]) == {"GSE_A", "GSE_B"}


def test_harmonize_cohorts_master_path_nonexistent_file_is_ignored(tmp_path):
    annotation_a = pd.DataFrame({"gsm_id": ["GSM1"], "gse_id": ["GSE_A"], "treatment": ["chemo"]})
    _write_cohort(tmp_path, "GSE_A", annotation_a)

    result = harmonize.harmonize_cohorts(
        ["GSE_A"], series_dir=tmp_path, master_path=tmp_path / "does_not_exist.tsv",
    )
    assert set(result["gsm_id"]) == {"GSM1"}


def test_harmonize_cohorts_returns_master_unchanged_when_all_cohorts_already_in_it(tmp_path):
    master_path = tmp_path / "master.tsv"
    pd.DataFrame({"gsm_id": ["GSM_OLD"], "gse_id": ["GSE_A"], "treatment": ["chemo"]}).to_csv(
        master_path, sep="\t", index=False
    )

    result = harmonize.harmonize_cohorts(["GSE_A"], series_dir=tmp_path, master_path=master_path)
    assert set(result["gsm_id"]) == {"GSM_OLD"}


# --- harmonize_and_report ------------------------------------------------------

def test_harmonize_and_report_writes_both_tables_together(tmp_path):
    series_dir = tmp_path / "series"
    annotation = pd.DataFrame({"gsm_id": ["GSM1"], "gse_id": ["GSE_A"], "expression_status": ["ok"]})
    _write_cohort(series_dir, "GSE_A", annotation)

    out_dir = tmp_path / "harmonized" / "myproject"
    sample_df, cohort_df = harmonize.harmonize_and_report(["GSE_A"], out_dir, series_dir=series_dir, match_columns=False)

    assert (out_dir / "annotation.tsv").exists()
    assert (out_dir / "cohort_annotations.tsv").exists()
    assert set(sample_df["gsm_id"]) == {"GSM1"}
    assert set(cohort_df["gse_id"]) == {"GSE_A"}
    assert cohort_df.iloc[0]["readiness"] == "ready"


def test_harmonize_and_report_still_writes_cohort_table_when_nothing_downloaded(tmp_path):
    out_dir = tmp_path / "harmonized" / "myproject"
    sample_df, cohort_df = harmonize.harmonize_and_report(["GSE_MISSING"], out_dir, series_dir=tmp_path)

    assert sample_df.empty
    assert not (out_dir / "annotation.tsv").exists()
    assert (out_dir / "cohort_annotations.tsv").exists()
    assert cohort_df.iloc[0]["downloaded"] == False


def test_harmonize_and_report_respects_collection_root_for_readiness(tmp_path):
    series_dir = tmp_path / "series"
    annotation = pd.DataFrame({"gsm_id": ["GSM1"], "gse_id": ["GSE_A"], "expression_status": ["ok"]})
    _write_cohort(series_dir, "GSE_A", annotation)

    collection_root = tmp_path / "collection"
    (collection_root / "GSE_A").mkdir(parents=True)
    # Deliberately no expression_final.tsv.gz -- readiness should reflect
    # collection_root, not expression_status, once collection_root is given.

    out_dir = tmp_path / "harmonized" / "myproject"
    _sample_df, cohort_df = harmonize.harmonize_and_report(
        ["GSE_A"], out_dir, series_dir=series_dir, collection_root=collection_root, match_columns=False,
    )
    assert cohort_df.iloc[0]["readiness"] == "not_ready"
