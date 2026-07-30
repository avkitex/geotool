import pandas as pd
import pytest

from geotool import harmonize_columns
from geotool.harmonize_columns import (
    ColumnCluster,
    CrossCohortMappingPlan,
    ValueMappingEntry,
    ValueUnificationPlan,
)


# --- clusterable_columns -----------------------------------------------------

def test_clusterable_columns_excludes_protected():
    df = pd.DataFrame({"gsm_id": ["G1"], "gse_id": ["GSE1"], "treatment": ["a"], "coo_hans": ["GCB"]})
    result = harmonize_columns.clusterable_columns(df, protected={"gsm_id", "gse_id", "treatment"})
    assert result == ["coo_hans"]


# --- diagnosis_context --------------------------------------------------------

def test_diagnosis_context_reads_llm_diagnosis_and_detail():
    df = pd.DataFrame({
        "llm_diagnosis": ["DLBCL", "DLBCL", "unknown"],
        "llm_diagnosis_detail": ["GCB subtype", "", ""],
    })
    ctx = harmonize_columns.diagnosis_context(df)
    assert "DLBCL" in ctx
    assert "GCB subtype" in ctx
    assert ctx != "unknown"


def test_diagnosis_context_defaults_to_unknown():
    df = pd.DataFrame({"some_col": ["x"]})
    assert harmonize_columns.diagnosis_context(df) == "unknown"


def test_diagnosis_context_all_unknown_values_stays_unknown():
    df = pd.DataFrame({"llm_diagnosis": ["unknown", "unknown"]})
    assert harmonize_columns.diagnosis_context(df) == "unknown"


# --- seed_hint_text ------------------------------------------------------------

def test_seed_hint_text_matches_diagnosis_case_insensitively():
    concepts = {"DLBCL": {"COO": ["GCB", "ABC", "Unclassified"]}}
    hint = harmonize_columns.seed_hint_text("dlbcl, GCB subtype", concepts=concepts)
    assert "COO" in hint
    assert "GCB" in hint


def test_seed_hint_text_matches_underscore_normalized_diagnosis():
    concepts = {"CLL_SLL": {"IGHV_status": ["mutated", "unmutated"]}}
    hint = harmonize_columns.seed_hint_text("patient has CLL SLL, indolent", concepts=concepts)
    assert "IGHV_status" in hint


def test_seed_hint_text_empty_when_no_match():
    concepts = {"DLBCL": {"COO": ["GCB", "ABC"]}}
    assert harmonize_columns.seed_hint_text("multiple_myeloma", concepts=concepts) == ""


def test_seed_hint_text_empty_when_diagnosis_unknown():
    concepts = {"DLBCL": {"COO": ["GCB", "ABC"]}}
    assert harmonize_columns.seed_hint_text("unknown", concepts=concepts) == ""


def test_load_clinical_concepts_reads_the_real_seed_file():
    concepts = harmonize_columns.load_clinical_concepts()
    assert "COO" in concepts["DLBCL"]
    assert "GCB" in concepts["DLBCL"]["COO"]


# --- build_column_summary / master_columns_summary ---------------------------

def test_build_column_summary_includes_cohorts_and_value_counts():
    df = pd.DataFrame({
        "gse_id": ["GSE_A", "GSE_A", "GSE_B"],
        "coo_hans": ["GCB", "GCB", None],
        "coo_other": [None, None, "ABC"],
    })
    summary = harmonize_columns.build_column_summary(df, ["coo_hans", "coo_other"])
    assert "coo_hans" in summary
    assert "GSE_A" in summary
    assert "'GCB' (n=2)" in summary
    assert "GSE_B" in summary


def test_master_columns_summary_none_master_returns_empty():
    assert harmonize_columns.master_columns_summary(None, protected=set()) == ""


def test_master_columns_summary_excludes_protected_columns():
    master_df = pd.DataFrame({"gsm_id": ["G1"], "gse_id": ["GSE1"], "coo": ["GCB"]})
    summary = harmonize_columns.master_columns_summary(master_df, protected={"gsm_id", "gse_id"})
    assert "coo" in summary
    assert "gsm_id" not in summary


# --- apply_column_clusters ----------------------------------------------------

def test_apply_column_clusters_merges_and_unifies_values():
    df = pd.DataFrame({
        "gsm_id": ["G1", "G2", "G3"],
        "gse_id": ["GSE_A", "GSE_A", "GSE_B"],
        "coo_hans": ["GCB", "Germinal center B-cell-like", None],
        "coo_other": [None, None, "ABC"],
    })
    plan = CrossCohortMappingPlan(clusters=[
        ColumnCluster(
            canonical_name="coo",
            source_columns=["coo_hans", "coo_other"],
            value_mapping={"GCB": "GCB", "Germinal center B-cell-like": "GCB", "ABC": "ABC"},
        )
    ])
    result = harmonize_columns.apply_column_clusters(df, plan)

    assert "coo" in result.columns
    assert "coo_hans" not in result.columns
    assert "coo_other" not in result.columns
    assert list(result["coo"]) == ["GCB", "GCB", "ABC"]


def test_apply_column_clusters_flags_same_row_conflicts_and_keeps_first():
    df = pd.DataFrame({
        "gsm_id": ["G1"],
        "col_a": ["x"],
        "col_b": ["y"],
    })
    plan = CrossCohortMappingPlan(clusters=[
        ColumnCluster(canonical_name="merged", source_columns=["col_a", "col_b"]),
    ])
    result = harmonize_columns.apply_column_clusters(df, plan)

    assert result.iloc[0]["merged"] == "x"
    assert "more than one source column" in result.attrs["harmonize_columns_notes"]


def test_apply_column_clusters_leaves_unmapped_values_raw_and_notes_it():
    df = pd.DataFrame({"col_a": ["weird_value"], "col_b": [None]})
    plan = CrossCohortMappingPlan(clusters=[
        ColumnCluster(canonical_name="merged", source_columns=["col_a", "col_b"], value_mapping={"x": "y"}),
    ])
    result = harmonize_columns.apply_column_clusters(df, plan)

    assert result.iloc[0]["merged"] == "weird_value"
    assert "weird_value" in result.attrs["harmonize_columns_notes"]


def test_apply_column_clusters_does_not_clobber_unrelated_existing_column():
    df = pd.DataFrame({"treatment": ["chemo"], "col_a": ["x"], "col_b": ["y"]})
    plan = CrossCohortMappingPlan(clusters=[
        ColumnCluster(canonical_name="treatment", source_columns=["col_a", "col_b"]),
    ])
    result = harmonize_columns.apply_column_clusters(df, plan)

    assert result.iloc[0]["treatment"] == "chemo"
    assert "col_a" in result.columns and "col_b" in result.columns
    assert "clobbering" in result.attrs["harmonize_columns_notes"]


def test_apply_column_clusters_single_source_column_renames_onto_canonical_name():
    """Master-alignment case: only one cohort's column matches an existing
    master concept -- still needs renaming so it joins the master column."""
    df = pd.DataFrame({"hans_algorithm_call": ["GCB", "ABC"]})
    plan = CrossCohortMappingPlan(clusters=[
        ColumnCluster(canonical_name="coo", source_columns=["hans_algorithm_call"]),
    ])
    result = harmonize_columns.apply_column_clusters(df, plan)

    assert list(result.columns) == ["coo"]
    assert list(result["coo"]) == ["GCB", "ABC"]


def test_apply_column_clusters_ignores_columns_not_present():
    df = pd.DataFrame({"gsm_id": ["G1"]})
    plan = CrossCohortMappingPlan(clusters=[
        ColumnCluster(canonical_name="ghost", source_columns=["does_not_exist"]),
    ])
    result = harmonize_columns.apply_column_clusters(df, plan)
    assert "ghost" not in result.columns


def test_apply_column_clusters_no_clusters_returns_df_unchanged():
    df = pd.DataFrame({"gsm_id": ["G1"], "col_a": ["x"]})
    result = harmonize_columns.apply_column_clusters(df, CrossCohortMappingPlan())
    assert list(result.columns) == ["gsm_id", "col_a"]
    assert "harmonize_columns_notes" not in result.attrs


# --- plan_column_clusters / get_column_mapping_plan (mocked LLM) -------------

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


def _fake_plan():
    return CrossCohortMappingPlan(clusters=[
        ColumnCluster(canonical_name="coo", source_columns=["coo_hans"], value_mapping={"GCB": "GCB"}),
    ])


def test_plan_column_clusters_includes_diagnosis_and_master_hints_in_prompt(monkeypatch):
    fake_client = _FakeClient(_fake_plan())
    monkeypatch.setattr(harmonize_columns.anthropic, "Anthropic", lambda: fake_client)

    df = pd.DataFrame({"gse_id": ["GSE_A"], "coo_hans": ["GCB"]})
    result = harmonize_columns.plan_column_clusters(
        df, ["coo_hans"], diagnosis_ctx="DLBCL",
        seed_hints="Known disease-specific concepts...\n- For DLBCL...",
        master_summary="- coo [cohorts: GSE_OLD]: 'GCB' (n=3)",
    )

    assert isinstance(result, CrossCohortMappingPlan)
    call = fake_client.messages.calls[0]
    assert "DLBCL" in call["system"]
    assert "Known disease-specific concepts" in call["system"]
    assert "GSE_OLD" in call["system"]
    assert "coo_hans" in call["messages"][0]["content"]


def test_get_column_mapping_plan_caches_across_calls(tmp_path, monkeypatch):
    fake_client = _FakeClient(_fake_plan())
    monkeypatch.setattr(harmonize_columns.anthropic, "Anthropic", lambda: fake_client)

    df = pd.DataFrame({"gse_id": ["GSE_A"], "coo_hans": ["GCB"]})
    plan1 = harmonize_columns.get_column_mapping_plan(
        df, ["coo_hans"], diagnosis_ctx="DLBCL", cache_dir=tmp_path,
    )
    plan2 = harmonize_columns.get_column_mapping_plan(
        df, ["coo_hans"], diagnosis_ctx="DLBCL", cache_dir=tmp_path,
    )

    assert len(fake_client.messages.calls) == 1  # second call reused the cache
    assert plan1.clusters[0].canonical_name == plan2.clusters[0].canonical_name == "coo"


def test_get_column_mapping_plan_cache_miss_on_different_columns(tmp_path, monkeypatch):
    fake_client = _FakeClient(_fake_plan())
    monkeypatch.setattr(harmonize_columns.anthropic, "Anthropic", lambda: fake_client)

    df = pd.DataFrame({"gse_id": ["GSE_A"], "coo_hans": ["GCB"], "other_col": ["x"]})
    harmonize_columns.get_column_mapping_plan(df, ["coo_hans"], diagnosis_ctx="DLBCL", cache_dir=tmp_path)
    harmonize_columns.get_column_mapping_plan(df, ["other_col"], diagnosis_ctx="DLBCL", cache_dir=tmp_path)

    assert len(fake_client.messages.calls) == 2


# --- value-mapping phase 2: _cluster_value_counts / _mapping_covers ---------

def test_cluster_value_counts_none_for_single_column():
    df = pd.DataFrame({"col_a": ["GCB", "ABC"]})
    assert harmonize_columns._cluster_value_counts(df, ["col_a"]) is None


def test_cluster_value_counts_none_when_only_one_distinct_value():
    df = pd.DataFrame({"col_a": ["GCB", "GCB"], "col_b": ["GCB", None]})
    assert harmonize_columns._cluster_value_counts(df, ["col_a", "col_b"]) is None


def test_cluster_value_counts_across_multiple_columns():
    df = pd.DataFrame({"col_a": ["GCB", "ABC"], "col_b": ["Germinal center", None]})
    counts = harmonize_columns._cluster_value_counts(df, ["col_a", "col_b"])
    assert set(counts.index) == {"GCB", "ABC", "Germinal center"}


def test_mapping_covers_true_when_all_values_present_case_insensitive():
    counts = pd.Series([2, 1], index=["GCB", "ABC"])
    assert harmonize_columns._mapping_covers({"gcb": "GCB", "ABC": "ABC"}, counts)


def test_mapping_covers_false_when_incomplete():
    counts = pd.Series([2, 1], index=["GCB", "ABC"])
    assert not harmonize_columns._mapping_covers({"GCB": "GCB"}, counts)


def test_mapping_covers_false_when_empty():
    counts = pd.Series([2], index=["GCB"])
    assert not harmonize_columns._mapping_covers({}, counts)


# --- fill_value_mappings / plan_value_mapping (mocked LLM) -------------------

def _fake_value_plan():
    return ValueUnificationPlan(mappings=[
        ValueMappingEntry(raw_value="GCB", canonical_value="GCB"),
        ValueMappingEntry(raw_value="GCB DLBCL", canonical_value="GCB"),
        ValueMappingEntry(raw_value="germinal center", canonical_value="GCB"),
        ValueMappingEntry(raw_value="ABC", canonical_value="ABC"),
        ValueMappingEntry(raw_value="ABC DLBCL", canonical_value="ABC"),
    ])


def test_fill_value_mappings_skips_single_column_cluster(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("should not call the LLM for a single-column cluster")

    monkeypatch.setattr(harmonize_columns, "plan_value_mapping", _boom)

    df = pd.DataFrame({"coo_hans": ["GCB", "ABC"]})
    plan = CrossCohortMappingPlan(clusters=[ColumnCluster(canonical_name="coo", source_columns=["coo_hans"])])

    result = harmonize_columns.fill_value_mappings(df, plan, diagnosis_ctx="DLBCL")
    assert result.clusters[0].value_mapping == {}


def test_fill_value_mappings_skips_when_already_fully_covered(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("should not call the LLM when value_mapping already covers every value")

    monkeypatch.setattr(harmonize_columns, "plan_value_mapping", _boom)

    df = pd.DataFrame({"coo_hans": ["GCB"], "coo_other": ["ABC"]})
    plan = CrossCohortMappingPlan(clusters=[
        ColumnCluster(
            canonical_name="coo", source_columns=["coo_hans", "coo_other"],
            value_mapping={"GCB": "GCB", "ABC": "ABC"},
        )
    ])

    result = harmonize_columns.fill_value_mappings(df, plan, diagnosis_ctx="DLBCL")
    assert result.clusters[0].value_mapping == {"GCB": "GCB", "ABC": "ABC"}


def test_fill_value_mappings_backfills_incomplete_multi_column_cluster(monkeypatch):
    calls = []

    def _fake_plan_value_mapping(canonical_name, value_counts, diagnosis_ctx, seed_hints="", model=None):
        calls.append((canonical_name, set(value_counts.index), diagnosis_ctx))
        return _fake_value_plan()

    monkeypatch.setattr(harmonize_columns, "plan_value_mapping", _fake_plan_value_mapping)

    df = pd.DataFrame({
        "coo_hans": ["GCB", "GCB DLBCL"],
        "coo_other": ["germinal center", "ABC"],
        "coo_third": ["ABC DLBCL", None],
    })
    plan = CrossCohortMappingPlan(clusters=[
        ColumnCluster(canonical_name="coo", source_columns=["coo_hans", "coo_other", "coo_third"]),
    ])

    result = harmonize_columns.fill_value_mappings(df, plan, diagnosis_ctx="DLBCL")

    assert len(calls) == 1
    assert calls[0][0] == "coo"
    assert calls[0][2] == "DLBCL"
    assert result.clusters[0].value_mapping == _fake_value_plan().value_mapping


def test_plan_value_mapping_prompt_includes_value_counts(monkeypatch):
    fake_client = _FakeClient(_fake_value_plan())
    monkeypatch.setattr(harmonize_columns.anthropic, "Anthropic", lambda: fake_client)

    value_counts = pd.Series([3, 2], index=["GCB", "ABC"])
    result = harmonize_columns.plan_value_mapping("coo", value_counts, diagnosis_ctx="DLBCL")

    assert isinstance(result, ValueUnificationPlan)
    call = fake_client.messages.calls[0]
    assert "coo" in call["system"]
    assert "DLBCL" in call["system"]
    assert "'GCB' (n=3)" in call["system"]
    assert "'ABC' (n=2)" in call["system"]


def test_get_column_mapping_plan_resolves_values_end_to_end(tmp_path, monkeypatch):
    """Integration: the outer caching wrapper runs both phases and caches
    the fully-resolved (values-unified) plan."""
    column_plan = CrossCohortMappingPlan(clusters=[
        ColumnCluster(canonical_name="coo", source_columns=["coo_hans", "coo_other"]),
    ])

    call_log = []

    class _TwoPhaseMessages:
        def parse(self, **kwargs):
            call_log.append(kwargs["output_format"])
            if kwargs["output_format"] is CrossCohortMappingPlan:
                return _FakeResponse(column_plan)
            return _FakeResponse(ValueUnificationPlan(mappings=[
                ValueMappingEntry(raw_value="GCB", canonical_value="GCB"),
                ValueMappingEntry(raw_value="germinal center", canonical_value="GCB"),
            ]))

    class _TwoPhaseClient:
        messages = _TwoPhaseMessages()

    monkeypatch.setattr(harmonize_columns.anthropic, "Anthropic", lambda: _TwoPhaseClient())

    df = pd.DataFrame({"gse_id": ["GSE_A", "GSE_B"], "coo_hans": ["GCB", None], "coo_other": [None, "germinal center"]})
    plan = harmonize_columns.get_column_mapping_plan(
        df, ["coo_hans", "coo_other"], diagnosis_ctx="DLBCL", cache_dir=tmp_path,
    )

    assert call_log == [CrossCohortMappingPlan, ValueUnificationPlan]
    assert plan.clusters[0].value_mapping == {"GCB": "GCB", "germinal center": "GCB"}

    # cached: a second call resolves entirely from disk, no further LLM calls
    plan2 = harmonize_columns.get_column_mapping_plan(
        df, ["coo_hans", "coo_other"], diagnosis_ctx="DLBCL", cache_dir=tmp_path,
    )
    assert len(call_log) == 2
    assert plan2.clusters[0].value_mapping == {"GCB": "GCB", "germinal center": "GCB"}
