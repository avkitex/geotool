import pandas as pd
import pytest

from geotool import clinical_annotate as ca


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


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Complete Response", "CR"),
        ("complete response", "CR"),
        ("CR", "CR"),
        ("partial response", "PR"),
        ("Stable Disease", "SD"),
        ("Progressive Disease", "PD"),
        ("progression", "PD"),
        ("Not Evaluable", "NE"),
        ("Not Available", "NE"),
        ("NA", "NE"),
        ("NE", "NE"),
        ("N/A", "NE"),
        ("some unrelated text", None),
        (None, None),
        ("", None),
    ],
)
def test_normalize_recist(raw, expected):
    assert ca.normalize_recist(raw) == expected


def test_parse_event_mapping_numeric():
    mapping = ca.parse_event_mapping("1=death, 0=censored")
    assert mapping == {"1": 1, "0": 0}


def test_parse_event_mapping_text_labels():
    mapping = ca.parse_event_mapping("Dead=event, Alive=censored")
    assert mapping == {"dead": 1, "alive": 0}


def test_parse_event_mapping_unparseable_returns_empty():
    assert ca.parse_event_mapping("") == {}
    assert ca.parse_event_mapping("some free text with no equals signs") == {}


def test_remap_event_column_maps_known_values_and_passes_through_unknown():
    series = pd.Series(["Dead", "Alive", "Unknown"])
    remapped = ca.remap_event_column(series, "Dead=event, Alive=censored")
    assert list(remapped) == [1, 0, "Unknown"]


def test_remap_event_column_maps_punctuation_only_placeholder_to_none():
    """Regression test: GSE183795's "survival status" column ('1'/'0'/'?')
    left '?' as a literal stray string in OS_event -- '1'/'0' matched the
    numeric mapping, '?' didn't, and fell through to raw passthrough. A
    punctuation-only value like '?' is GEO's own "not reported" marker, not
    a real unmapped value worth surfacing like "Unknown" (still passed
    through raw, see the test above) -- so it should become a real missing
    value instead."""
    series = pd.Series(["1", "0", "?"])
    remapped = ca.remap_event_column(series, "1=death, 0=censored")
    result = list(remapped)
    assert result[:2] == [1, 0]
    assert pd.isna(result[2])


def test_remap_event_column_returns_raw_when_meaning_unparseable():
    series = pd.Series(["Dead", "Alive"])
    remapped = ca.remap_event_column(series, "")
    assert list(remapped) == ["Dead", "Alive"]


def test_remap_event_column_matches_value_with_repeated_label_prefix():
    """Regression test: GSE10846 live run showed 'Follow up status: DEAD' style
    values (raw characteristic 'clinical info_3: Follow up status: DEAD' only
    splits on the first colon) don't exact-match a plain 'dead' mapping key."""
    series = pd.Series(["Follow up status: DEAD", "Follow up status: ALIVE"])
    remapped = ca.remap_event_column(series, "Dead=event, Alive=censored")
    assert list(remapped) == [1, 0]


def test_convert_time_to_months_extracts_number_from_labeled_value():
    """Regression test: GSE10846 live run showed 'Follow up years: 5.2' style
    values producing NaN under plain pd.to_numeric."""
    series = pd.Series(["Follow up years: 5.2", "Follow up years: 1.0"])
    result = ca.convert_time_to_months(series, "years")
    assert list(result) == [pytest.approx(62.4), pytest.approx(12.0)]


def test_convert_time_to_months_genuinely_unparseable_value_does_not_crash():
    """Regression test: a second GSE10846 live run crashed with
    "Invalid value '[None ...]' for dtype 'float64'" because the fallback
    path assigned None (not NaN) into an existing float64 column slice --
    pandas 3.x rejects that. Must produce NaN, not raise."""
    series = pd.Series(["Follow up years: 5.2", "no number here at all", None])
    result = ca.convert_time_to_months(series, "years")
    assert result.iloc[0] == pytest.approx(62.4)
    assert pd.isna(result.iloc[1])
    assert pd.isna(result.iloc[2])


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Chemotherapy: CHOP-Like Regimen", "CHOP-Like Regimen"),
        ("Stage: 3", "3"),
        ("no colon here", "no colon here"),
        ("R-CHOP", "R-CHOP"),
        (None, None),
    ],
)
def test_strip_repeated_label(raw, expected):
    assert ca.strip_repeated_label(raw) == expected


@pytest.mark.parametrize(
    "unit,value,expected",
    [
        ("months", 24.0, 24.0),
        ("years", 2.0, 24.0),
        ("days", 30.4375, 1.0),
        ("unknown", 24.0, 24.0),
    ],
)
def test_convert_time_to_months(unit, value, expected):
    result = ca.convert_time_to_months(pd.Series([value]), unit)
    assert result.iloc[0] == pytest.approx(expected)


def make_samples_df():
    return pd.DataFrame({
        "gsm_id": ["GSM1", "GSM2", "GSM3"],
        "gse_id": ["GSE1", "GSE1", "GSE1"],
        "platform_id": ["GPL1", "GPL1", "GPL1"],
        "organism": ["Homo sapiens", "Homo sapiens", "Homo sapiens"],  # constant -> dropped
        "drug": ["R-CHOP", "R-CHOP", "CHOP"],
        "dose_schedule": ["375mg/m2 d1", "375mg/m2 d1", "standard"],
        "tumor response": ["Complete Response", "partial response", "PD"],
        "os_months": ["24.5", "10.2", "5.0"],
        "vital_status": ["Alive", "Dead", "Dead"],
    })


def make_plan():
    return ca.ColumnMappingPlan(
        redundant_columns=["organism"],
        treatment_columns=["drug"],
        treatment_detail_columns=["dose_schedule"],
        response_column="tumor response",
        survival=[
            ca.SurvivalMapping(
                survival_type="OS", time_column="os_months", time_unit="months",
                event_column="vital_status", event_value_meaning="Dead=event, Alive=censored",
            )
        ],
    )


def test_apply_column_mapping_full_pipeline():
    out = ca.apply_column_mapping(make_samples_df(), make_plan())

    assert "organism" not in out.columns  # redundant, dropped
    assert set(out["treatment"]) == {"R-CHOP", "CHOP"}
    assert set(out["treatment_detail"]) == {"375mg/m2 d1", "standard"}
    assert list(out["recist"]) == ["CR", "PR", "PD"]
    assert list(out["OS_time"]) == [24.5, 10.2, 5.0]
    assert list(out["OS_event"]) == [0, 1, 1]
    assert "os_months" not in out.columns
    assert "vital_status" not in out.columns


def test_apply_column_mapping_treatment_column_with_missing_values():
    """Regression test: a real GSE181063 live run crashed with
    'float' object has no attribute 'lower' -- some pandas versions leave a
    missing value as an actual float NaN even after .astype(str) (rather
    than the literal string "nan"), and float('nan') is truthy so the old
    `if v` guard didn't filter it out before calling v.lower()."""
    df = pd.DataFrame({
        "gsm_id": ["GSM1", "GSM2", "GSM3"],
        "gse_id": ["GSE1", "GSE1", "GSE1"],
        "platform_id": ["GPL1", "GPL1", "GPL1"],
        "firstline_regimen": ["CHOP-R", None, "CVP-R"],
    })
    plan = ca.ColumnMappingPlan(treatment_columns=["firstline_regimen"])

    out = ca.apply_column_mapping(df, plan)

    assert list(out["treatment"]) == ["CHOP-R", "", "CVP-R"]


def test_apply_column_mapping_handles_repeated_label_values_end_to_end():
    """Regression test for the GSE10846 live smoke test: GEO characteristics
    like 'clinical info_5: Chemotherapy: CHOP-Like Regimen' parse (via
    annotate.parse_characteristics, which only splits on the first colon)
    into a column literally named 'clinical info_5' whose values still embed
    'Chemotherapy: '. Treatment/response/survival must come out clean anyway.
    """
    df = pd.DataFrame({
        "gsm_id": ["GSM1", "GSM2"],
        "gse_id": ["GSE10846", "GSE10846"],
        "platform_id": ["GPL570", "GPL570"],
        "clinical info_5": ["Chemotherapy: CHOP-Like Regimen", "Chemotherapy: R-CHOP-Like Regimen"],
        "clinical info_2": ["Final microarray diagnosis: GCB DLBCL", "Final microarray diagnosis: ABC DLBCL"],
        "clinical info_3": ["Follow up status: DEAD", "Follow up status: ALIVE"],
        "clinical info_4": ["Follow up years: 1.5", "Follow up years: 6.9"],
    })
    plan = ca.ColumnMappingPlan(
        treatment_columns=["clinical info_5"],
        survival=[
            ca.SurvivalMapping(
                survival_type="OS", time_column="clinical info_4", time_unit="years",
                event_column="clinical info_3", event_value_meaning="Dead=event, Alive=censored",
            )
        ],
    )
    out = ca.apply_column_mapping(df, plan)

    assert list(out["treatment"]) == ["CHOP-Like Regimen", "R-CHOP-Like Regimen"]
    assert list(out["OS_event"]) == [1, 0]
    assert out["OS_time"].tolist() == [pytest.approx(1.5 * 12), pytest.approx(6.9 * 12)]
    # untouched raw column also benefits from the generic label-stripping pass
    assert list(out["clinical info_2"]) == ["GCB DLBCL", "ABC DLBCL"]


def test_apply_column_mapping_protects_identity_columns_even_if_constant():
    out = ca.apply_column_mapping(make_samples_df(), ca.ColumnMappingPlan())
    for col in ca.PROTECTED_COLUMNS:
        assert col in out.columns


def test_apply_column_mapping_protects_identity_columns_even_when_llm_flags_them_redundant():
    """Regression test: a real GSE19246 live run showed gse_id/platform_id
    missing from annotation.tsv entirely. Root cause: they ARE constant
    within one cohort, so the LLM reasonably listed them in
    redundant_columns -- and the drop step used that list directly without
    filtering out PROTECTED_COLUMNS first (the code-level safety net only
    guards the *second* loop, not this one)."""
    plan = ca.ColumnMappingPlan(redundant_columns=["gse_id", "platform_id", "organism"])
    out = ca.apply_column_mapping(make_samples_df(), plan)
    assert "gse_id" in out.columns
    assert "platform_id" in out.columns
    assert "organism" not in out.columns  # non-protected redundant column still dropped


def test_apply_column_mapping_drops_constant_columns_not_flagged_by_llm():
    # LLM plan doesn't mention 'organism' as redundant -- code-level safety net should still drop it.
    plan = ca.ColumnMappingPlan()
    out = ca.apply_column_mapping(make_samples_df(), plan)
    assert "organism" not in out.columns


def test_apply_column_mapping_skips_survival_entry_missing_event_column():
    df = make_samples_df()
    plan = ca.ColumnMappingPlan(survival=[ca.SurvivalMapping(survival_type="OS", time_column="os_months", event_column=None)])
    out = ca.apply_column_mapping(df, plan)
    assert "OS_time" not in out.columns
    assert "OS_event" not in out.columns
    assert "os_months" in out.columns  # left untouched, not silently dropped


def test_apply_column_mapping_notes_unparseable_event_meaning():
    df = make_samples_df()
    plan = ca.ColumnMappingPlan(
        survival=[ca.SurvivalMapping(survival_type="OS", time_column="os_months", event_column="vital_status", event_value_meaning="")]
    )
    out = ca.apply_column_mapping(df, plan)
    assert list(out["OS_event"]) == ["Alive", "Dead", "Dead"]  # left raw
    assert "could not parse event_value_meaning" in out.attrs.get("clinical_annotate_notes", "")


def test_plan_column_mapping_returns_parsed_plan(monkeypatch):
    plan = make_plan()
    fake_client = _FakeClient(plan)
    monkeypatch.setattr(ca.anthropic, "Anthropic", lambda: fake_client)

    result = ca.plan_column_mapping(make_samples_df())
    assert result == plan
    assert len(fake_client.messages.calls) == 1


def test_classify_expression_status_no_matrix():
    """GSE108651 shape: only differential-expression/splicing-analysis
    output was ever published, no raw or normalized matrix at all."""
    assert ca.classify_expression_status([], has_matrix=False) == ca.EXPRESSION_STATUS_NO_MATRIX
    # qc_notes are irrelevant once there's no matrix to have generated them from.
    assert ca.classify_expression_status(["some note"], has_matrix=False) == ca.EXPRESSION_STATUS_NO_MATRIX


def test_classify_expression_status_ok_when_matrix_found_and_clean():
    assert ca.classify_expression_status([], has_matrix=True) == ca.EXPRESSION_STATUS_OK


def test_classify_expression_status_unparseable():
    notes = ["GSE1_TPM.csv.gz: could not parse for QC"]
    assert ca.classify_expression_status(notes, has_matrix=True) == ca.EXPRESSION_STATUS_UNPARSEABLE


def test_classify_expression_status_not_log2_transformed():
    notes = ["expression.tsv.gz: linear-scale, not log2-transformed (max value 16096.1)"]
    assert ca.classify_expression_status(notes, has_matrix=True) == ca.EXPRESSION_STATUS_NOT_LOG2_TRANSFORMED


def test_classify_expression_status_negative_values():
    notes = ["expression.tsv.gz: 3 negative value(s) found -- possible log2 transform without a +1 pseudocount"]
    assert ca.classify_expression_status(notes, has_matrix=True) == ca.EXPRESSION_STATUS_NEGATIVE_VALUES


def test_classify_expression_status_joins_both_tags_when_both_present():
    notes = [
        "expression.tsv.gz: linear-scale, not log2-transformed (max value 99.0)",
        "expression.tsv.gz: 1 negative value(s) found -- possible log2 transform without a +1 pseudocount",
    ]
    status = ca.classify_expression_status(notes, has_matrix=True)
    assert status == f"{ca.EXPRESSION_STATUS_NOT_LOG2_TRANSFORMED};{ca.EXPRESSION_STATUS_NEGATIVE_VALUES}"


def test_classify_expression_status_looks_transposed():
    notes = ["expression.tsv.gz: more columns (20) than rows (15) -- expected features (genes) as rows and samples as columns; this matrix may be transposed"]
    assert ca.classify_expression_status(notes, has_matrix=True) == ca.EXPRESSION_STATUS_LOOKS_TRANSPOSED


def test_classify_expression_status_low_gene_count():
    notes = ["expression.tsv.gz: only 7833 genes (< 16000) -- likely a filtered/truncated gene list, not the full transcriptome"]
    assert ca.classify_expression_status(notes, has_matrix=True) == ca.EXPRESSION_STATUS_LOW_GENE_COUNT


def test_classify_expression_status_two_channel_signal_unresolved():
    """A two-channel cohort whose signal/tumor channel couldn't be resolved
    is flagged even with an otherwise clean matrix -- readiness is about
    whether the right file exists, not just whether the ratio itself
    processed without incident."""
    assert (
        ca.classify_expression_status([], has_matrix=True, two_channel_signal_unresolved=True)
        == ca.EXPRESSION_STATUS_TWO_CHANNEL_SIGNAL_UNRESOLVED
    )


def test_classify_expression_status_two_channel_signal_unresolved_joins_with_other_tags():
    notes = ["expression.tsv.gz: linear-scale, not log2-transformed (max value 99.0)"]
    status = ca.classify_expression_status(notes, has_matrix=True, two_channel_signal_unresolved=True)
    assert status == f"{ca.EXPRESSION_STATUS_TWO_CHANNEL_SIGNAL_UNRESOLVED};{ca.EXPRESSION_STATUS_NOT_LOG2_TRANSFORMED}"


def test_classify_expression_status_no_numeric_data():
    """Live example: GSE243850's misparsed multi-row-header file -- must not
    silently come out "ok" just because none of check_expression_qc's other
    value-based checks had anything numeric to look at."""
    notes = ["GSE243850_Raw_counts_and_normalized_read_count.tsv.gz: no numeric column(s) found among 127 column(s) -- likely a misparsed/shifted header row (e.g. a multi-row header), not a genuine identifier-only matrix"]
    assert ca.classify_expression_status(notes, has_matrix=True) == ca.EXPRESSION_STATUS_NO_NUMERIC_DATA
