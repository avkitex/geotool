import pandas as pd

from geotool import sample_id_matching as sim


def make_annotation(rows):
    return pd.DataFrame(rows)


def result_map(df):
    return dict(zip(df["expression_id"], df["gsm_id"]))


def result_methods(df):
    return dict(zip(df["expression_id"], df["match_method"]))


# --- exact gsm_id ------------------------------------------------------------

def test_matches_expression_column_that_is_literally_the_gsm_id():
    annotation = make_annotation([
        {"gsm_id": "GSM1", "title": "sample one"},
        {"gsm_id": "GSM2", "title": "sample two"},
    ])
    result = sim.match_expression_columns(["GSM1", "GSM2"], annotation)
    assert result_map(result) == {"GSM1": "GSM1", "GSM2": "GSM2"}
    assert set(result_methods(result).values()) == {"exact_gsm_id"}


# --- exact text match ---------------------------------------------------------

def test_matches_exact_title_real_gse241402_shape():
    """Real GSE241402 shape: expression matrix column names are exactly
    each GSM's title ("DMSO_1", "PRMT5i_1")."""
    annotation = make_annotation([
        {"gsm_id": "GSM1", "title": "DMSO_1"},
        {"gsm_id": "GSM2", "title": "DMSO_2"},
        {"gsm_id": "GSM3", "title": "PRMT5i_1"},
    ])
    result = sim.match_expression_columns(["PRMT5i_1", "DMSO_1", "DMSO_2"], annotation)
    assert result_map(result) == {"PRMT5i_1": "GSM3", "DMSO_1": "GSM1", "DMSO_2": "GSM2"}
    assert set(result_methods(result).values()) == {"exact"}


# --- normalized exact match ---------------------------------------------------

def test_matches_after_normalizing_punctuation_and_case():
    annotation = make_annotation([
        {"gsm_id": "GSM1", "title": "D5-DMSO.r1"},
        {"gsm_id": "GSM2", "title": "D5-EPZ.r1"},
    ])
    result = sim.match_expression_columns(["d5_dmso_r1", "d5_epz_r1"], annotation)
    assert result_map(result) == {"d5_dmso_r1": "GSM1", "d5_epz_r1": "GSM2"}
    assert set(result_methods(result).values()) == {"normalized_exact"}


# --- substring match -----------------------------------------------------------

def test_matches_via_substring_in_description_real_gse282794_shape():
    """Real GSE282794 shape: expression column "DMSO_1" appears embedded in
    the GSM's description field ("Sample 1 DMSO_1"), not as its title."""
    annotation = make_annotation([
        {"gsm_id": "GSM1", "title": "A172 cells, DMSO, rep 1", "description": "Sample 1 DMSO_1"},
        {"gsm_id": "GSM2", "title": "A172 cells, DMSO, rep 2", "description": "Sample 2 DMSO_2"},
        {"gsm_id": "GSM3", "title": "A172 cells, PRMT5i, rep 1", "description": "Sample 4 MRTX_1"},
    ])
    result = sim.match_expression_columns(["DMSO_1", "DMSO_2", "MRTX_1"], annotation)
    assert result_map(result) == {"DMSO_1": "GSM1", "DMSO_2": "GSM2", "MRTX_1": "GSM3"}
    assert set(result_methods(result).values()) == {"substring"}


def test_matches_via_normalized_substring_real_gse335198_shape():
    """Real GSE335198 shape: expression column "dmso1" (no separator, no
    case) is a normalized substring of the GSM's description
    ("HCT116_DMSO_1 Total RNA from...")."""
    annotation = make_annotation([
        {"gsm_id": "GSM1", "title": "HCT116 DMSO RNA-seq replicate 1", "description": "HCT116_DMSO_1 Total RNA from HCT116 cells"},
        {"gsm_id": "GSM2", "title": "HCT116 GSK025 RNA-seq replicate 1", "description": "HCT116_GSK025_1 Total RNA from HCT116 cells"},
    ])
    result = sim.match_expression_columns(["dmso1", "gsk0251"], annotation)
    assert result_map(result) == {"dmso1": "GSM1", "gsk0251": "GSM2"}


# --- reverse substring -----------------------------------------------------------

def test_matches_via_reverse_substring():
    """The metadata field is the short label; the expression column is
    decorated with extra context around it."""
    annotation = make_annotation([
        {"gsm_id": "GSM1", "title": "veh"},
        {"gsm_id": "GSM2", "title": "trt"},
    ])
    result = sim.match_expression_columns(["sample_veh_rep1", "sample_trt_rep1"], annotation)
    assert result_map(result) == {"sample_veh_rep1": "GSM1", "sample_trt_rep1": "GSM2"}
    assert set(result_methods(result).values()) == {"reverse_substring"}


# --- positional fallback -------------------------------------------------------

def test_positional_fallback_when_no_name_based_match_possible():
    annotation = make_annotation([
        {"gsm_id": "GSM1", "title": "condition A"},
        {"gsm_id": "GSM2", "title": "condition B"},
        {"gsm_id": "GSM3", "title": "condition C"},
    ])
    result = sim.match_expression_columns(["col1", "col2", "col3"], annotation)
    assert result_map(result) == {"col1": "GSM1", "col2": "GSM2", "col3": "GSM3"}
    assert set(result_methods(result).values()) == {"positional_fallback"}


def test_positional_fallback_only_covers_the_unresolved_remainder():
    """One column resolves by exact title match; the other two fall back to
    order among whichever gsm_ids weren't already used."""
    annotation = make_annotation([
        {"gsm_id": "GSM1", "title": "known_label"},
        {"gsm_id": "GSM2", "title": "condition B"},
        {"gsm_id": "GSM3", "title": "condition C"},
    ])
    result = sim.match_expression_columns(["known_label", "colX", "colY"], annotation)
    mapping = result_map(result)
    assert mapping["known_label"] == "GSM1"
    assert set(mapping.values()) == {"GSM1", "GSM2", "GSM3"}
    assert mapping["colX"] == "GSM2"
    assert mapping["colY"] == "GSM3"


def test_no_positional_fallback_when_counts_dont_match():
    """A genuine sample-count mismatch (matrix has fewer/more columns than
    the cohort has samples) must surface as unmatched, not a guess."""
    annotation = make_annotation([
        {"gsm_id": "GSM1", "title": "condition A"},
        {"gsm_id": "GSM2", "title": "condition B"},
        {"gsm_id": "GSM3", "title": "condition C"},
    ])
    result = sim.match_expression_columns(["col1", "col2"], annotation)
    assert result_map(result) == {"col1": None, "col2": None}
    assert set(result_methods(result).values()) == {"unmatched"}


# --- ambiguity is never guessed -------------------------------------------------

def test_ambiguous_exact_match_left_unresolved():
    annotation = make_annotation([
        {"gsm_id": "GSM1", "title": "DMSO"},
        {"gsm_id": "GSM2", "title": "DMSO"},
    ])
    result = sim.match_expression_columns(["DMSO"], annotation)
    row = result.iloc[0]
    assert row["gsm_id"] is None
    assert row["match_method"] == "unmatched"


def test_ambiguous_substring_falls_through_to_next_column_then_unmatched():
    annotation = make_annotation([
        {"gsm_id": "GSM1", "title": "DMSO_1 replicate", "description": "batch DMSO_1"},
        {"gsm_id": "GSM2", "title": "DMSO_1 control", "description": "batch DMSO_1"},
    ])
    result = sim.match_expression_columns(["DMSO_1"], annotation)
    row = result.iloc[0]
    assert row["gsm_id"] is None


# --- confidence -----------------------------------------------------------------

def test_confidence_reflects_match_method_strength():
    annotation = make_annotation([{"gsm_id": "GSM1", "title": "DMSO_1"}])
    result = sim.match_expression_columns(["DMSO_1"], annotation)
    assert result.iloc[0]["confidence"] == 0.95  # "exact"


def test_unmatched_has_zero_confidence():
    annotation = make_annotation([{"gsm_id": "GSM1", "title": "condition A"}, {"gsm_id": "GSM2", "title": "condition B"}])
    result = sim.match_expression_columns(["col1"], annotation)
    assert result.iloc[0]["confidence"] == 0.0


# --- custom text_columns ---------------------------------------------------------

def test_custom_text_columns_restricts_search():
    annotation = make_annotation([
        {"gsm_id": "GSM1", "title": "DMSO_1", "notes": "irrelevant"},
        {"gsm_id": "GSM2", "title": "DMSO_2", "notes": "irrelevant"},
    ])
    result = sim.match_expression_columns(["DMSO_1", "col2"], annotation, text_columns=["notes"])
    # title is excluded from the search, so "DMSO_1" can't be found in "notes"
    # -- and with a genuine second unresolved column, positional fallback
    # can't trivially "resolve" it either (it applies to both, not verifiably).
    row = result[result["expression_id"] == "DMSO_1"].iloc[0]
    assert row["match_method"] == "positional_fallback"  # falls back to order, not a name match
    assert row["gsm_id"] == "GSM1"
