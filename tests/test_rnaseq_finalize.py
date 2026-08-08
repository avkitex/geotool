import json

import pandas as pd
import pytest

from geotool import gene_symbol_mapping as gsm
from geotool import rnaseq_finalize as finalize


def make_reference(gene_to_symbol=None, transcript_to_symbol=None, gene_length=None) -> gsm.GencodeReference:
    gene_to_symbol = gene_to_symbol or {}
    transcript_to_symbol = transcript_to_symbol or {}
    known = frozenset(gene_to_symbol.values()) | frozenset(transcript_to_symbol.values())
    return gsm.GencodeReference("50", transcript_to_symbol, gene_to_symbol, gene_length or {}, known)


# --- unwrap_embedded_ensembl_ids ---------------------------------------------

def test_unwrap_embedded_ensembl_ids_extracts_majority_shape():
    df = pd.DataFrame(
        {"s1": [1.0, 2.0]},
        index=["RP1-67K17.4_ENSG00000237851.1", "FOO_ENSG00000000005.7"],
    )
    result = finalize.unwrap_embedded_ensembl_ids(df)
    assert list(result.index) == ["ENSG00000237851.1", "ENSG00000000005.7"]


def test_unwrap_embedded_ensembl_ids_leaves_minority_shape_unchanged():
    df = pd.DataFrame({"s1": [1.0, 2.0]}, index=["TSPAN6", "TNMD"])
    result = finalize.unwrap_embedded_ensembl_ids(df)
    assert list(result.index) == ["TSPAN6", "TNMD"]


# --- to_linear_scale ----------------------------------------------------------

def test_to_linear_scale_inverts_log2_data():
    df = pd.DataFrame({"s1": [1.0, 5.0], "s2": [2.0, 6.0]})
    linear, was_log2 = finalize.to_linear_scale(df)
    assert was_log2 is True
    assert linear.loc[0, "s1"] == pytest.approx(2 ** 1.0 - 1)


def test_to_linear_scale_leaves_already_linear_data():
    df = pd.DataFrame({"s1": [100.0, 5000.0], "s2": [200.0, 6000.0]})
    linear, was_log2 = finalize.to_linear_scale(df)
    assert was_log2 is False
    pd.testing.assert_frame_equal(linear, df)


# --- looks_like_integer_data ---------------------------------------------------

def test_looks_like_integer_data_true_for_whole_numbers():
    df = pd.DataFrame({"s1": [10.0, 20.0], "s2": [0.0, 15.0]})
    assert finalize.looks_like_integer_data(df) is True


def test_looks_like_integer_data_tolerates_float_noise():
    df = pd.DataFrame({"s1": [10.0000001, 19.9999999]})
    assert finalize.looks_like_integer_data(df) is True


def test_looks_like_integer_data_false_for_fractional_values():
    df = pd.DataFrame({"s1": [10.5, 20.0], "s2": [0.0, 15.25]})
    assert finalize.looks_like_integer_data(df) is False


def test_looks_like_integer_data_false_when_empty():
    df = pd.DataFrame({"s1": [], "s2": []}, dtype=float)
    assert finalize.looks_like_integer_data(df) is False


# --- guess_unit -----------------------------------------------------------------

def test_guess_unit_integer_values_assumed_counts():
    df = pd.DataFrame({"s1": [10.0, 20.0]}, index=["TSPAN6", "TNMD"])
    unit, why = finalize.guess_unit(df, _REF)
    assert unit == "count"
    assert "whole number" in why


def test_guess_unit_non_integer_transcript_level_assumed_cpm():
    ref = make_reference(transcript_to_symbol={"ENST00000000001": "TSPAN6", "ENST00000000002": "TNMD"})
    df = pd.DataFrame({"s1": [10.5, 20.25]}, index=["ENST00000000001", "ENST00000000002"])
    unit, why = finalize.guess_unit(df, ref)
    assert unit == "cpm"
    assert "transcript-level" in why


def test_guess_unit_non_integer_gene_level_assumed_fpkm():
    """Genuinely non-integer values (not a lossy log2 reconstruction --
    see the integer_check_matrix tests below for that case), ENSG
    gene-level identifiers, unit unrecoverable from filename or content
    alone."""
    ref = make_reference(gene_to_symbol={"ENSG00000000003": "TSPAN6", "ENSG00000000005": "TNMD"})
    df = pd.DataFrame({"s1": [8.5, 0.25]}, index=["ENSG00000000003", "ENSG00000000005"])
    unit, why = finalize.guess_unit(df, ref)
    assert unit == "fpkm"
    assert "gene-level" in why


def test_guess_unit_non_integer_symbol_level_assumed_fpkm():
    df = pd.DataFrame({"s1": [8.5, 0.25]}, index=["TSPAN6", "TNMD"])
    unit, why = finalize.guess_unit(df, _REF)
    assert unit == "fpkm"


def test_guess_unit_integer_check_matrix_overrides_linear_matrix():
    """Real GSE230065 shape: linear_matrix is a *reconstruction* (2**x - 1
    from an already-log2, 3-decimal-rounded file) and looks non-integer,
    but integer_check_matrix (the preserved pre-transform raw file) is
    exact and genuinely integer -- the count guess must win."""
    ref = make_reference(gene_to_symbol={"ENSG00000000003": "TSPAN6"})
    linear_matrix = pd.DataFrame({"s1": [428.9], "s2": [643.6]}, index=["ENSG00000000003"])
    integer_check_matrix = pd.DataFrame({"s1": [429.0], "s2": [644.0]}, index=["ENSG00000000003"])
    unit, why = finalize.guess_unit(linear_matrix, ref, integer_check_matrix=integer_check_matrix)
    assert unit == "count"
    assert "whole number" in why


def test_guess_unit_without_integer_check_matrix_falls_back_to_linear_matrix():
    df = pd.DataFrame({"s1": [8.5, 0.25]}, index=["TSPAN6", "TNMD"])
    unit, why = finalize.guess_unit(df, _REF, integer_check_matrix=None)
    assert unit == "fpkm"


# --- find_original_raw_file -----------------------------------------------------

def test_find_original_raw_file_present(tmp_path):
    expr_dir = tmp_path / "expression"
    expr_dir.mkdir()
    primary = expr_dir / "GSE1_genes.tsv.gz"
    primary.write_bytes(b"x")
    original = expr_dir / "GSE1_genes.original.tsv.gz"
    original.write_bytes(b"y")

    assert finalize.find_original_raw_file(primary) == original


def test_find_original_raw_file_absent(tmp_path):
    expr_dir = tmp_path / "expression"
    expr_dir.mkdir()
    primary = expr_dir / "GSE1_genes.tsv.gz"
    primary.write_bytes(b"x")

    assert finalize.find_original_raw_file(primary) is None


# --- renormalize_to_1e6 --------------------------------------------------------

def test_renormalize_to_1e6_rescales_columns_to_sum():
    df = pd.DataFrame({"s1": [1.0, 3.0], "s2": [2.0, 2.0]}, index=["A", "B"])
    result, err = finalize.renormalize_to_1e6(df)
    assert err is None
    assert result["s1"].sum() == pytest.approx(1e6)
    assert result["s2"].sum() == pytest.approx(1e6)


def test_renormalize_to_1e6_reports_zero_sum_columns():
    df = pd.DataFrame({"s1": [0.0, 0.0], "s2": [1.0, 1.0]}, index=["A", "B"])
    result, err = finalize.renormalize_to_1e6(df)
    assert result is None
    assert "s1" in err


# --- restrict_to_clean_genes ---------------------------------------------------

def test_restrict_to_clean_genes_keeps_only_known_symbols():
    df = pd.DataFrame({"s1": [1.0, 2.0, 3.0]}, index=["TSPAN6", "XLOC_1", "TNMD"])
    result = finalize.restrict_to_clean_genes(df, {"TSPAN6", "TNMD"})
    assert set(result.index) == {"TSPAN6", "TNMD"}


def test_restrict_to_clean_genes_none_when_nothing_kept():
    df = pd.DataFrame({"s1": [1.0]}, index=["UNKNOWN"])
    assert finalize.restrict_to_clean_genes(df, {"TSPAN6"}) is None


def test_restrict_to_clean_genes_collapses_duplicate_symbols():
    df = pd.DataFrame({"s1": [1.0, 2.0]}, index=["TSPAN6", "TSPAN6"])
    result = finalize.restrict_to_clean_genes(df, {"TSPAN6"})
    assert result.loc["TSPAN6", "s1"] == 3.0


# --- find_local_expression_file -------------------------------------------------

def test_find_local_expression_file_found(tmp_path):
    expr_dir = tmp_path / "expression"
    expr_dir.mkdir()
    (expr_dir / "GSE1_counts.tsv.gz").write_bytes(b"x")
    qc = {"primary_expression_file": "/somewhere/else/GSE1_counts.tsv.gz"}
    result = finalize.find_local_expression_file(tmp_path, qc)
    assert result == expr_dir / "GSE1_counts.tsv.gz"


def test_find_local_expression_file_missing(tmp_path):
    qc = {"primary_expression_file": "GSE1_counts.tsv.gz"}
    assert finalize.find_local_expression_file(tmp_path, qc) is None


def test_find_local_expression_file_no_primary_file():
    assert finalize.find_local_expression_file(None, {}) is None


# --- finalize_cohort -----------------------------------------------------------

def _write_qc(cohort_dir, primary_expression_file, unit):
    cohort_dir.mkdir(parents=True, exist_ok=True)
    payload = {"primary_expression_file": str(primary_expression_file), "primary_expression_unit": unit}
    (cohort_dir / "expression_qc.json").write_text(json.dumps(payload))


def _write_matrix(cohort_dir, filename, df):
    expr_dir = cohort_dir / "expression"
    expr_dir.mkdir(parents=True, exist_ok=True)
    path = expr_dir / filename
    df.to_csv(path, sep="\t", compression="gzip")
    return path


_REF = make_reference(gene_to_symbol={"ENSG00000000003": "TSPAN6", "ENSG00000000005": "TNMD"})
_CLEAN_SYMBOLS = {"TSPAN6", "TNMD"}


def test_finalize_cohort_no_qc_json_skipped(tmp_path):
    row = finalize.finalize_cohort(tmp_path, _REF, _CLEAN_SYMBOLS)
    assert row["status"] == "skipped"
    assert "no expression_qc.json" in row["reason"]


def test_finalize_cohort_unknown_unit_guessed_as_count_when_integer(tmp_path):
    # unit "count" needs compute_tpm's per-gene length data to reach
    # "processed" -- unlike tpm/fpkm/rpkm, which skip that step entirely.
    ref_with_length = make_reference(
        gene_to_symbol={"ENSG00000000003": "TSPAN6", "ENSG00000000005": "TNMD"},
        gene_length={"ENSG00000000003": 2200, "ENSG00000000005": 1200},
    )
    df = pd.DataFrame({"s1": [10.0, 20.0], "s2": [15.0, 25.0]}, index=["TSPAN6", "TNMD"])
    df.index.name = "gene_id"
    path = _write_matrix(tmp_path, "unknown_unit.tsv.gz", df)
    _write_qc(tmp_path, path, "unknown")

    row = finalize.finalize_cohort(tmp_path, ref_with_length, _CLEAN_SYMBOLS)
    assert row["status"] == "processed"
    assert row["unit"] == "count"
    assert "quantification unit unknown" in row["reason"]
    assert "whole number" in row["reason"]


def test_finalize_cohort_unknown_unit_guessed_as_fpkm_for_gene_level_non_integer(tmp_path):
    """No filename/content unit hint, genuinely non-integer values, ENSG
    gene-level identifiers, no preserved .original raw file to check
    instead -- guessed FPKM rather than skipped."""
    df = pd.DataFrame({"s1": [8.5, 0.25], "s2": [9.5, 1.25]}, index=["ENSG00000000003", "ENSG00000000005"])
    df.index.name = "id"
    path = _write_matrix(tmp_path, "genes.tsv.gz", df)
    _write_qc(tmp_path, path, "unknown")

    row = finalize.finalize_cohort(tmp_path, _REF, _CLEAN_SYMBOLS)
    assert row["status"] == "processed"
    assert row["unit"] == "fpkm"
    assert "quantification unit unknown" in row["reason"]
    assert "gene-level" in row["reason"]


def test_finalize_cohort_unknown_unit_uses_original_raw_file_for_integer_check(tmp_path):
    """Real GSE230065 bug: the primary file was already log2-transformed by
    geotool.download at ingest time (values like 8.748 = log2(429 + 1)),
    so to_linear_scale's reconstruction (2**x - 1) is lossy -- large enough,
    at real magnitudes, to look non-integer even though the true underlying
    data (preserved as "<name>.original.tsv.gz" right next to the primary
    file, exactly geotool.download's own convention) is exact raw counts.
    Must guess "count", not "fpkm"."""
    log2_df = pd.DataFrame(
        {"s1": [8.748, 0.0], "s2": [9.333, 1.0]}, index=["ENSG00000000003", "ENSG00000000005"],
    )
    log2_df.index.name = "id"
    path = _write_matrix(tmp_path, "GSE230065_genes.tsv.gz", log2_df)

    raw_counts_df = pd.DataFrame(
        {"s1": [429, 0], "s2": [644, 1]}, index=["ENSG00000000003", "ENSG00000000005"],
    )
    raw_counts_df.index.name = "id"
    raw_counts_df.to_csv(tmp_path / "expression" / "GSE230065_genes.original.tsv.gz", sep="\t", compression="gzip")

    _write_qc(tmp_path, path, "unknown")

    # "count" is in _LENGTH_NORM_UNITS, so compute_tpm needs gene-length
    # data to reach "processed" (unlike fpkm, which skips that step).
    ref_with_length = make_reference(
        gene_to_symbol={"ENSG00000000003": "TSPAN6", "ENSG00000000005": "TNMD"},
        gene_length={"ENSG00000000003": 2200, "ENSG00000000005": 1200},
    )
    row = finalize.finalize_cohort(tmp_path, ref_with_length, _CLEAN_SYMBOLS)
    assert row["status"] == "processed"
    assert row["unit"] == "count"
    assert "whole number" in row["reason"]


def test_finalize_cohort_multi_file_skipped(tmp_path):
    _write_qc(tmp_path, None, None)
    row = finalize.finalize_cohort(tmp_path, _REF, _CLEAN_SYMBOLS)
    assert row["status"] == "skipped"
    assert "multi-file" in row["reason"]


def test_finalize_cohort_unrecoverable_ids_failed(tmp_path):
    df = pd.DataFrame({"s1": [10.0, 20.0], "s2": [15.0, 25.0]}, index=["XLOC_1", "XLOC_2"])
    df.index.name = "gene_id"
    path = _write_matrix(tmp_path, "counts.tsv.gz", df)
    _write_qc(tmp_path, path, "count")

    row = finalize.finalize_cohort(tmp_path, _REF, _CLEAN_SYMBOLS)
    assert row["status"] == "failed"
    assert row["reason"] == "gene names unrecoverable"


def test_finalize_cohort_processes_gene_symbol_tpm_matrix(tmp_path):
    df = pd.DataFrame(
        {"s1": [10.0, 20.0], "s2": [15.0, 25.0]},
        index=["TSPAN6", "TNMD"],
    )
    df.index.name = "gene_id"
    path = _write_matrix(tmp_path, "tpm.tsv.gz", df)
    _write_qc(tmp_path, path, "tpm")

    row = finalize.finalize_cohort(tmp_path, _REF, _CLEAN_SYMBOLS)
    assert row["status"] == "processed"
    assert row["n_genes"] == 2
    assert row["n_samples"] == 2

    out_path = tmp_path / "expression_final.tsv.gz"
    assert out_path.exists()
    written = pd.read_csv(out_path, sep="\t", index_col=0)
    assert set(written.index) == {"TSPAN6", "TNMD"}
    # log2(x+1) applied since the renormalized TPM values are >50 (linear scale)
    assert (written.to_numpy() < 50).all()


def test_finalize_cohort_local_file_missing_skipped(tmp_path):
    _write_qc(tmp_path, "does_not_exist.tsv.gz", "tpm")
    row = finalize.finalize_cohort(tmp_path, _REF, _CLEAN_SYMBOLS)
    assert row["status"] == "skipped"
    assert "not found locally" in row["reason"]


# --- sample id map integration ----------------------------------------------------

def test_write_sample_id_map_none_without_local_annotation(tmp_path):
    assert finalize.write_sample_id_map(tmp_path, ["DMSO_1"]) is None
    assert not (tmp_path / "sample_id_map.tsv").exists()


def test_write_sample_id_map_matches_and_writes_file(tmp_path):
    pd.DataFrame({
        "gsm_id": ["GSM1", "GSM2"], "title": ["DMSO_1", "DMSO_2"],
    }).to_csv(tmp_path / "annotation.tsv", sep="\t", index=False)

    result = finalize.write_sample_id_map(tmp_path, ["DMSO_1", "DMSO_2"])
    assert dict(zip(result["expression_id"], result["gsm_id"])) == {"DMSO_1": "GSM1", "DMSO_2": "GSM2"}

    out_path = tmp_path / "sample_id_map.tsv"
    assert out_path.exists()
    written = pd.read_csv(out_path, sep="\t")
    assert set(written["expression_id"]) == {"DMSO_1", "DMSO_2"}


def test_finalize_cohort_writes_sample_id_map_alongside_expression_final(tmp_path):
    pd.DataFrame({
        "gsm_id": ["GSM1", "GSM2"], "title": ["TSPAN6", "TNMD"],
    }).to_csv(tmp_path / "annotation.tsv", sep="\t", index=False)
    df = pd.DataFrame({"TSPAN6": [10.0, 20.0], "TNMD": [15.0, 25.0]}, index=["TSPAN6", "TNMD"])
    df.index.name = "gene_id"
    path = _write_matrix(tmp_path, "tpm.tsv.gz", df)
    _write_qc(tmp_path, path, "tpm")

    row = finalize.finalize_cohort(tmp_path, _REF, _CLEAN_SYMBOLS)
    assert row["status"] == "processed"
    assert row["n_samples_matched_to_gsm"] == 2
    assert (tmp_path / "sample_id_map.tsv").exists()


def test_finalize_cohort_writes_sample_id_map_even_when_finalization_later_skips(tmp_path):
    """The id map is diagnostic information independent of whether the
    matrix itself makes it through gene-symbol conversion / clean-gene-set
    filtering -- write it as soon as the sample columns are known."""
    pd.DataFrame({
        "gsm_id": ["GSM1", "GSM2"], "title": ["col_a", "col_b"],
    }).to_csv(tmp_path / "annotation.tsv", sep="\t", index=False)
    df = pd.DataFrame({"col_a": [10.0, 20.0], "col_b": [15.0, 25.0]}, index=["XLOC_1", "XLOC_2"])
    df.index.name = "gene_id"
    path = _write_matrix(tmp_path, "counts.tsv.gz", df)
    _write_qc(tmp_path, path, "count")

    row = finalize.finalize_cohort(tmp_path, _REF, _CLEAN_SYMBOLS)
    assert row["status"] == "failed"  # unrecoverable gene identifiers
    assert (tmp_path / "sample_id_map.tsv").exists()
    id_map = pd.read_csv(tmp_path / "sample_id_map.tsv", sep="\t")
    assert dict(zip(id_map["expression_id"], id_map["gsm_id"])) == {"col_a": "GSM1", "col_b": "GSM2"}


# --- merge_sample_id_map_into_series_annotation ----------------------------------

def _write_series_annotation(series_dir, gse_id, df):
    out_dir = series_dir / gse_id
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "annotation.tsv", sep="\t", index=False)
    return out_dir / "annotation.tsv"


def test_merge_sample_id_map_noop_without_series_annotation(tmp_path):
    id_map = pd.DataFrame({
        "expression_id": ["DMSO_1"], "gsm_id": ["GSM1"], "match_method": ["exact"], "confidence": [0.95],
    })
    # No data/series/GSE1/annotation.tsv at all -- must not raise, must not create one.
    finalize.merge_sample_id_map_into_series_annotation("GSE1", id_map, series_dir=tmp_path)
    assert not (tmp_path / "GSE1").exists()


def test_merge_sample_id_map_noop_without_gsm_id_column(tmp_path):
    path = _write_series_annotation(tmp_path, "GSE1", pd.DataFrame({"title": ["a"]}))
    id_map = pd.DataFrame({
        "expression_id": ["DMSO_1"], "gsm_id": ["GSM1"], "match_method": ["exact"], "confidence": [0.95],
    })
    finalize.merge_sample_id_map_into_series_annotation("GSE1", id_map, series_dir=tmp_path)
    written = pd.read_csv(path, sep="\t")
    assert "expression_id" not in written.columns


def test_merge_sample_id_map_adds_columns_joined_on_gsm_id(tmp_path):
    path = _write_series_annotation(
        tmp_path, "GSE1", pd.DataFrame({"gsm_id": ["GSM1", "GSM2"], "title": ["a", "b"]}),
    )
    id_map = pd.DataFrame({
        "expression_id": ["DMSO_1", "DMSO_2"], "gsm_id": ["GSM1", "GSM2"],
        "match_method": ["exact", "substring"], "confidence": [0.95, 0.75],
    })
    finalize.merge_sample_id_map_into_series_annotation("GSE1", id_map, series_dir=tmp_path)

    written = pd.read_csv(path, sep="\t")
    assert list(written["expression_id"]) == ["DMSO_1", "DMSO_2"]
    assert list(written["sample_id_match_method"]) == ["exact", "substring"]
    assert list(written["sample_id_match_confidence"]) == [0.95, 0.75]
    assert list(written["title"]) == ["a", "b"]  # original columns preserved


def test_merge_sample_id_map_nan_for_unmatched_sample(tmp_path):
    path = _write_series_annotation(
        tmp_path, "GSE1", pd.DataFrame({"gsm_id": ["GSM1", "GSM2"], "title": ["a", "b"]}),
    )
    # Only GSM1 resolved -- GSM2's expression column was ambiguous/unmatched.
    id_map = pd.DataFrame({
        "expression_id": ["DMSO_1", "colX"], "gsm_id": ["GSM1", None],
        "match_method": ["exact", "unmatched"], "confidence": [0.95, 0.0],
    })
    finalize.merge_sample_id_map_into_series_annotation("GSE1", id_map, series_dir=tmp_path)

    written = pd.read_csv(path, sep="\t")
    gsm2_row = written[written["gsm_id"] == "GSM2"].iloc[0]
    assert pd.isna(gsm2_row["expression_id"])
    assert pd.isna(gsm2_row["sample_id_match_method"])
    assert len(written) == 2  # the unmatched id_map row doesn't create a spurious extra row


def test_merge_sample_id_map_is_idempotent(tmp_path):
    path = _write_series_annotation(
        tmp_path, "GSE1", pd.DataFrame({"gsm_id": ["GSM1"], "title": ["a"]}),
    )
    id_map = pd.DataFrame({
        "expression_id": ["DMSO_1"], "gsm_id": ["GSM1"], "match_method": ["exact"], "confidence": [0.95],
    })
    finalize.merge_sample_id_map_into_series_annotation("GSE1", id_map, series_dir=tmp_path)
    finalize.merge_sample_id_map_into_series_annotation("GSE1", id_map, series_dir=tmp_path)

    written = pd.read_csv(path, sep="\t")
    assert len(written) == 1
    assert list(written.columns).count("expression_id") == 1


def test_write_sample_id_map_merges_into_series_annotation(tmp_path):
    """The collection-root copy (cohort_dir) and the canonical series
    annotation are different directories in real usage -- confirm both get
    consulted/written correctly, not just the collection-root copy."""
    cohort_dir = tmp_path / "collection" / "GSE1"
    cohort_dir.mkdir(parents=True)
    pd.DataFrame({"gsm_id": ["GSM1", "GSM2"], "title": ["DMSO_1", "DMSO_2"]}).to_csv(
        cohort_dir / "annotation.tsv", sep="\t", index=False
    )
    series_dir = tmp_path / "series"
    series_path = _write_series_annotation(
        series_dir, "GSE1", pd.DataFrame({"gsm_id": ["GSM1", "GSM2"], "title": ["DMSO_1", "DMSO_2"]}),
    )

    finalize.write_sample_id_map(cohort_dir, ["DMSO_1", "DMSO_2"], series_dir=series_dir)

    written = pd.read_csv(series_path, sep="\t")
    assert list(written["expression_id"]) == ["DMSO_1", "DMSO_2"]


def test_finalize_cohort_merges_sample_id_map_into_series_annotation(tmp_path):
    cohort_dir = tmp_path / "collection" / "GSE1"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"gsm_id": ["GSM1", "GSM2"], "title": ["TSPAN6", "TNMD"]}).to_csv(
        cohort_dir / "annotation.tsv", sep="\t", index=False,
    )
    df = pd.DataFrame({"TSPAN6": [10.0, 20.0], "TNMD": [15.0, 25.0]}, index=["TSPAN6", "TNMD"])
    df.index.name = "gene_id"
    path = _write_matrix(cohort_dir, "tpm.tsv.gz", df)
    _write_qc(cohort_dir, path, "tpm")

    series_dir = tmp_path / "series"
    _write_series_annotation(
        series_dir, "GSE1", pd.DataFrame({"gsm_id": ["GSM1", "GSM2"], "title": ["TSPAN6", "TNMD"]}),
    )

    row = finalize.finalize_cohort(cohort_dir, _REF, _CLEAN_SYMBOLS, series_dir=series_dir)
    assert row["status"] == "processed"

    written = pd.read_csv(series_dir / "GSE1" / "annotation.tsv", sep="\t")
    assert set(written["expression_id"]) == {"TSPAN6", "TNMD"}
    assert written["sample_id_match_confidence"].notna().all()


# --- build_final_matrices -------------------------------------------------------

def _write_clean_genes_reference(references_dir, version, symbols):
    base = references_dir / f"gencode{version}"
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "transcript_id": [f"ENST{i:011d}" for i in range(len(symbols))],
        "gene_symbol": symbols,
    }).to_csv(
        base / f"clean_transcript_gene_symbol_v{version}.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    pd.DataFrame({"ID": ["ENSG00000000003"], "Gene": ["TSPAN6"]}).to_csv(
        base / f"ensg2hugo_gencode_v{version}.tsv.gz", sep="\t", index=False, compression="gzip"
    )


def test_build_final_matrices_end_to_end(tmp_path):
    references_dir = tmp_path / "references"
    _write_clean_genes_reference(references_dir, "50", ["TSPAN6", "TNMD"])

    collection_root = tmp_path / "collection"
    cohort_dir = collection_root / "GSE1"
    df = pd.DataFrame({"s1": [10.0, 20.0]}, index=["TSPAN6", "TNMD"])
    df.index.name = "gene_id"
    path = _write_matrix(cohort_dir, "tpm.tsv.gz", df)
    _write_qc(cohort_dir, path, "tpm")

    report_df = finalize.build_final_matrices([collection_root], gencode_version="50", references_dir=references_dir)
    assert list(report_df["status"]) == ["processed"]
    assert (cohort_dir / "expression_final.tsv.gz").exists()


def test_build_final_matrices_multiple_roots(tmp_path):
    references_dir = tmp_path / "references"
    _write_clean_genes_reference(references_dir, "50", ["TSPAN6"])

    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    for root in (root_a, root_b):
        cohort_dir = root / "GSE1"
        cohort_dir.mkdir(parents=True)
        # No expression_qc.json -- both should be reported as skipped.

    report_df = finalize.build_final_matrices([root_a, root_b], gencode_version="50", references_dir=references_dir)
    assert len(report_df) == 2
    assert (report_df["status"] == "skipped").all()
