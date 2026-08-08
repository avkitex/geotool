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


def test_finalize_cohort_unknown_unit_skipped(tmp_path):
    _write_qc(tmp_path, "x.tsv.gz", "unknown")
    row = finalize.finalize_cohort(tmp_path, _REF, _CLEAN_SYMBOLS)
    assert row["status"] == "skipped"
    assert "unit unknown" in row["reason"]


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
