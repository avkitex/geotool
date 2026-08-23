import pandas as pd
import pytest

from geotool import microarray_finalize as finalize

_CLEAN_SYMBOLS = {"TSPAN6", "TNMD"}


def _write_annotation(series_dir, gse_id, expression_status="ok", n=2):
    out_dir = series_dir / gse_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "gsm_id": [f"GSM{i}" for i in range(1, n + 1)],
        "expression_status": [expression_status] * n,
    }).to_csv(out_dir / "annotation.tsv", sep="\t", index=False)


def _write_expression(cohort_dir, filename="expression.tsv.gz", index=("TSPAN6", "TNMD", "NOT_CLEAN"), columns=None):
    columns = columns or ["GSM1", "GSM2"]
    cohort_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {col: [float(i + 1) for i in range(len(index))] for col in columns},
        index=list(index),
    )
    df.index.name = "gene_symbol"
    df.to_csv(cohort_dir / filename, sep="\t", compression="gzip")
    return df


# --- _source_matrix_path -------------------------------------------------

def test_source_matrix_path_prefers_channel_signal_over_plain_expression(tmp_path):
    (tmp_path / "expression.tsv.gz").write_bytes(b"")
    (tmp_path / "channel_signal_expression.tsv.gz").write_bytes(b"")
    path, name = finalize._source_matrix_path(tmp_path)
    assert name == "channel_signal_expression.tsv.gz"
    assert path == tmp_path / "channel_signal_expression.tsv.gz"


def test_source_matrix_path_falls_back_to_plain_expression(tmp_path):
    (tmp_path / "expression.tsv.gz").write_bytes(b"")
    path, name = finalize._source_matrix_path(tmp_path)
    assert name == "expression.tsv.gz"


def test_source_matrix_path_none_when_neither_exists(tmp_path):
    path, name = finalize._source_matrix_path(tmp_path)
    assert path is None
    assert name == ""


# --- finalize_cohort -------------------------------------------------------

def test_finalize_cohort_no_annotation_skipped(tmp_path):
    cohort_dir = tmp_path / "cohorts" / "GSE1"
    _write_expression(cohort_dir)
    row = finalize.finalize_cohort(cohort_dir, _CLEAN_SYMBOLS, series_dir=tmp_path / "series")
    assert row["status"] == "skipped"
    assert "not 'ok'" in row["reason"]


def test_finalize_cohort_two_channel_unresolved_skipped(tmp_path):
    series_dir = tmp_path / "series"
    cohort_dir = tmp_path / "cohorts" / "GSE1"
    _write_annotation(series_dir, "GSE1", expression_status="two_channel_signal_unresolved")
    _write_expression(cohort_dir)  # only the raw ratio, no channel_signal_expression.tsv.gz

    row = finalize.finalize_cohort(cohort_dir, _CLEAN_SYMBOLS, series_dir=series_dir)
    assert row["status"] == "skipped"
    assert "two_channel_signal_unresolved" in row["reason"]


def test_finalize_cohort_no_expression_file_skipped(tmp_path):
    series_dir = tmp_path / "series"
    cohort_dir = tmp_path / "cohorts" / "GSE1"
    cohort_dir.mkdir(parents=True)
    _write_annotation(series_dir, "GSE1")

    row = finalize.finalize_cohort(cohort_dir, _CLEAN_SYMBOLS, series_dir=series_dir)
    assert row["status"] == "skipped"
    assert "no expression.tsv.gz" in row["reason"]


def test_finalize_cohort_restricts_to_clean_genes(tmp_path):
    series_dir = tmp_path / "series"
    cohort_dir = tmp_path / "cohorts" / "GSE1"
    _write_annotation(series_dir, "GSE1")
    _write_expression(cohort_dir)  # TSPAN6, TNMD, NOT_CLEAN

    row = finalize.finalize_cohort(cohort_dir, _CLEAN_SYMBOLS, series_dir=series_dir)
    assert row["status"] == "processed"
    assert row["source_file"] == "expression.tsv.gz"
    assert row["n_genes"] == 2
    assert row["n_samples"] == 2

    out_path = cohort_dir / "expression_final.tsv.gz"
    assert out_path.exists()
    written = pd.read_csv(out_path, sep="\t", index_col=0)
    assert set(written.index) == {"TSPAN6", "TNMD"}


def test_finalize_cohort_uses_channel_signal_expression_when_present(tmp_path):
    series_dir = tmp_path / "series"
    cohort_dir = tmp_path / "cohorts" / "GSE1"
    _write_annotation(series_dir, "GSE1")
    _write_expression(cohort_dir, filename="expression.tsv.gz")
    _write_expression(cohort_dir, filename="channel_signal_expression.tsv.gz", index=("TSPAN6",))

    row = finalize.finalize_cohort(cohort_dir, _CLEAN_SYMBOLS, series_dir=series_dir)
    assert row["status"] == "processed"
    assert row["source_file"] == "channel_signal_expression.tsv.gz"
    assert row["n_genes"] == 1


def test_finalize_cohort_no_clean_genes_skipped(tmp_path):
    series_dir = tmp_path / "series"
    cohort_dir = tmp_path / "cohorts" / "GSE1"
    _write_annotation(series_dir, "GSE1")
    _write_expression(cohort_dir, index=("NOT_CLEAN_1", "NOT_CLEAN_2"))

    row = finalize.finalize_cohort(cohort_dir, _CLEAN_SYMBOLS, series_dir=series_dir)
    assert row["status"] == "skipped"
    assert "clean reference gene set" in row["reason"]


# --- build_final_matrices ---------------------------------------------------

def test_build_final_matrices_skips_rnaseq_cohort_with_no_bare_expression_file(tmp_path, monkeypatch):
    """An RNA-seq cohort's raw matrix lives under <cohort>/expression/, not
    directly at <cohort>/expression.tsv.gz -- build_final_matrices must not
    mistake one for the other."""
    references_dir = tmp_path / "references"
    clean_path = references_dir / "gencode50" / "clean_transcript_gene_symbol_v50.tsv.gz"
    clean_path.parent.mkdir(parents=True)
    pd.DataFrame({"transcript_id": ["ENST1"], "gene_symbol": ["TSPAN6"]}).to_csv(clean_path, sep="\t", index=False)

    root = tmp_path / "cohorts"
    rnaseq_dir = root / "GSE_RNASEQ"
    (rnaseq_dir / "expression").mkdir(parents=True)
    _write_annotation(tmp_path / "series", "GSE_RNASEQ")

    report_df = finalize.build_final_matrices([root], references_dir=references_dir, series_dir=tmp_path / "series")
    row = report_df[report_df["gse_id"] == "GSE_RNASEQ"].iloc[0]
    assert row["status"] == "skipped"
    assert "no expression.tsv.gz" in row["reason"]
