import json

import pandas as pd
import pytest

from geotool import cohort_report


def _write_series(series_dir, gse_id, *, title="", organism="Homo sapiens", summary="", platform_details=None):
    out_dir = series_dir / gse_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "gse_id": gse_id, "title": title, "organism": organism, "summary": summary,
        "platform_details": json.dumps(platform_details or []),
    }]).to_csv(out_dir / "series.tsv", sep="\t", index=False)
    return out_dir


def _write_annotation(out_dir, n_samples=2, expression_status="ok"):
    pd.DataFrame([
        {"gsm_id": f"GSM{i}", "gse_id": out_dir.name, "expression_status": expression_status}
        for i in range(n_samples)
    ]).to_csv(out_dir / "annotation.tsv", sep="\t", index=False)


def _write_superseries_marker(out_dir, subseries, orphaned_gsm_ids=None, orphaned_supplementary_files=None):
    payload = {
        "subseries": subseries,
        "orphaned_gsm_ids": orphaned_gsm_ids or [],
        "orphaned_supplementary_files": orphaned_supplementary_files or [],
    }
    with open(out_dir / "superseries.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)


# --- resolve_effective_cohort_ids -------------------------------------------

def test_resolve_effective_cohort_ids_no_superseries(tmp_path):
    _write_series(tmp_path, "GSE1")
    all_ids, parent_of = cohort_report.resolve_effective_cohort_ids(["GSE1"], series_dir=tmp_path)
    assert all_ids == ["GSE1"]
    assert parent_of == {}


def test_resolve_effective_cohort_ids_expands_subseries(tmp_path):
    parent_dir = _write_series(tmp_path, "GSE_PARENT")
    _write_superseries_marker(parent_dir, ["GSE_SUB1", "GSE_SUB2"])
    _write_series(tmp_path, "GSE_SUB1")
    _write_series(tmp_path, "GSE_SUB2")

    all_ids, parent_of = cohort_report.resolve_effective_cohort_ids(["GSE_PARENT"], series_dir=tmp_path)
    assert all_ids == ["GSE_PARENT", "GSE_SUB1", "GSE_SUB2"]
    assert parent_of == {"GSE_SUB1": "GSE_PARENT", "GSE_SUB2": "GSE_PARENT"}


def test_resolve_effective_cohort_ids_recurses_to_fixed_point(tmp_path):
    top_dir = _write_series(tmp_path, "GSE_TOP")
    _write_superseries_marker(top_dir, ["GSE_MID"])
    mid_dir = _write_series(tmp_path, "GSE_MID")
    _write_superseries_marker(mid_dir, ["GSE_LEAF"])
    _write_series(tmp_path, "GSE_LEAF")

    all_ids, parent_of = cohort_report.resolve_effective_cohort_ids(["GSE_TOP"], series_dir=tmp_path)
    assert all_ids == ["GSE_TOP", "GSE_MID", "GSE_LEAF"]
    assert parent_of == {"GSE_MID": "GSE_TOP", "GSE_LEAF": "GSE_MID"}


def test_resolve_effective_cohort_ids_does_not_duplicate_already_requested_subseries(tmp_path):
    parent_dir = _write_series(tmp_path, "GSE_PARENT")
    _write_superseries_marker(parent_dir, ["GSE_SUB1"])
    _write_series(tmp_path, "GSE_SUB1")

    all_ids, parent_of = cohort_report.resolve_effective_cohort_ids(
        ["GSE_PARENT", "GSE_SUB1"], series_dir=tmp_path
    )
    assert all_ids == ["GSE_PARENT", "GSE_SUB1"]
    assert parent_of == {"GSE_SUB1": "GSE_PARENT"}


# --- cohort_report_row -------------------------------------------------------

def test_cohort_report_row_not_downloaded(tmp_path):
    row = cohort_report.cohort_report_row("GSE_MISSING", tmp_path, {}, {"GSE_MISSING"})
    assert row["downloaded"] is False
    assert row["readiness"] == "not_ready"
    assert "not downloaded yet" in row["reason_if_not_ready"]


def test_cohort_report_row_ready_from_expression_status(tmp_path):
    out_dir = _write_series(tmp_path, "GSE1", title="A study", platform_details=[{"assay_type": "bulk_rnaseq", "gpl_id": "GPL1"}])
    _write_annotation(out_dir, n_samples=3, expression_status="ok")

    row = cohort_report.cohort_report_row("GSE1", tmp_path, {}, {"GSE1"})
    assert row["downloaded"] is True
    assert row["title"] == "A study"
    assert row["assay_type"] == "bulk_rnaseq"
    assert row["platforms"] == "GPL1"
    assert row["n_samples"] == 3
    assert row["readiness"] == "ready"
    assert row["reason_if_not_ready"] == ""


def test_cohort_report_row_not_ready_from_expression_status(tmp_path):
    out_dir = _write_series(tmp_path, "GSE1")
    _write_annotation(out_dir, expression_status="no_expression_matrix")

    row = cohort_report.cohort_report_row("GSE1", tmp_path, {}, {"GSE1"})
    assert row["readiness"] == "not_ready"
    assert row["reason_if_not_ready"] == "no_expression_matrix"


def test_cohort_report_row_superseries_with_no_annotation(tmp_path):
    out_dir = _write_series(tmp_path, "GSE_PARENT")
    _write_superseries_marker(out_dir, ["GSE_SUB"])

    row = cohort_report.cohort_report_row("GSE_PARENT", tmp_path, {}, {"GSE_PARENT"})
    assert row["readiness"] == "not_ready"
    assert "SuperSeries record with no expression data of its own" in row["reason_if_not_ready"]


def test_cohort_report_row_superseries_orphan_note(tmp_path):
    out_dir = _write_series(tmp_path, "GSE_PARENT")
    _write_superseries_marker(out_dir, ["GSE_SUB"], orphaned_gsm_ids=["GSM99"])

    row = cohort_report.cohort_report_row("GSE_PARENT", tmp_path, {}, {"GSE_PARENT"})
    assert "orphaned" in row["reason_if_not_ready"].lower() or "SuperSeries carries data" in row["reason_if_not_ready"]


def test_cohort_report_row_collection_root_ready(tmp_path):
    out_dir = _write_series(tmp_path, "GSE1")
    _write_annotation(out_dir, expression_status="ok")

    collection_root = tmp_path / "collection"
    cohort_dir = collection_root / "GSE1"
    cohort_dir.mkdir(parents=True)
    (cohort_dir / "expression_final.tsv.gz").write_bytes(b"x")

    row = cohort_report.cohort_report_row("GSE1", tmp_path, {}, {"GSE1"}, collection_root=collection_root)
    assert row["readiness"] == "ready"
    assert row["expression_file"] == "GSE1/expression_final.tsv.gz"


def test_cohort_report_row_collection_root_missing_file(tmp_path):
    out_dir = _write_series(tmp_path, "GSE1")
    _write_annotation(out_dir, expression_status="ok")

    collection_root = tmp_path / "collection"
    collection_root.mkdir()

    row = cohort_report.cohort_report_row("GSE1", tmp_path, {}, {"GSE1"}, collection_root=collection_root)
    assert row["readiness"] == "not_ready"
    assert "expression_final.tsv.gz" in row["reason_if_not_ready"]


# --- sample_id_match_status ---------------------------------------------------

def test_cohort_report_row_sample_id_match_blank_without_collection_root(tmp_path):
    _write_series(tmp_path, "GSE1")
    row = cohort_report.cohort_report_row("GSE1", tmp_path, {}, {"GSE1"})
    assert row["sample_id_match_status"] == ""
    assert row["sample_id_match_detail"] == ""


def test_cohort_report_row_sample_id_match_not_available_without_map_file(tmp_path):
    _write_series(tmp_path, "GSE1")
    collection_root = tmp_path / "collection"
    (collection_root / "GSE1").mkdir(parents=True)

    row = cohort_report.cohort_report_row("GSE1", tmp_path, {}, {"GSE1"}, collection_root=collection_root)
    assert row["sample_id_match_status"] == "not_available"
    assert "sample_id_map.tsv" in row["sample_id_match_detail"]


def test_cohort_report_row_sample_id_match_matched(tmp_path):
    _write_series(tmp_path, "GSE1")
    collection_root = tmp_path / "collection"
    cohort_dir = collection_root / "GSE1"
    cohort_dir.mkdir(parents=True)
    pd.DataFrame({
        "expression_id": ["DMSO_1", "DMSO_2"], "gsm_id": ["GSM1", "GSM2"],
        "match_method": ["exact", "exact"], "confidence": [0.95, 0.95],
    }).to_csv(cohort_dir / "sample_id_map.tsv", sep="\t", index=False)

    row = cohort_report.cohort_report_row("GSE1", tmp_path, {}, {"GSE1"}, collection_root=collection_root)
    assert row["sample_id_match_status"] == "matched"
    assert row["sample_id_match_detail"] == "2/2 matched"


def test_cohort_report_row_sample_id_match_needs_review(tmp_path):
    _write_series(tmp_path, "GSE1")
    collection_root = tmp_path / "collection"
    cohort_dir = collection_root / "GSE1"
    cohort_dir.mkdir(parents=True)
    pd.DataFrame({
        "expression_id": ["DMSO_1", "DMSO_2", "DMSO_3"],
        "gsm_id": ["GSM1", None, None],
        "match_method": ["exact", "unmatched", "unmatched"],
        "confidence": [0.95, 0.0, 0.0],
    }).to_csv(cohort_dir / "sample_id_map.tsv", sep="\t", index=False)

    row = cohort_report.cohort_report_row("GSE1", tmp_path, {}, {"GSE1"}, collection_root=collection_root)
    assert row["sample_id_match_status"] == "needs_review"
    assert row["sample_id_match_detail"].startswith("1/3 matched")


def test_cohort_report_row_in_requested_list_flag(tmp_path):
    _write_series(tmp_path, "GSE1")
    row_requested = cohort_report.cohort_report_row("GSE1", tmp_path, {}, {"GSE1"})
    row_not_requested = cohort_report.cohort_report_row("GSE1", tmp_path, {}, {"GSE_OTHER"})
    assert row_requested["in_requested_list"] is True
    assert row_not_requested["in_requested_list"] is False


# --- build_cohort_report ------------------------------------------------------

def test_build_cohort_report_end_to_end(tmp_path):
    parent_dir = _write_series(tmp_path, "GSE_PARENT")
    _write_superseries_marker(parent_dir, ["GSE_SUB"])
    sub_dir = _write_series(tmp_path, "GSE_SUB")
    _write_annotation(sub_dir, expression_status="ok")

    df = cohort_report.build_cohort_report(["GSE_PARENT"], series_dir=tmp_path)
    assert set(df["gse_id"]) == {"GSE_PARENT", "GSE_SUB"}
    sub_row = df[df["gse_id"] == "GSE_SUB"].iloc[0]
    assert sub_row["parent_series"] == "GSE_PARENT"
    assert sub_row["in_requested_list"] == False
    assert sub_row["readiness"] == "ready"


def test_build_cohort_report_empty_ids_returns_empty_frame(tmp_path):
    df = cohort_report.build_cohort_report([], series_dir=tmp_path)
    assert df.empty
