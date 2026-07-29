import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from geotool import renormalize


def test_chip_package_for_known_platform():
    family, package = renormalize.chip_package_for("GPL570")
    assert family == "3prime"
    assert package == "hgu133plus2"


def test_chip_package_for_unknown_platform_raises():
    with pytest.raises(renormalize.RmaUnavailable):
        renormalize.chip_package_for("GPL999999")


def test_run_rma_raises_when_no_cel_files():
    with pytest.raises(renormalize.RmaUnavailable):
        renormalize.run_rma({}, "GPL570")


def test_run_rma_raises_when_rscript_missing(monkeypatch):
    monkeypatch.setattr(renormalize.shutil, "which", lambda name: None)
    with pytest.raises(renormalize.RmaUnavailable, match="Rscript"):
        renormalize.run_rma({"GSM1": Path("GSM1.CEL")}, "GPL570")


def test_run_rma_raises_for_unknown_platform(monkeypatch):
    monkeypatch.setattr(renormalize.shutil, "which", lambda name: "/usr/bin/Rscript")
    with pytest.raises(renormalize.RmaUnavailable, match="no known Bioconductor"):
        renormalize.run_rma({"GSM1": Path("GSM1.CEL")}, "GPL999999")


def test_run_rma_raises_when_rscript_fails(monkeypatch, tmp_path):
    cel = tmp_path / "GSM1.CEL"
    cel.write_bytes(b"fake cel data")
    monkeypatch.setattr(renormalize.shutil, "which", lambda name: "/usr/bin/Rscript")

    def fake_run(cmd, capture_output, text, timeout):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Error: package 'hgu133plus2' not found")

    monkeypatch.setattr(renormalize.subprocess, "run", fake_run)

    with pytest.raises(renormalize.RmaUnavailable, match="package 'hgu133plus2' not found"):
        renormalize.run_rma({"GSM1": cel}, "GPL570")


def _fake_run_writes_csv(csv_text: str):
    def fake_run(cmd, capture_output, text, timeout):
        script_path = Path(cmd[1])
        script_text = script_path.read_text()
        out_csv = re.search(r'file = "([^"]+)"', script_text)
        assert out_csv, f"could not find output path in generated R script:\n{script_text}"
        out_csv = out_csv.group(1)
        Path(out_csv).write_text(csv_text)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return fake_run


def test_run_rma_assigns_gsm_id_columns_positionally(monkeypatch, tmp_path):
    """R's own column names (from CEL filenames, sanitized via make.names) are
    unreliable to parse back -- columns must map to gsm_ids by position, in
    the same order the CEL paths were passed to R."""
    cel_files = {
        "GSM1": tmp_path / "GSM1_raw.CEL.gz",
        "GSM2": tmp_path / "GSM2_raw.CEL.gz",
    }
    for path in cel_files.values():
        path.write_bytes(b"fake")

    monkeypatch.setattr(renormalize.shutil, "which", lambda name: "/usr/bin/Rscript")
    monkeypatch.setattr(
        renormalize.subprocess,
        "run",
        _fake_run_writes_csv("probe_id,GSM1_raw.CEL.gz,GSM2_raw.CEL.gz\n1007_s_at,5.1,5.4\n1053_at,6.2,6.6\n"),
    )

    matrix = renormalize.run_rma(cel_files, "GPL570")

    assert list(matrix.columns) == ["GSM1", "GSM2"]
    assert matrix.loc["1007_s_at", "GSM1"] == 5.1
    assert matrix.loc["1053_at", "GSM2"] == 6.6


def test_run_rma_raises_when_output_column_count_mismatches(monkeypatch, tmp_path):
    cel_files = {"GSM1": tmp_path / "a.CEL", "GSM2": tmp_path / "b.CEL"}
    for path in cel_files.values():
        path.write_bytes(b"fake")

    monkeypatch.setattr(renormalize.shutil, "which", lambda name: "/usr/bin/Rscript")
    monkeypatch.setattr(
        renormalize.subprocess, "run", _fake_run_writes_csv("probe_id,only_one_col\n1007_s_at,5.1\n")
    )

    with pytest.raises(renormalize.RmaUnavailable, match="expected 2"):
        renormalize.run_rma(cel_files, "GPL570")
