from click.testing import CliRunner

from geotool import cli, download as download_mod


class FakeStream:
    def __init__(self, encoding):
        self.encoding = encoding
        self.reconfigure_calls = []

    def reconfigure(self, **kwargs):
        self.reconfigure_calls.append(kwargs)


def test_ensure_utf8_streams_reconfigures_stdout_and_stderr(monkeypatch):
    """Regression test: a real `geotool download` run on GSE38516 (title
    contains 'IFN-γ') raised UnicodeEncodeError from a plain print() near
    the end of an otherwise fully successful run, because this Windows
    terminal's stdout defaulted to cp1252. The CLI must reconfigure stdout/
    stderr to UTF-8 (with replacement rather than raising) so GEO metadata's
    non-ASCII characters never crash a successful command."""
    fake_stdout = FakeStream("cp1252")
    fake_stderr = FakeStream("cp1252")
    monkeypatch.setattr(cli.sys, "stdout", fake_stdout)
    monkeypatch.setattr(cli.sys, "stderr", fake_stderr)

    cli._ensure_utf8_streams()

    assert fake_stdout.reconfigure_calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert fake_stderr.reconfigure_calls == [{"encoding": "utf-8", "errors": "replace"}]


def test_ensure_utf8_streams_skips_streams_without_reconfigure(monkeypatch):
    """E.g. when stdout has been replaced with a plain io.BytesIO or similar
    in some embedding context -- must not raise just because .reconfigure
    isn't available."""
    class NoReconfigure:
        encoding = "utf-8"

    monkeypatch.setattr(cli.sys, "stdout", NoReconfigure())
    monkeypatch.setattr(cli.sys, "stderr", NoReconfigure())

    cli._ensure_utf8_streams()  # must not raise


def _fake_result(gse_id: str) -> dict:
    return {
        "gse_id": gse_id, "assay_types": ["microarray"], "expression_path": None,
        "annotation_path": f"data/series/{gse_id}/annotation.tsv",
    }


def test_download_command_calls_download_cohort_directly_for_a_plain_series(monkeypatch):
    monkeypatch.setattr(download_mod, "resolve_download_targets", lambda gse_id, force=False: [gse_id])
    calls = []
    monkeypatch.setattr(
        download_mod, "download_cohort",
        lambda gse_id, rma=False, force=False: (calls.append(gse_id), _fake_result(gse_id))[1],
    )

    result = CliRunner().invoke(cli.main, ["download", "GSE_LEAF"])

    assert result.exit_code == 0
    assert calls == ["GSE_LEAF"]
    assert "SuperSeries" not in result.output


def test_download_command_expands_and_downloads_every_subseries(monkeypatch):
    monkeypatch.setattr(
        download_mod, "resolve_download_targets", lambda gse_id, force=False: ["GSE101", "GSE102"]
    )
    calls = []
    monkeypatch.setattr(
        download_mod, "download_cohort",
        lambda gse_id, rma=False, force=False: (calls.append(gse_id), _fake_result(gse_id))[1],
    )

    result = CliRunner().invoke(cli.main, ["download", "GSE100"])

    assert result.exit_code == 0
    assert calls == ["GSE101", "GSE102"]
    assert "SuperSeries with 2 sub-series -- downloading each: GSE101, GSE102" in result.output


def test_download_command_reports_failure_per_target_and_continues(monkeypatch):
    monkeypatch.setattr(
        download_mod, "resolve_download_targets", lambda gse_id, force=False: ["GSE101", "GSE102"]
    )

    def fake_download_cohort(gse_id, rma=False, force=False):
        if gse_id == "GSE101":
            raise download_mod.UnsupportedCohortError("cna array, not a supported mRNA-expression platform")
        return _fake_result(gse_id)

    monkeypatch.setattr(download_mod, "download_cohort", fake_download_cohort)

    result = CliRunner().invoke(cli.main, ["download", "GSE100"])

    assert result.exit_code == 0
    assert "GSE101:" in result.output
    assert "FAILED: cna array" in result.output
    assert "GSE102:" in result.output
