from geotool import cli


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
