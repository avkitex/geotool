from geotool import geo_fetch


class FakeGSE:
    def __init__(self, relation=None):
        self.metadata = {"relation": relation or []}


def test_resolve_leaf_series_ids_returns_self_when_not_a_superseries(monkeypatch):
    monkeypatch.setattr(geo_fetch, "fetch_series", lambda gse_id: FakeGSE())
    assert geo_fetch.resolve_leaf_series_ids("GSE1") == ["GSE1"]


def test_resolve_leaf_series_ids_ignores_subseries_of_and_bioproject_relations(monkeypatch):
    """A leaf series fetched directly shows 'SubSeries of: <parent>' (pointing up) and
    a BioProject link, neither of which should be mistaken for children."""
    gse = FakeGSE(["SubSeries of: GSE100", "BioProject: https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1"])
    monkeypatch.setattr(geo_fetch, "fetch_series", lambda gse_id: gse)
    assert geo_fetch.resolve_leaf_series_ids("GSE101") == ["GSE101"]


def test_resolve_leaf_series_ids_expands_direct_children(monkeypatch):
    """Real shape verified live against GSE222665, a genuine SuperSeries: its
    'relation' field lists one 'SuperSeries of: GSEXXXX' entry per subseries plus a
    trailing BioProject link."""
    records = {
        "GSE100": FakeGSE([
            "SuperSeries of: GSE101", "SuperSeries of: GSE102",
            "BioProject: https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1",
        ]),
        "GSE101": FakeGSE(["SubSeries of: GSE100"]),
        "GSE102": FakeGSE(["SubSeries of: GSE100"]),
    }
    monkeypatch.setattr(geo_fetch, "fetch_series", lambda gse_id: records[gse_id])
    assert geo_fetch.resolve_leaf_series_ids("GSE100") == ["GSE101", "GSE102"]


def test_resolve_leaf_series_ids_recurses_into_nested_superseries(monkeypatch):
    records = {
        "GSE100": FakeGSE(["SuperSeries of: GSE200"]),
        "GSE200": FakeGSE(["SuperSeries of: GSE201", "SuperSeries of: GSE202"]),
        "GSE201": FakeGSE(),
        "GSE202": FakeGSE(),
    }
    monkeypatch.setattr(geo_fetch, "fetch_series", lambda gse_id: records[gse_id])
    assert geo_fetch.resolve_leaf_series_ids("GSE100") == ["GSE201", "GSE202"]


def test_resolve_leaf_series_ids_guards_against_cycles(monkeypatch):
    """Malformed/circular relation data must not infinite-recurse -- not a shape
    real GEO data should ever produce, just a safety net."""
    records = {
        "GSE100": FakeGSE(["SuperSeries of: GSE101"]),
        "GSE101": FakeGSE(["SuperSeries of: GSE100"]),
    }
    fetch_calls = []
    monkeypatch.setattr(
        geo_fetch, "fetch_series", lambda gse_id: (fetch_calls.append(gse_id), records[gse_id])[1]
    )
    geo_fetch.resolve_leaf_series_ids("GSE100")  # must return, not hang
    assert fetch_calls.count("GSE100") == 1
    assert fetch_calls.count("GSE101") == 1
