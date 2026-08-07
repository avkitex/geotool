from geotool import geo_fetch


class FakeGSM:
    def __init__(self, supplementary_file=None):
        self.metadata = {"supplementary_file_1": supplementary_file or []}


class FakeGSE:
    def __init__(self, relation=None, supplementary_file=None, gsms=None):
        self.metadata = {"relation": relation or [], "supplementary_file": supplementary_file or []}
        self.gsms = gsms or {}


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


# --- all_supplementary_file_urls ---

def test_all_supplementary_file_urls_includes_series_and_sample_level():
    gse = FakeGSE(
        supplementary_file=["ftp://series_level.tar"],
        gsms={"GSM1": FakeGSM(["ftp://gsm1_file.txt.gz"]), "GSM2": FakeGSM(["ftp://gsm2_file.txt.gz"])},
    )
    assert geo_fetch.all_supplementary_file_urls(gse) == {
        "ftp://series_level.tar", "ftp://gsm1_file.txt.gz", "ftp://gsm2_file.txt.gz",
    }


def test_all_supplementary_file_urls_excludes_none_placeholders():
    gse = FakeGSE(supplementary_file=["NONE"], gsms={"GSM1": FakeGSM(["none", ""])})
    assert geo_fetch.all_supplementary_file_urls(gse) == set()


# --- find_superseries_orphans ---

def test_find_superseries_orphans_returns_empty_when_fully_covered(monkeypatch):
    """The common, expected case (per resolve_leaf_series_ids' own docstring):
    a SuperSeries' own record is just the union of its children's samples/files."""
    parent = FakeGSE(
        supplementary_file=["ftp://shared.tar"],
        gsms={"GSM1": FakeGSM(), "GSM2": FakeGSM()},
    )
    leaf1 = FakeGSE(supplementary_file=["ftp://shared.tar"], gsms={"GSM1": FakeGSM()})
    leaf2 = FakeGSE(gsms={"GSM2": FakeGSM()})
    records = {"GSE100": parent, "GSE101": leaf1, "GSE102": leaf2}
    monkeypatch.setattr(geo_fetch, "fetch_series", lambda gid: records[gid])

    orphans = geo_fetch.find_superseries_orphans("GSE100", ["GSE101", "GSE102"])

    assert orphans == {"orphaned_gsm_ids": [], "orphaned_supplementary_files": []}


def test_find_superseries_orphans_detects_sample_and_file_not_in_any_subseries(monkeypatch):
    parent = FakeGSE(
        supplementary_file=["ftp://covered.tar", "ftp://orphaned.tar"],
        gsms={"GSM1": FakeGSM(), "GSM_ORPHAN": FakeGSM()},
    )
    leaf = FakeGSE(supplementary_file=["ftp://covered.tar"], gsms={"GSM1": FakeGSM()})
    records = {"GSE100": parent, "GSE101": leaf}
    monkeypatch.setattr(geo_fetch, "fetch_series", lambda gid: records[gid])

    orphans = geo_fetch.find_superseries_orphans("GSE100", ["GSE101"])

    assert orphans == {
        "orphaned_gsm_ids": ["GSM_ORPHAN"],
        "orphaned_supplementary_files": ["ftp://orphaned.tar"],
    }


def test_find_superseries_orphans_does_not_flag_a_file_covered_only_at_sample_level(monkeypatch):
    """The duplication risk a series-level-only comparison would miss: the
    parent lists a file at the series level, but the leaf that actually
    covers it only lists it under one of its own samples -- still "covered",
    not "orphaned", since download_rnaseq_files itself looks at both levels."""
    parent = FakeGSE(supplementary_file=["ftp://file.tar"], gsms={"GSM1": FakeGSM()})
    leaf = FakeGSE(gsms={"GSM1": FakeGSM(["ftp://file.tar"])})
    records = {"GSE100": parent, "GSE101": leaf}
    monkeypatch.setattr(geo_fetch, "fetch_series", lambda gid: records[gid])

    orphans = geo_fetch.find_superseries_orphans("GSE100", ["GSE101"])

    assert orphans == {"orphaned_gsm_ids": [], "orphaned_supplementary_files": []}
