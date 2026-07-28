import pytest

from geotool import search as search_mod


class FakeGSM:
    def __init__(self, metadata):
        self.metadata = metadata


class FakeGSE:
    def __init__(self, metadata, gsms, gpls=None):
        self.metadata = metadata
        self.gsms = gsms
        self.gpls = gpls or {}


def make_gse(gse_id, characteristics_by_sample):
    gsms = {
        gsm_id: FakeGSM(
            {
                "title": [gsm_id],
                "source_name_ch1": ["cells"],
                "organism_ch1": ["Homo sapiens"],
                "molecule_ch1": ["total RNA"],
                "platform_id": ["GPL1"],
                "description": [],
                "characteristics_ch1": chars,
            }
        )
        for gsm_id, chars in characteristics_by_sample.items()
    }
    metadata = {"geo_accession": [gse_id], "title": [gse_id], "summary": [], "pubmed_id": []}
    return FakeGSE(metadata, gsms)


def test_parse_property_filter():
    assert search_mod.parse_property_filter("Tissue:Liver") == ("tissue", "liver")


def test_parse_property_filter_requires_colon():
    with pytest.raises(ValueError):
        search_mod.parse_property_filter("tissue")


def test_filter_by_sample_properties_keeps_series_with_a_match(monkeypatch):
    gse_match = make_gse("GSE1", {"GSM1": ["tissue: liver"], "GSM2": ["tissue: kidney"]})
    gse_no_match = make_gse("GSE2", {"GSM3": ["tissue: kidney"]})

    def fake_fetch(gse_id):
        return {"GSE1": gse_match, "GSE2": gse_no_match}[gse_id]

    monkeypatch.setattr(search_mod.geo_fetch, "fetch_series", fake_fetch)

    candidates = [{"gse_id": "GSE1"}, {"gse_id": "GSE2"}]
    kept = search_mod.filter_by_sample_properties(candidates, ["tissue:liver"], write_annotations=False)

    assert [c["gse_id"] for c in kept] == ["GSE1"]
    assert kept[0]["sample_property_matches"] == "tissue:liver=1/2"


def test_filter_by_sample_properties_requires_all_filters_to_match(monkeypatch):
    gse = make_gse("GSE1", {"GSM1": ["tissue: liver", "treatment: drugA"]})
    monkeypatch.setattr(search_mod.geo_fetch, "fetch_series", lambda gse_id: gse)

    candidates = [{"gse_id": "GSE1"}]
    kept = search_mod.filter_by_sample_properties(
        candidates, ["tissue:liver", "treatment:drugB"], write_annotations=False
    )
    assert kept == []


def test_filter_by_sample_properties_falls_back_to_full_row_search_for_unknown_key(monkeypatch):
    gse = make_gse("GSE1", {"GSM1": ["cell line: HeLa"]})
    monkeypatch.setattr(search_mod.geo_fetch, "fetch_series", lambda gse_id: gse)

    candidates = [{"gse_id": "GSE1"}]
    kept = search_mod.filter_by_sample_properties(candidates, ["anykey:hela"], write_annotations=False)
    assert [c["gse_id"] for c in kept] == ["GSE1"]


def test_filter_by_sample_properties_keeps_series_on_fetch_error(monkeypatch):
    def raise_error(gse_id):
        raise RuntimeError("network down")

    monkeypatch.setattr(search_mod.geo_fetch, "fetch_series", raise_error)

    candidates = [{"gse_id": "GSE1"}]
    kept = search_mod.filter_by_sample_properties(candidates, ["tissue:liver"], write_annotations=False)
    assert len(kept) == 1
    assert "ERROR" in kept[0]["sample_property_matches"]


def test_search_without_sample_properties_skips_fetch(monkeypatch):
    called = []
    monkeypatch.setattr(search_mod.geo_fetch, "fetch_series", lambda gse_id: called.append(gse_id))
    monkeypatch.setattr(
        search_mod.entrez,
        "search_series",
        lambda **kwargs: [{"gse_id": "GSE1", "title": "t"}],
    )

    rows = search_mod.search(title="cancer")
    assert rows == [{"gse_id": "GSE1", "title": "t", "sample_property_matches": ""}]
    assert called == []
