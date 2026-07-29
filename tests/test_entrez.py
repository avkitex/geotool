import pytest

from geotool import entrez


def test_build_query_requires_a_term():
    with pytest.raises(ValueError):
        entrez.build_query()


def test_build_query_combines_terms_and_restricts_entry_type():
    query = entrez.build_query(title="breast cancer", organism="Homo sapiens")
    assert query == '("breast cancer"[Title] AND "Homo sapiens"[Organism]) AND GSE[ETYP]'


def test_build_query_title_only():
    query = entrez.build_query(title="liver")
    assert query == '("liver"[Title]) AND GSE[ETYP]'


def test_esearch_gds_parses_idlist_and_count(requests_mock):
    requests_mock.get(
        entrez.ESEARCH_URL,
        json={"esearchresult": {"idlist": ["1", "2"], "count": "2"}},
    )
    ids, total = entrez.esearch_gds("liver[Title]")
    assert ids == ["1", "2"]
    assert total == 2


def test_esearch_gds_retries_on_429_then_succeeds(requests_mock, monkeypatch):
    monkeypatch.setattr(entrez.time, "sleep", lambda seconds: None)
    requests_mock.get(
        entrez.ESEARCH_URL,
        [
            {"status_code": 429},
            {"json": {"esearchresult": {"idlist": ["1"], "count": "1"}}, "status_code": 200},
        ],
    )
    ids, total = entrez.esearch_gds("liver[Title]")
    assert ids == ["1"]
    assert total == 1
    assert requests_mock.call_count == 2


def test_esearch_gds_gives_up_after_max_retries(requests_mock, monkeypatch):
    monkeypatch.setattr(entrez.time, "sleep", lambda seconds: None)
    requests_mock.get(entrez.ESEARCH_URL, status_code=429)
    with pytest.raises(Exception):
        entrez.esearch_gds("liver[Title]")
    assert requests_mock.call_count == entrez.MAX_RETRIES


def test_esearch_gds_raises_on_entrez_error(requests_mock):
    requests_mock.get(
        entrez.ESEARCH_URL,
        json={"esearchresult": {"ERROR": "Invalid db name specified"}},
    )
    with pytest.raises(RuntimeError):
        entrez.esearch_gds("liver[Title]")


def test_esummary_gds_returns_docsums_in_uid_order(requests_mock):
    requests_mock.get(
        entrez.ESUMMARY_URL,
        json={
            "result": {
                "uids": ["1", "2"],
                "1": {"accession": "GSE1"},
                "2": {"accession": "GSE2"},
            }
        },
    )
    docsums = entrez.esummary_gds(["1", "2"])
    assert [d["accession"] for d in docsums] == ["GSE1", "GSE2"]


def test_esummary_gds_empty_input_makes_no_request(requests_mock):
    assert entrez.esummary_gds([]) == []
    assert not requests_mock.request_history


def test_normalize_docsum():
    docsum = {
        "accession": "GSE339488",
        "title": "Some title",
        "summary": "Some summary",
        "taxon": "Homo sapiens",
        "gpl": "34284;570",
        "n_samples": 6,
        "pdat": "2026/07/27",
        "pubmedids": ["12345"],
    }
    normalized = entrez.normalize_docsum(docsum)
    assert normalized == {
        "gse_id": "GSE339488",
        "title": "Some title",
        "summary": "Some summary",
        "organism": "Homo sapiens",
        "platforms": ["GPL34284", "GPL570"],
        "n_samples": 6,
        "submission_date": "2026/07/27",
        "pubmed_ids": ["12345"],
    }
