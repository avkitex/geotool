import json

import pandas as pd
import requests

from geotool import clinical_annotate, download


class FakeGSM:
    def __init__(self, metadata, table=None):
        self.metadata = metadata
        self.table = table if table is not None else pd.DataFrame()


class FakeGPL:
    def __init__(self, metadata):
        self.metadata = metadata


class FakeGSE:
    def __init__(self, metadata, gsms, gpls=None):
        self.metadata = metadata
        self.gsms = gsms
        self.gpls = gpls or {}


def test_should_skip_url_filters_raw_data_extensions():
    assert download._should_skip_url("ftp://example.com/GSE1_data.CEL.gz")
    assert download._should_skip_url("ftp://example.com/GSM1.fastq.gz")
    assert download._should_skip_url("ftp://example.com/GSM1.bam")
    assert not download._should_skip_url("ftp://example.com/GSE1_counts.tsv.gz")
    assert not download._should_skip_url("ftp://example.com/GSE1_expression.txt")


def make_rnaseq_gse():
    metadata = {
        "geo_accession": ["GSE_RNASEQ"],
        "title": ["An RNA-seq series"],
        "summary": ["s"],
        "supplementary_file": ["ftp://example.com/GSE_RNASEQ_counts.tsv.gz"],
    }
    gsms = {
        "GSM1": FakeGSM({
            "title": ["s1"], "geo_accession": ["GSM1"], "platform_id": ["GPL34284"],
            "organism_ch1": ["Homo sapiens"], "supplementary_file_1": ["NONE"],
            "characteristics_ch1": [],
        }),
    }
    gpls = {"GPL34284": FakeGPL({"title": ["Illumina NovaSeq"], "technology": ["high-throughput sequencing"]})}
    return FakeGSE(metadata, gsms, gpls)


def test_as_https_rewrites_ftp_scheme():
    """requests has no ftp:// adapter; NCBI's FTP host also serves the same
    paths over HTTPS (verified live), so ftp:// URLs must be rewritten
    rather than failing with 'No connection adapters were found'."""
    assert download._as_https("ftp://ftp.ncbi.nlm.nih.gov/geo/foo.tsv.gz") == "https://ftp.ncbi.nlm.nih.gov/geo/foo.tsv.gz"
    assert download._as_https("https://example.com/foo.tsv.gz") == "https://example.com/foo.tsv.gz"


def test_download_file_retries_transient_failure_then_succeeds(requests_mock, tmp_path):
    """Regression test: a real GSE339488 live run hit
    'IncompleteRead(16384 bytes read, 781786 more expected)' mid-download --
    the request+write must be retried as one unit rather than crashing the
    whole cohort on the first transient connection drop."""
    requests_mock.get(
        "https://example.com/file.tsv.gz",
        [
            {"exc": requests.exceptions.ConnectionError("dropped")},
            {"content": b"gene\tcount\nA1BG\t5\n"},
        ],
    )
    path = download._download_file("ftp://example.com/file.tsv.gz", tmp_path)
    assert path is not None
    assert path.read_bytes() == b"gene\tcount\nA1BG\t5\n"


def test_download_file_gives_up_after_max_retries_and_leaves_no_partial_file(requests_mock, tmp_path):
    requests_mock.get("https://example.com/file.tsv.gz", exc=requests.exceptions.ConnectionError("dropped"))
    path = download._download_file("ftp://example.com/file.tsv.gz", tmp_path, retries=2)
    assert path is None
    assert not (tmp_path / "file.tsv.gz").exists()
    assert requests_mock.call_count == 2


def test_download_rnaseq_files_downloads_series_level_file_and_skips_none(requests_mock, tmp_path):
    gse = make_rnaseq_gse()
    # supplementary_file is ftp://, but the actual request must go out over https
    requests_mock.get("https://example.com/GSE_RNASEQ_counts.tsv.gz", content=b"gene\tcount\nA1BG\t5\n")

    downloaded = download.download_rnaseq_files(gse, tmp_path)
    assert len(downloaded) == 1
    assert downloaded[0].name == "GSE_RNASEQ_counts.tsv.gz"
    assert downloaded[0].read_bytes() == b"gene\tcount\nA1BG\t5\n"


def test_download_rnaseq_files_skips_cel_supplementary_files(requests_mock, tmp_path):
    gse = make_rnaseq_gse()
    gse.metadata["supplementary_file"].append("ftp://example.com/GSE_RNASEQ_raw.CEL.gz")
    requests_mock.get("https://example.com/GSE_RNASEQ_counts.tsv.gz", content=b"data")

    downloaded = download.download_rnaseq_files(gse, tmp_path)
    names = [p.name for p in downloaded]
    assert "GSE_RNASEQ_counts.tsv.gz" in names
    assert not any(name.endswith(".CEL.gz") for name in names)


def make_microarray_gse():
    metadata = {"geo_accession": ["GSE_ARRAY"], "title": ["An array series"], "summary": ["s"]}
    gsms = {
        "GSM1": FakeGSM(
            {"title": ["s1"], "geo_accession": ["GSM1"], "platform_id": ["GPL96"], "organism_ch1": ["Homo sapiens"], "characteristics_ch1": []},
            table=pd.DataFrame({"ID_REF": ["1007_s_at", "1053_at"], "VALUE": [656.6, 320.8]}),
        ),
        "GSM2": FakeGSM(
            {"title": ["s2"], "geo_accession": ["GSM2"], "platform_id": ["GPL96"], "organism_ch1": ["Homo sapiens"], "characteristics_ch1": []},
            table=pd.DataFrame({"ID_REF": ["1007_s_at", "1053_at"], "VALUE": [700.1, 310.2]}),
        ),
    }
    gpls = {"GPL96": FakeGPL({"title": ["HG-U133A"], "technology": ["in situ oligonucleotide"], "manufacturer": ["Affymetrix"]})}
    return FakeGSE(metadata, gsms, gpls)


def test_build_and_map_expression_matrix_writes_both_files(monkeypatch, tmp_path):
    gse = make_microarray_gse()
    fake_map = pd.DataFrame([
        {"probe_id": "1007_s_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
        {"probe_id": "1053_at", "gene_symbol": "RFC2", "entrez_id": "5982", "source": "direct_columns"},
    ])
    monkeypatch.setattr(download.probe_mapping, "get_or_build_probe_gene_map", lambda gpl_id: fake_map)

    expr_path = download.build_and_map_expression_matrix(gse, tmp_path)

    assert (tmp_path / "probe_matrix.tsv").exists()
    assert expr_path == tmp_path / "expression.tsv"
    genes = pd.read_csv(expr_path, sep="\t")
    assert set(genes["gene_symbol"]) == {"DDR1", "RFC2"}


def test_download_cohort_routes_rnaseq_and_writes_annotation(monkeypatch, tmp_path):
    gse = make_rnaseq_gse()
    monkeypatch.setattr(download.geo_fetch, "fetch_series", lambda gse_id: gse)
    monkeypatch.setattr(
        download.clinical_annotate, "plan_column_mapping", lambda samples, model=None: clinical_annotate.ColumnMappingPlan()
    )

    class _NoopResponse:
        content = b"gene\tcount\n"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(download.requests, "get", lambda *a, **k: _NoopResponse())

    result = download.download_cohort("GSE_RNASEQ", series_dir=tmp_path)

    assert result["assay_types"] == ["bulk_rnaseq"]
    assert (tmp_path / "GSE_RNASEQ" / "series.tsv").exists()
    assert (tmp_path / "GSE_RNASEQ" / "samples.tsv").exists()
    assert (tmp_path / "GSE_RNASEQ" / "annotation.tsv").exists()
    assert (tmp_path / "GSE_RNASEQ" / "expression").exists()


def test_download_cohort_routes_microarray(monkeypatch, tmp_path):
    gse = make_microarray_gse()
    monkeypatch.setattr(download.geo_fetch, "fetch_series", lambda gse_id: gse)
    monkeypatch.setattr(
        download.clinical_annotate, "plan_column_mapping", lambda samples, model=None: clinical_annotate.ColumnMappingPlan()
    )
    fake_map = pd.DataFrame([
        {"probe_id": "1007_s_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
        {"probe_id": "1053_at", "gene_symbol": "RFC2", "entrez_id": "5982", "source": "direct_columns"},
    ])
    monkeypatch.setattr(download.probe_mapping, "get_or_build_probe_gene_map", lambda gpl_id: fake_map)

    result = download.download_cohort("GSE_ARRAY", series_dir=tmp_path)

    assert result["assay_types"] == ["microarray"]
    assert (tmp_path / "GSE_ARRAY" / "probe_matrix.tsv").exists()
    assert (tmp_path / "GSE_ARRAY" / "expression.tsv").exists()
    assert (tmp_path / "GSE_ARRAY" / "annotation.tsv").exists()
