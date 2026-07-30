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


def test_write_matrix_gzips_and_rounds_numeric_columns_only(tmp_path):
    df = pd.DataFrame(
        {"gene_symbol": ["DDR1"], "entrez_id": ["780"], "GSM1": [1.234567891]}
    )
    path = tmp_path / "expression.tsv.gz"

    download._write_matrix(df, path, index=False)

    assert path.exists()
    with open(path, "rb") as f:
        assert f.read(2) == b"\x1f\x8b"  # gzip magic bytes
    result = pd.read_csv(path, sep="\t")
    assert result.loc[0, "GSM1"] == 1.235
    assert result.loc[0, "gene_symbol"] == "DDR1"
    assert str(result.loc[0, "entrez_id"]) == "780"


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

    assert (tmp_path / "probe_matrix.tsv.gz").exists()
    assert expr_path == tmp_path / "expression.tsv.gz"
    genes = pd.read_csv(expr_path, sep="\t")
    assert set(genes["gene_symbol"]) == {"DDR1", "RFC2"}


def make_agilent_two_channel_gse():
    metadata = {"geo_accession": ["GSE_AGILENT"], "title": ["A two-channel Agilent series"], "summary": ["s"]}
    gsms = {
        "GSM1": FakeGSM(
            {
                "title": ["s1"], "geo_accession": ["GSM1"], "platform_id": ["GPL2011"],
                "organism_ch1": ["Homo sapiens"], "characteristics_ch1": [], "channel_count": ["2"],
            },
            table=pd.DataFrame({
                "ID_REF": ["1007_s_at", "1053_at"],
                "ch1 Intensity": [10.0, 20.0],
                # Kept under the log2-transform threshold so these channel
                # tests aren't entangled with that separate concern -- see
                # test_probe_mapping.py's dedicated log2-transform tests.
                "ch2 Intensity": [15.0, 25.0],
                "VALUE": [1.0, 1.0],
            }),
        ),
        "GSM2": FakeGSM(
            {
                "title": ["s2"], "geo_accession": ["GSM2"], "platform_id": ["GPL2011"],
                "organism_ch1": ["Homo sapiens"], "characteristics_ch1": [], "channel_count": ["2"],
            },
            table=pd.DataFrame({
                "ID_REF": ["1007_s_at", "1053_at"],
                "ch1 Intensity": [11.0, 21.0],
                "ch2 Intensity": [16.0, 26.0],
                "VALUE": [1.0, 1.0],
            }),
        ),
    }
    gpls = {"GPL2011": FakeGPL({"title": ["Agilent two-color array"], "technology": ["in situ oligonucleotide"], "manufacturer": ["Agilent Technologies"]})}
    return FakeGSE(metadata, gsms, gpls)


_AGILENT_PLATFORM_DETAILS = [{"gpl_id": "GPL2011", "assay_type": "microarray", "vendor": "agilent", "coverage": "full_transcriptome"}]


def test_build_and_map_channel_expression_matrices_writes_both_channels(monkeypatch, tmp_path):
    gse = make_agilent_two_channel_gse()
    fake_map = pd.DataFrame([
        {"probe_id": "1007_s_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
        {"probe_id": "1053_at", "gene_symbol": "RFC2", "entrez_id": "5982", "source": "direct_columns"},
    ])
    monkeypatch.setattr(download.probe_mapping, "get_or_build_probe_gene_map", lambda gpl_id: fake_map)

    result = download.build_and_map_channel_expression_matrices(gse, tmp_path, _AGILENT_PLATFORM_DETAILS)

    assert set(result.keys()) == {1, 2}
    assert result[1] == tmp_path / "channel1_expression.tsv.gz"
    assert result[2] == tmp_path / "channel2_expression.tsv.gz"
    assert (tmp_path / "channel1_probe_matrix.tsv.gz").exists()
    assert (tmp_path / "channel2_probe_matrix.tsv.gz").exists()

    channel1 = pd.read_csv(result[1], sep="\t")
    channel2 = pd.read_csv(result[2], sep="\t")
    assert set(channel1["gene_symbol"]) == {"DDR1", "RFC2"}
    ddr1 = channel1[channel1["gene_symbol"] == "DDR1"].iloc[0]
    assert ddr1["GSM1"] == 10.0
    ddr1_ch2 = channel2[channel2["gene_symbol"] == "DDR1"].iloc[0]
    assert ddr1_ch2["GSM1"] == 15.0


def test_build_and_map_channel_expression_matrices_does_not_touch_ratio_expression(monkeypatch, tmp_path):
    """Splitting must be purely additive -- the existing ratio-based
    expression.tsv.gz (built separately by build_and_map_expression_matrix)
    is untouched by this function."""
    gse = make_agilent_two_channel_gse()
    fake_map = pd.DataFrame([
        {"probe_id": "1007_s_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
        {"probe_id": "1053_at", "gene_symbol": "RFC2", "entrez_id": "5982", "source": "direct_columns"},
    ])
    monkeypatch.setattr(download.probe_mapping, "get_or_build_probe_gene_map", lambda gpl_id: fake_map)

    download.build_and_map_expression_matrix(gse, tmp_path)
    download.build_and_map_channel_expression_matrices(gse, tmp_path, _AGILENT_PLATFORM_DETAILS)

    ratio = pd.read_csv(tmp_path / "expression.tsv.gz", sep="\t")
    ddr1 = ratio[ratio["gene_symbol"] == "DDR1"].iloc[0]
    assert ddr1["GSM1"] == 1.0  # the VALUE column, unaffected by channel splitting


def test_build_and_map_channel_expression_matrices_returns_empty_without_agilent_platform():
    gse = make_microarray_gse()  # Affymetrix, per make_microarray_gse()
    result = download.build_and_map_channel_expression_matrices(
        gse, None, [{"gpl_id": "GPL96", "assay_type": "microarray", "vendor": "affymetrix"}]
    )
    assert result == {}


def test_build_and_map_channel_expression_matrices_returns_empty_when_no_channel_columns():
    """The common case: a 2-channel Agilent series that only publishes the
    VALUE ratio, with no per-channel columns to split."""
    gse = FakeGSE(
        {"geo_accession": ["GSE_X"]},
        {
            "GSM1": FakeGSM(
                {"platform_id": ["GPL887"], "channel_count": ["2"]},
                table=pd.DataFrame({"ID_REF": ["p1"], "VALUE": [0.5]}),
            ),
        },
    )
    result = download.build_and_map_channel_expression_matrices(
        gse, None, [{"gpl_id": "GPL887", "assay_type": "microarray", "vendor": "agilent"}]
    )
    assert result == {}


def test_download_cohort_includes_channel_expression_paths_for_agilent(monkeypatch, tmp_path):
    gse = make_agilent_two_channel_gse()
    monkeypatch.setattr(download.geo_fetch, "fetch_series", lambda gse_id: gse)
    monkeypatch.setattr(
        download.clinical_annotate, "plan_column_mapping", lambda samples, model=None: clinical_annotate.ColumnMappingPlan()
    )
    fake_map = pd.DataFrame([
        {"probe_id": "1007_s_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
        {"probe_id": "1053_at", "gene_symbol": "RFC2", "entrez_id": "5982", "source": "direct_columns"},
    ])
    monkeypatch.setattr(download.probe_mapping, "get_or_build_probe_gene_map", lambda gpl_id: fake_map)

    result = download.download_cohort("GSE_AGILENT", series_dir=tmp_path)

    assert "channel_expression_paths" in result
    assert set(result["channel_expression_paths"].keys()) == {"1", "2"}
    assert (tmp_path / "GSE_AGILENT" / "channel1_expression.tsv.gz").exists()
    assert (tmp_path / "GSE_AGILENT" / "channel2_expression.tsv.gz").exists()
    assert (tmp_path / "GSE_AGILENT" / "expression.tsv.gz").exists()  # ratio path still runs


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


def test_download_cel_files_downloads_only_cel_urls_keyed_by_gsm(requests_mock, tmp_path):
    gsms = {
        "GSM1": FakeGSM({
            "platform_id": ["GPL570"],
            "supplementary_file_1": ["ftp://example.com/GSM1.CEL.gz"],
        }),
        "GSM2": FakeGSM({
            "platform_id": ["GPL570"],
            "supplementary_file_1": ["ftp://example.com/GSM2_processed.txt.gz"],
        }),
    }
    gse = FakeGSE({}, gsms)
    requests_mock.get("https://example.com/GSM1.CEL.gz", content=b"celdata")

    downloaded = download.download_cel_files(gse, tmp_path)

    assert list(downloaded.keys()) == ["GSM1"]
    assert downloaded["GSM1"].read_bytes() == b"celdata"
    assert not (tmp_path / "cel" / "GSM2_processed.txt.gz").exists()


def test_download_cel_files_skips_none_and_missing_supplementary(requests_mock, tmp_path):
    gsms = {"GSM1": FakeGSM({"platform_id": ["GPL570"], "supplementary_file_1": ["NONE"]})}
    gse = FakeGSE({}, gsms)

    downloaded = download.download_cel_files(gse, tmp_path)

    assert downloaded == {}
    assert not (tmp_path / "cel").exists()


def test_build_and_renormalize_expression_matrix_returns_none_without_affymetrix_platform(tmp_path):
    gse = make_microarray_gse()
    result = download.build_and_renormalize_expression_matrix(
        gse, tmp_path, [{"gpl_id": "GPL96", "assay_type": "microarray", "vendor": "illumina"}]
    )
    assert result is None


def test_build_and_renormalize_expression_matrix_writes_expression_rma(monkeypatch, tmp_path):
    gse = make_microarray_gse()
    monkeypatch.setattr(
        download, "download_cel_files", lambda gse, out_dir: {"GSM1": tmp_path / "a.CEL", "GSM2": tmp_path / "b.CEL"}
    )
    fake_probe_matrix = pd.DataFrame(
        {"GSM1": [1.0, 2.0], "GSM2": [3.0, 4.0]}, index=["1007_s_at", "1053_at"]
    )
    monkeypatch.setattr(download.renormalize, "run_rma", lambda cel_files, gpl_id: fake_probe_matrix)
    fake_map = pd.DataFrame([
        {"probe_id": "1007_s_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
        {"probe_id": "1053_at", "gene_symbol": "RFC2", "entrez_id": "5982", "source": "direct_columns"},
    ])
    monkeypatch.setattr(download.probe_mapping, "get_or_build_probe_gene_map", lambda gpl_id: fake_map)

    platform_details = [{"gpl_id": "GPL96", "assay_type": "microarray", "vendor": "affymetrix", "coverage": "full_transcriptome"}]
    expr_path = download.build_and_renormalize_expression_matrix(gse, tmp_path, platform_details)

    assert expr_path == tmp_path / "expression_rma.tsv.gz"
    assert (tmp_path / "probe_matrix_rma_GPL96.tsv.gz").exists()
    genes = pd.read_csv(expr_path, sep="\t")
    assert set(genes["gene_symbol"]) == {"DDR1", "RFC2"}


def test_build_and_renormalize_expression_matrix_deletes_cel_files_after_success(monkeypatch, tmp_path):
    cel_a, cel_b = tmp_path / "a.CEL", tmp_path / "b.CEL"
    cel_a.write_bytes(b"fake")
    cel_b.write_bytes(b"fake")
    gse = make_microarray_gse()
    monkeypatch.setattr(download, "download_cel_files", lambda gse, out_dir: {"GSM1": cel_a, "GSM2": cel_b})
    fake_probe_matrix = pd.DataFrame({"GSM1": [1.0, 2.0], "GSM2": [3.0, 4.0]}, index=["1007_s_at", "1053_at"])
    monkeypatch.setattr(download.renormalize, "run_rma", lambda cel_files, gpl_id: fake_probe_matrix)
    fake_map = pd.DataFrame([
        {"probe_id": "1007_s_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
        {"probe_id": "1053_at", "gene_symbol": "RFC2", "entrez_id": "5982", "source": "direct_columns"},
    ])
    monkeypatch.setattr(download.probe_mapping, "get_or_build_probe_gene_map", lambda gpl_id: fake_map)

    platform_details = [{"gpl_id": "GPL96", "assay_type": "microarray", "vendor": "affymetrix"}]
    download.build_and_renormalize_expression_matrix(gse, tmp_path, platform_details)

    assert not cel_a.exists()
    assert not cel_b.exists()


def test_build_and_renormalize_expression_matrix_keeps_cel_files_when_rma_unavailable(monkeypatch, tmp_path):
    cel_a = tmp_path / "a.CEL"
    cel_a.write_bytes(b"fake")
    gse = make_microarray_gse()
    monkeypatch.setattr(download, "download_cel_files", lambda gse, out_dir: {"GSM1": cel_a})

    def raise_unavailable(cel_files, gpl_id):
        raise download.renormalize.RmaUnavailable("no known Bioconductor CDF/pd package")

    monkeypatch.setattr(download.renormalize, "run_rma", raise_unavailable)

    platform_details = [{"gpl_id": "GPL96", "assay_type": "microarray", "vendor": "affymetrix"}]
    download.build_and_renormalize_expression_matrix(gse, tmp_path, platform_details)

    assert cel_a.exists()  # not deleted -- RMA never succeeded, so raw data is still needed


def test_build_and_renormalize_expression_matrix_rounds_values_to_3_decimals(monkeypatch, tmp_path):
    gse = make_microarray_gse()
    monkeypatch.setattr(download, "download_cel_files", lambda gse, out_dir: {"GSM1": tmp_path / "a.CEL"})
    fake_probe_matrix = pd.DataFrame({"GSM1": [1.234567891]}, index=["1007_s_at"])
    monkeypatch.setattr(download.renormalize, "run_rma", lambda cel_files, gpl_id: fake_probe_matrix)
    fake_map = pd.DataFrame([
        {"probe_id": "1007_s_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
    ])
    monkeypatch.setattr(download.probe_mapping, "get_or_build_probe_gene_map", lambda gpl_id: fake_map)

    platform_details = [{"gpl_id": "GPL96", "assay_type": "microarray", "vendor": "affymetrix"}]
    download.build_and_renormalize_expression_matrix(gse, tmp_path, platform_details)

    probes = pd.read_csv(tmp_path / "probe_matrix_rma_GPL96.tsv.gz", sep="\t", index_col=0)
    assert probes.loc["1007_s_at", "GSM1"] == 1.235


def test_build_and_renormalize_expression_matrix_skips_platform_when_rma_unavailable(monkeypatch, tmp_path):
    gse = make_microarray_gse()
    monkeypatch.setattr(download, "download_cel_files", lambda gse, out_dir: {"GSM1": tmp_path / "a.CEL"})

    def raise_unavailable(cel_files, gpl_id):
        raise download.renormalize.RmaUnavailable("Rscript not found on PATH -- install R to use --rma")

    monkeypatch.setattr(download.renormalize, "run_rma", raise_unavailable)

    platform_details = [{"gpl_id": "GPL96", "assay_type": "microarray", "vendor": "affymetrix"}]
    result = download.build_and_renormalize_expression_matrix(gse, tmp_path, platform_details)

    assert result is None
    assert not (tmp_path / "expression_rma.tsv.gz").exists()


def test_download_cohort_with_rma_flag_adds_rma_expression(monkeypatch, tmp_path):
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
    monkeypatch.setattr(
        download, "download_cel_files", lambda gse, out_dir: {"GSM1": tmp_path / "a.CEL", "GSM2": tmp_path / "b.CEL"}
    )
    fake_rma_matrix = pd.DataFrame({"GSM1": [1.0, 2.0], "GSM2": [3.0, 4.0]}, index=["1007_s_at", "1053_at"])
    monkeypatch.setattr(download.renormalize, "run_rma", lambda cel_files, gpl_id: fake_rma_matrix)

    result = download.download_cohort("GSE_ARRAY", series_dir=tmp_path, rma=True)

    assert result["expression_rma_path"] is not None
    assert (tmp_path / "GSE_ARRAY" / "expression_rma.tsv.gz").exists()
    assert (tmp_path / "GSE_ARRAY" / "expression.tsv.gz").exists()  # submitter-value path still runs


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
    assert (tmp_path / "GSE_ARRAY" / "probe_matrix.tsv.gz").exists()
    assert (tmp_path / "GSE_ARRAY" / "expression.tsv.gz").exists()
    assert (tmp_path / "GSE_ARRAY" / "annotation.tsv").exists()
