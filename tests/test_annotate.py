from geotool import annotate


class FakeGSM:
    def __init__(self, metadata):
        self.metadata = metadata


class FakeGPL:
    def __init__(self, metadata):
        self.metadata = metadata


class FakeGSE:
    def __init__(self, metadata, gsms, gpls):
        self.metadata = metadata
        self.gsms = gsms
        self.gpls = gpls


def make_gse():
    gsms = {
        "GSM1": FakeGSM(
            {
                "title": ["siNC rep1"],
                "source_name_ch1": ["MDA-MB-231"],
                "organism_ch1": ["Homo sapiens"],
                "molecule_ch1": ["total RNA"],
                "platform_id": ["GPL34284"],
                "description": ["Library name: rep1"],
                "characteristics_ch1": ["cell line: MDA-MB-231", "treatment: Negative-control siRNA"],
            }
        ),
        "GSM2": FakeGSM(
            {
                "title": ["siBYSL rep1"],
                "source_name_ch1": ["MDA-MB-231"],
                "organism_ch1": ["Homo sapiens"],
                "molecule_ch1": ["total RNA"],
                "platform_id": ["GPL34284"],
                "description": ["Library name: rep2"],
                "characteristics_ch1": ["cell line: MDA-MB-231", "treatment: siBYSL knockdown"],
            }
        ),
    }
    metadata = {
        "geo_accession": ["GSE339488"],
        "title": ["Transcriptomic profiling"],
        "summary": ["A summary."],
        "overall_design": ["Two conditions."],
        "submission_date": ["Jul 22 2026"],
        "pubmed_id": [],
    }
    gpl_metadata = {
        "title": ["Illumina NovaSeq X Plus (Homo sapiens)"],
        "technology": ["high-throughput sequencing"],
    }
    return FakeGSE(metadata, gsms, gpls={"GPL34284": FakeGPL(gpl_metadata)})


def test_parse_characteristics_splits_key_value():
    parsed = annotate.parse_characteristics(["tissue: liver", "age: 45"])
    assert parsed == {"tissue": "liver", "age": "45"}


def test_parse_characteristics_dedupes_repeated_keys():
    parsed = annotate.parse_characteristics(["treatment: A", "treatment: B"])
    assert parsed == {"treatment": "A", "treatment_2": "B"}


def test_parse_characteristics_handles_missing_colon():
    parsed = annotate.parse_characteristics(["just a value"])
    assert parsed == {"just a value": ""}


def test_series_row_pulls_organism_from_first_sample():
    row = annotate.series_row(make_gse())
    assert row["gse_id"] == "GSE339488"
    assert row["organism"] == "Homo sapiens"
    assert row["platforms"] == "GPL34284"
    assert row["n_samples"] == 2


def test_samples_table_has_one_row_per_gsm_and_parsed_characteristics():
    df = annotate.samples_table(make_gse())
    assert list(df["gsm_id"]) == ["GSM1", "GSM2"]
    assert df.loc[df["gsm_id"] == "GSM1", "treatment"].iloc[0] == "Negative-control siRNA"
    assert df.loc[df["gsm_id"] == "GSM2", "treatment"].iloc[0] == "siBYSL knockdown"


def test_write_series_files_writes_tsvs(tmp_path):
    series_path, samples_path = annotate.write_series_files(make_gse(), series_dir=tmp_path)
    assert series_path == tmp_path / "GSE339488" / "series.tsv"
    assert samples_path == tmp_path / "GSE339488" / "samples.tsv"
    assert series_path.exists()
    assert samples_path.exists()
    assert "GSE339488" in series_path.read_text()
