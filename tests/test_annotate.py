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


def make_two_channel_gse(n_reference_hint_samples=3):
    """Real GSE71729 shape (the Moffitt "Virtual Microdissection of PDAC"
    cohort): channel 1 is always "Human Reference" on every array (a fixed
    pooled reference, matched by probe_mapping._REFERENCE_HINT_RE), channel
    2 carries the real per-sample tumor/met identity in characteristics_ch2
    -- entirely different keys/values from channel 1's, and previously never
    read by samples_table at all. No per-channel expression VALUE columns in
    the data table (most two-channel Agilent series only publish the ratio),
    so this can only be resolved from metadata, not
    probe_mapping.detect_reference_channel's variance signal.
    """
    gsms = {}
    for i in range(n_reference_hint_samples):
        gsms[f"GSM{i}"] = FakeGSM({
            "title": [f"5384{i}-Met-LymphNode"],
            "channel_count": ["2"],
            "source_name_ch1": ["Human Reference"],
            "organism_ch1": ["Homo sapiens"],
            "molecule_ch1": ["total RNA"],
            "characteristics_ch1": ["sample type: Stratagene Human reference RNA"],
            "source_name_ch2": ["LymphNode_Metastasis"],
            "organism_ch2": ["Homo sapiens"],
            "molecule_ch2": ["total RNA"],
            "characteristics_ch2": ["cell line/tissue: LymphNode", "tissue type: Metastasis"],
            "platform_id": ["GPL20769"],
        })
    metadata = {"geo_accession": ["GSE71729"], "title": ["Virtual Microdissection of PDAC"], "pubmed_id": []}
    return FakeGSE(metadata, gsms, gpls={})


def test_series_reference_channel_detects_ch1_as_fixed_reference():
    assert annotate._series_reference_channel(make_two_channel_gse()) == 1


def test_series_reference_channel_none_for_single_channel_series():
    assert annotate._series_reference_channel(make_gse()) is None


def test_series_reference_channel_none_when_hints_disagree():
    """Two samples calling channel 1 the reference, two calling channel 2 --
    no clear series-level majority (probe_mapping._MIN_METADATA_AGREEMENT),
    so no call is made rather than guessing."""
    gsms = {
        "GSM1": FakeGSM({
            "channel_count": ["2"],
            "source_name_ch1": ["Reference pool"], "characteristics_ch1": [],
            "source_name_ch2": ["Tumor sample A"], "characteristics_ch2": [],
        }),
        "GSM2": FakeGSM({
            "channel_count": ["2"],
            "source_name_ch1": ["Tumor sample B"], "characteristics_ch1": [],
            "source_name_ch2": ["Reference pool"], "characteristics_ch2": [],
        }),
    }
    gse = FakeGSE({"geo_accession": ["GSE1"], "pubmed_id": []}, gsms, gpls={})
    assert annotate._series_reference_channel(gse) is None


def test_samples_table_captures_ch2_fields_and_reference_channel_call():
    df = annotate.samples_table(make_two_channel_gse())
    row = df.iloc[0]
    assert row["source_name_ch1"] == "Human Reference"
    assert row["source_name_ch2"] == "LymphNode_Metastasis"
    assert row["tissue type_ch2"] == "Metastasis"
    assert row["cell line/tissue_ch2"] == "LymphNode"
    assert row["channel_count"] == "2"
    assert row["reference_channel"] == 1


def test_samples_table_single_channel_series_has_no_channel2_columns():
    """Regression: the overwhelming majority (single-channel) case must stay
    completely unaffected -- no new columns appear at all."""
    df = annotate.samples_table(make_gse())
    assert not any(c.endswith("_ch2") for c in df.columns)
    assert "channel_count" not in df.columns
    assert "reference_channel" not in df.columns


def test_write_series_files_writes_tsvs(tmp_path):
    series_path, samples_path = annotate.write_series_files(make_gse(), series_dir=tmp_path)
    assert series_path == tmp_path / "GSE339488" / "series.tsv"
    assert samples_path == tmp_path / "GSE339488" / "samples.tsv"
    assert series_path.exists()
    assert samples_path.exists()
    assert "GSE339488" in series_path.read_text()
