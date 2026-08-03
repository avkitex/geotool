import numpy as np
import pandas as pd

from geotool import probe_mapping

# Real annotation text captured live from GEO during planning.
GPL96_GENE_ASSIGNMENT_UNUSED = None  # GPL96 uses direct columns, not gene_assignment

GPL17586_GENE_ASSIGNMENT = (
    "NR_046018 // DDX11L1 // DEAD/H (Asp-Glu-Ala-Asp/His) box helicase 11 like 1 // "
    "1p36.33 // 100287102 /// ENST00000456328 // DDX11L5 // DEAD/H box helicase 11 like 5 // "
    "9p24.3 // 100287596 /// ENST00000456328 // DDX11L1 // DEAD/H box helicase 11 like 1 // "
    "1p36.33 // 100287102"
)

GPL17586_GENE_ASSIGNMENT_NO_SYMBOL = "--- /// ---"


def test_parse_direct_columns_gpl96_style_takes_first_of_multi_value_cell():
    df = pd.DataFrame([
        {"ID": "1007_s_at", "Gene Symbol": "DDR1 /// MIR4640", "ENTREZ_GENE_ID": "780 /// 100616237"},
        {"ID": "1053_at", "Gene Symbol": "RFC2", "ENTREZ_GENE_ID": "5982"},
    ])
    result = probe_mapping.parse_direct_columns(df)
    assert result is not None
    row0 = result[result["probe_id"] == "1007_s_at"].iloc[0]
    assert row0["gene_symbol"] == "DDR1"
    assert row0["entrez_id"] == "780"
    assert (result["source"] == "direct_columns").all()


def test_parse_direct_columns_skips_dash_placeholder():
    df = pd.DataFrame([{"ID": "AFFX-ctrl", "Gene Symbol": "---", "ENTREZ_GENE_ID": "---"}])
    result = probe_mapping.parse_direct_columns(df)
    assert result is not None
    assert result.empty


def test_parse_direct_columns_returns_none_when_no_known_columns():
    df = pd.DataFrame([{"ID": "1007_s_at", "gene_assignment": "whatever"}])
    assert probe_mapping.parse_direct_columns(df) is None


def test_parse_gene_assignment_extracts_first_valid_symbol_and_entrez():
    symbol, entrez = probe_mapping.parse_gene_assignment(GPL17586_GENE_ASSIGNMENT)
    assert symbol == "DDX11L1"
    assert entrez == "100287102"


def test_parse_gene_assignment_handles_no_usable_entry():
    assert probe_mapping.parse_gene_assignment(GPL17586_GENE_ASSIGNMENT_NO_SYMBOL) == (None, None)
    assert probe_mapping.parse_gene_assignment(None) == (None, None)
    assert probe_mapping.parse_gene_assignment("") == (None, None)


def test_extract_probe_gene_table_prefers_direct_columns_over_gene_assignment():
    df = pd.DataFrame([{"ID": "1007_s_at", "Gene Symbol": "DDR1", "ENTREZ_GENE_ID": "780", "gene_assignment": "x // Y // z"}])
    result = probe_mapping.extract_probe_gene_table(df)
    assert (result["source"] == "direct_columns").all()


def test_extract_probe_gene_table_falls_back_to_gene_assignment():
    df = pd.DataFrame([{"ID": "TC01000001.hg.1", "gene_assignment": GPL17586_GENE_ASSIGNMENT}])
    result = probe_mapping.extract_probe_gene_table(df)
    assert len(result) == 1
    assert result.iloc[0]["gene_symbol"] == "DDX11L1"
    assert result.iloc[0]["source"] == "gene_assignment"


def test_extract_probe_gene_table_unmapped_when_nothing_usable():
    df = pd.DataFrame([{"ID": "custom_probe_1", "some_other_column": "n/a"}])
    result = probe_mapping.extract_probe_gene_table(df)
    assert result.empty
    assert list(result.columns) == ["probe_id", "gene_symbol", "entrez_id", "source"]


# Real GPL23432 (Brainarray ENSG custom-CDF re-annotation of Affymetrix
# HG-U133 Plus 2) annotation table shape, captured live during planning.
GPL23432_ROWS = [
    {
        "ID": "ENSG00000000003_at",
        "ORF": "ENSG00000000003",
        "Description": "tetraspanin 6 [Source:HGNC Symbol;Acc:11858]",
    },
    {
        "ID": "ENSG00000000005_at",
        "ORF": "ENSG00000000005",
        "Description": "tenomodulin [Source:HGNC Symbol;Acc:17757]",
    },
]


def test_parse_orf_ensembl_column_maps_probe_to_ensembl_gene_id():
    df = pd.DataFrame(GPL23432_ROWS)
    result = probe_mapping.parse_orf_ensembl_column(df)
    assert result is not None
    row0 = result[result["probe_id"] == "ENSG00000000003_at"].iloc[0]
    assert row0["gene_symbol"] == "ENSG00000000003"
    assert row0["entrez_id"] is None
    assert (result["source"] == "ensembl_orf").all()


def test_parse_orf_ensembl_column_returns_none_without_orf_column():
    df = pd.DataFrame([{"ID": "1007_s_at", "Gene Symbol": "DDR1"}])
    assert probe_mapping.parse_orf_ensembl_column(df) is None


def test_parse_orf_ensembl_column_returns_none_when_orf_is_not_ensembl_ids():
    """Guards against older spotted-array platforms that also name a column
    "ORF" but for an unrelated identifier scheme -- must not misinterpret
    those values as Ensembl gene IDs just because the column name matches."""
    df = pd.DataFrame([{"ID": "probe1", "ORF": "IMAGE:2450123"}])
    assert probe_mapping.parse_orf_ensembl_column(df) is None


def test_extract_probe_gene_table_falls_back_to_orf_ensembl_column():
    df = pd.DataFrame(GPL23432_ROWS)
    result = probe_mapping.extract_probe_gene_table(df)
    assert len(result) == 2
    assert set(result["gene_symbol"]) == {"ENSG00000000003", "ENSG00000000005"}
    assert (result["source"] == "ensembl_orf").all()


def test_aggregate_probes_to_genes_works_with_ensembl_orf_mapping():
    """The empty-expression-matrix bug on GSE98588 (GPL23432): probes must
    still aggregate into a non-empty gene-level matrix even when the only
    available gene key is the Ensembl-ID-flavored gene_symbol, not entrez_id."""
    probe_matrix = pd.DataFrame(
        {"GSM1": [5.0], "GSM2": [7.0]}, index=["ENSG00000000003_at"]
    )
    probe_gene_map = pd.DataFrame([
        {"probe_id": "ENSG00000000003_at", "gene_symbol": "ENSG00000000003", "entrez_id": None, "source": "ensembl_orf"},
    ])
    genes = probe_mapping.aggregate_probes_to_genes(probe_matrix, probe_gene_map)
    assert not genes.empty
    assert genes.iloc[0]["gene_symbol"] == "ENSG00000000003"


# Real GPL7091 (older Agilent spotted oligo array) annotation table shape,
# captured live during planning: annotated probes carry a real gene symbol
# in PrimarySequenceName, unannotated ones a bare internal clone ID instead.
GPL7091_ROWS = [
    {"ID": "1", "PrimarySequenceName": "LOC51235", "ORF": "LOC51235"},
    {"ID": "2", "PrimarySequenceName": "I_959282", "ORF": "I_959282"},
    {"ID": "4", "PrimarySequenceName": "NIT2", "ORF": "NIT2"},
]


def test_parse_primary_sequence_name_column_maps_annotated_probes_only():
    df = pd.DataFrame(GPL7091_ROWS)
    result = probe_mapping.parse_primary_sequence_name_column(df)
    assert result is not None
    assert set(result["probe_id"]) == {"1", "4"}  # probe 2's I_959282 clone id excluded
    assert set(result["gene_symbol"]) == {"LOC51235", "NIT2"}
    assert (result["source"] == "primary_sequence_name").all()


def test_parse_primary_sequence_name_column_returns_none_without_column():
    df = pd.DataFrame([{"ID": "1007_s_at", "Gene Symbol": "DDR1"}])
    assert probe_mapping.parse_primary_sequence_name_column(df) is None


def test_extract_probe_gene_table_falls_back_to_primary_sequence_name():
    df = pd.DataFrame(GPL7091_ROWS)
    result = probe_mapping.extract_probe_gene_table(df)
    assert len(result) == 2
    assert set(result["gene_symbol"]) == {"LOC51235", "NIT2"}
    assert (result["source"] == "primary_sequence_name").all()


def test_extract_probe_gene_table_prefers_orf_ensembl_over_primary_sequence_name():
    df = pd.DataFrame([{
        "ID": "ENSG00000000003_at", "ORF": "ENSG00000000003", "PrimarySequenceName": "TSPAN6",
    }])
    result = probe_mapping.extract_probe_gene_table(df)
    assert (result["source"] == "ensembl_orf").all()


class FakeGSM:
    def __init__(self, table, metadata=None):
        self.table = table
        self.metadata = metadata if metadata is not None else {"platform_id": ["GPL96"]}


class FakeGSE:
    def __init__(self, gsms):
        self.gsms = gsms


def test_build_probe_matrix_from_sample_tables():
    gse = FakeGSE({
        "GSM1": FakeGSM(pd.DataFrame({"ID_REF": ["p1", "p2"], "VALUE": [1.0, 2.0]})),
        "GSM2": FakeGSM(pd.DataFrame({"ID_REF": ["p1", "p2"], "VALUE": [3.0, 4.0]})),
    })
    matrix = probe_mapping.build_probe_matrix(gse)
    assert list(matrix.columns) == ["GSM1", "GSM2"]
    assert matrix.loc["p1", "GSM1"] == 1.0
    assert matrix.loc["p2", "GSM2"] == 4.0


def test_build_probe_matrix_skips_samples_without_a_data_table():
    gse = FakeGSE({
        "GSM1": FakeGSM(pd.DataFrame({"ID_REF": ["p1"], "VALUE": [1.0]})),
        "GSM2": FakeGSM(pd.DataFrame()),  # RNA-seq sample: empty table
    })
    matrix = probe_mapping.build_probe_matrix(gse)
    assert list(matrix.columns) == ["GSM1"]


def test_build_probe_matrix_all_empty_returns_empty_dataframe():
    gse = FakeGSE({"GSM1": FakeGSM(pd.DataFrame())})
    assert probe_mapping.build_probe_matrix(gse).empty


def test_build_probe_matrix_handles_mixed_int_and_str_id_ref_across_samples():
    # Purely-numeric probe IDs (e.g. GPL6801/GPL7723-style) can come back as
    # int64 for one sample's table and str for another's, depending on what
    # else pandas saw while parsing each GSM's own table. Aligning those
    # differently-typed indexes used to crash with
    # "'<' not supported between instances of 'int' and 'str'" (GSE32688).
    gse = FakeGSE({
        "GSM1": FakeGSM(pd.DataFrame({"ID_REF": [1, 2], "VALUE": [1.0, 2.0]})),
        "GSM2": FakeGSM(pd.DataFrame({"ID_REF": ["1", "2"], "VALUE": [3.0, 4.0]})),
    })
    matrix = probe_mapping.build_probe_matrix(gse)
    assert list(matrix.columns) == ["GSM1", "GSM2"]
    assert list(matrix.index) == ["1", "2"]
    assert matrix.loc["1", "GSM1"] == 1.0
    assert matrix.loc["2", "GSM2"] == 4.0


def test_aggregate_probes_to_genes_averages_multiple_probes_for_same_gene():
    # Values kept under the log2-transform threshold (see
    # _LOG2_ALREADY_TRANSFORMED_MAX) so this test's averaging arithmetic
    # isn't entangled with that separate concern -- see the dedicated
    # test_aggregate_probes_to_genes_log2_transforms_* tests for that.
    probe_matrix = pd.DataFrame(
        {"GSM1": [6.566, 3.208, 3.613], "GSM2": [7.001, 3.102, 4.0]},
        index=["1007_s_at", "1053_at", "117_at"],
    )
    probe_gene_map = pd.DataFrame([
        {"probe_id": "1007_s_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
        {"probe_id": "1053_at", "gene_symbol": "RFC2", "entrez_id": "5982", "source": "direct_columns"},
        {"probe_id": "117_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
    ])
    genes = probe_mapping.aggregate_probes_to_genes(probe_matrix, probe_gene_map)
    ddr1 = genes[genes["gene_symbol"] == "DDR1"].iloc[0]
    assert ddr1["GSM1"] == (6.566 + 3.613) / 2
    assert ddr1["GSM2"] == (7.001 + 4.0) / 2
    assert len(genes) == 2  # DDR1 + RFC2, not 3 rows


def test_needs_log2_transform_true_when_any_value_over_threshold():
    assert probe_mapping.needs_log2_transform(pd.DataFrame({"GSM1": [1.0, 656.6]}))
    assert not probe_mapping.needs_log2_transform(pd.DataFrame({"GSM1": [1.0, 49.9]}))


def test_needs_log2_transform_false_for_negative_log_ratio_values():
    """A two-channel log-ratio (or any already-log2 data) can be negative --
    must never be mistaken for "needs transforming"."""
    assert not probe_mapping.needs_log2_transform(pd.DataFrame({"GSM1": [-4.68, 6.29, -0.02]}))


def test_needs_log2_transform_false_for_empty_matrix():
    assert not probe_mapping.needs_log2_transform(pd.DataFrame())


def test_maybe_log2_transform_applies_when_needed():
    matrix = pd.DataFrame({"GSM1": [0.0, 99.0]})
    result = probe_mapping.maybe_log2_transform(matrix)
    assert result["GSM1"].tolist() == [np.log2(1.0), np.log2(100.0)]


def test_maybe_log2_transform_leaves_already_log2_data_untouched():
    matrix = pd.DataFrame({"GSM1": [-4.68, 6.29]})
    result = probe_mapping.maybe_log2_transform(matrix)
    assert result["GSM1"].tolist() == [-4.68, 6.29]


def test_aggregate_probes_to_genes_log2_transforms_raw_linear_scale_values():
    """Raw, untransformed microarray values (e.g. Affymetrix MAS5-style
    intensities in the hundreds) must come out log2(x + 1)-transformed at
    the gene-expression level."""
    probe_matrix = pd.DataFrame({"GSM1": [656.6], "GSM2": [700.1]}, index=["1007_s_at"])
    probe_gene_map = pd.DataFrame([
        {"probe_id": "1007_s_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
    ])
    genes = probe_mapping.aggregate_probes_to_genes(probe_matrix, probe_gene_map)
    ddr1 = genes.iloc[0]
    assert ddr1["GSM1"] == np.log2(656.6 + 1)
    assert ddr1["GSM2"] == np.log2(700.1 + 1)


def test_aggregate_probes_to_genes_leaves_already_log2_ratio_untouched():
    """A VALUE ratio (or any already-log2 data) can be negative and must
    never be re-transformed, even at the gene-expression level."""
    probe_matrix = pd.DataFrame({"GSM1": [-4.68], "GSM2": [6.29]}, index=["1007_s_at"])
    probe_gene_map = pd.DataFrame([
        {"probe_id": "1007_s_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
    ])
    genes = probe_mapping.aggregate_probes_to_genes(probe_matrix, probe_gene_map)
    ddr1 = genes.iloc[0]
    assert ddr1["GSM1"] == -4.68
    assert ddr1["GSM2"] == 6.29


def test_aggregate_probes_to_genes_drops_unmapped_probes():
    probe_matrix = pd.DataFrame({"GSM1": [1.0, 2.0]}, index=["p1", "p2"])
    probe_gene_map = pd.DataFrame([
        {"probe_id": "p1", "gene_symbol": "GENEA", "entrez_id": "1", "source": "direct_columns"},
        {"probe_id": "p2", "gene_symbol": None, "entrez_id": None, "source": "unmapped"},
    ])
    genes = probe_mapping.aggregate_probes_to_genes(probe_matrix, probe_gene_map)
    assert len(genes) == 1
    assert genes.iloc[0]["gene_symbol"] == "GENEA"


def test_aggregate_probes_to_genes_falls_back_to_symbol_key_when_no_entrez():
    probe_matrix = pd.DataFrame({"GSM1": [1.0, 3.0]}, index=["p1", "p2"])
    probe_gene_map = pd.DataFrame([
        {"probe_id": "p1", "gene_symbol": "GENEB", "entrez_id": None, "source": "direct_columns"},
        {"probe_id": "p2", "gene_symbol": "GENEB", "entrez_id": None, "source": "direct_columns"},
    ])
    genes = probe_mapping.aggregate_probes_to_genes(probe_matrix, probe_gene_map)
    assert len(genes) == 1
    assert genes.iloc[0]["GSM1"] == 2.0


def test_get_or_build_probe_gene_map_caches_to_disk(monkeypatch, tmp_path):
    class FakeGPL:
        table = pd.DataFrame([{"ID": "1007_s_at", "Gene Symbol": "DDR1", "ENTREZ_GENE_ID": "780"}])

    calls = []
    monkeypatch.setattr(probe_mapping.geo_fetch, "fetch_platform", lambda gpl_id: (calls.append(gpl_id), FakeGPL())[1])

    first = probe_mapping.get_or_build_probe_gene_map("GPL96", platforms_dir=tmp_path)
    assert len(calls) == 1
    assert (tmp_path / "GPL96" / "probe_gene_map.tsv").exists()

    second = probe_mapping.get_or_build_probe_gene_map("GPL96", platforms_dir=tmp_path)
    assert len(calls) == 1  # cache hit, no second fetch
    assert second.iloc[0]["gene_symbol"] == "DDR1"


def test_get_or_build_probe_gene_map_probe_id_dtype_matches_on_cache_miss_and_hit(monkeypatch, tmp_path):
    """Regression test: platforms with purely-numeric probe IDs (e.g.
    GPL7091's "1", "2", ...) inherit the platform table's own int64 "ID"
    dtype on a cache miss, but the cache-hit path forces str -- this
    mismatch silently broke aggregate_probes_to_genes's index join
    depending on which one happened to run first in a given process (found
    live on GSE12234: the ratio-based expression.tsv.gz worked because it
    ran on a cache miss, but channel1/channel2_expression.tsv.gz came back
    empty because they ran right after, on a cache hit)."""
    class FakeGPL:
        table = pd.DataFrame([{"ID": 1, "Gene Symbol": "DDR1", "ENTREZ_GENE_ID": "780"}])

    monkeypatch.setattr(probe_mapping.geo_fetch, "fetch_platform", lambda gpl_id: FakeGPL())

    cache_miss = probe_mapping.get_or_build_probe_gene_map("GPL7091", platforms_dir=tmp_path)
    cache_hit = probe_mapping.get_or_build_probe_gene_map("GPL7091", platforms_dir=tmp_path)

    # Both must join a numeric-indexed probe_matrix identically -- that's
    # what actually broke, not the exact dtype name (pandas has more than one
    # string-flavored dtype depending on version/config).
    probe_matrix = pd.DataFrame({"GSM1": [5.0]}, index=pd.Index([1], dtype="int64"))
    genes_from_miss = probe_mapping.aggregate_probes_to_genes(probe_matrix, cache_miss)
    genes_from_hit = probe_mapping.aggregate_probes_to_genes(probe_matrix, cache_hit)
    assert not genes_from_miss.empty
    assert not genes_from_hit.empty
    assert genes_from_miss.iloc[0]["gene_symbol"] == genes_from_hit.iloc[0]["gene_symbol"] == "DDR1"


def test_aggregate_probes_to_genes_matches_despite_numeric_probe_matrix_index():
    """Regression test companion: even if a caller passes a probe_matrix
    with an int64 index (as GEOparse produces for numeric-ID platforms),
    the join against probe_gene_map's str probe_id must still succeed."""
    probe_matrix = pd.DataFrame({"GSM1": [5.0]}, index=pd.Index([1], dtype="int64"))
    probe_gene_map = pd.DataFrame([
        {"probe_id": "1", "gene_symbol": "DDR1", "entrez_id": None, "source": "direct_columns"},
    ])
    genes = probe_mapping.aggregate_probes_to_genes(probe_matrix, probe_gene_map)
    assert not genes.empty
    assert genes.iloc[0]["gene_symbol"] == "DDR1"
    assert genes.iloc[0]["GSM1"] == 5.0


def test_detect_channel_columns_finds_known_naming_variants():
    # ch1/ch2-style, seen live on GPL2011
    df = pd.DataFrame({"ID_REF": ["p1"], "ch1 Intensity": [1.0], "ch2 Intensity": [2.0]})
    assert probe_mapping.detect_channel_columns(df) == ("ch1 Intensity", "ch2 Intensity")

    # Cy3/Cy5-named, seen live on GPL7091
    df2 = pd.DataFrame({"ID_REF": ["p1"], "Intensity_Cy3": [1.0], "Intensity_Cy5": [2.0]})
    assert probe_mapping.detect_channel_columns(df2) == ("Intensity_Cy3", "Intensity_Cy5")


def test_detect_channel_columns_none_when_only_ratio_value_present():
    """The common case: most two-channel series only publish the precomputed
    VALUE ratio, with no per-channel columns at all -- must not guess."""
    df = pd.DataFrame({"ID_REF": ["p1"], "VALUE": [0.5]})
    assert probe_mapping.detect_channel_columns(df) is None


def test_detect_channel_columns_requires_both_columns_of_a_pair():
    df = pd.DataFrame({"ID_REF": ["p1"], "ch1 Intensity": [1.0]})  # ch2 Intensity missing
    assert probe_mapping.detect_channel_columns(df) is None


def test_build_channel_probe_matrices_splits_two_channel_samples():
    gse = FakeGSE({
        "GSM1": FakeGSM(
            pd.DataFrame({"ID_REF": ["p1", "p2"], "ch1 Intensity": [10.0, 20.0], "ch2 Intensity": [100.0, 200.0]}),
            metadata={"platform_id": ["GPL2011"], "channel_count": ["2"]},
        ),
        "GSM2": FakeGSM(
            pd.DataFrame({"ID_REF": ["p1", "p2"], "ch1 Intensity": [11.0, 21.0], "ch2 Intensity": [101.0, 201.0]}),
            metadata={"platform_id": ["GPL2011"], "channel_count": ["2"]},
        ),
    })

    channel1, channel2 = probe_mapping.build_channel_probe_matrices(gse)

    assert list(channel1.columns) == ["GSM1", "GSM2"]
    assert channel1.loc["p1", "GSM1"] == 10.0
    assert channel2.loc["p2", "GSM2"] == 201.0


def test_build_channel_probe_matrices_skips_single_channel_samples():
    gse = FakeGSE({
        "GSM1": FakeGSM(
            pd.DataFrame({"ID_REF": ["p1"], "VALUE": [5.0]}),
            metadata={"platform_id": ["GPL96"], "channel_count": ["1"]},
        ),
    })
    channel1, channel2 = probe_mapping.build_channel_probe_matrices(gse)
    assert channel1.empty
    assert channel2.empty


def test_build_channel_probe_matrices_skips_two_channel_samples_without_detectable_columns():
    """Most two-channel series only ever publish the ratio -- nothing to split."""
    gse = FakeGSE({
        "GSM1": FakeGSM(
            pd.DataFrame({"ID_REF": ["p1"], "VALUE": [0.5]}),
            metadata={"platform_id": ["GPL887"], "channel_count": ["2"]},
        ),
    })
    channel1, channel2 = probe_mapping.build_channel_probe_matrices(gse)
    assert channel1.empty
    assert channel2.empty
