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


class FakeGSM:
    def __init__(self, table):
        self.table = table
        self.metadata = {"platform_id": ["GPL96"]}


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


def test_aggregate_probes_to_genes_averages_multiple_probes_for_same_gene():
    probe_matrix = pd.DataFrame(
        {"GSM1": [656.6, 320.8, 361.3], "GSM2": [700.1, 310.2, 400.0]},
        index=["1007_s_at", "1053_at", "117_at"],
    )
    probe_gene_map = pd.DataFrame([
        {"probe_id": "1007_s_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
        {"probe_id": "1053_at", "gene_symbol": "RFC2", "entrez_id": "5982", "source": "direct_columns"},
        {"probe_id": "117_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
    ])
    genes = probe_mapping.aggregate_probes_to_genes(probe_matrix, probe_gene_map)
    ddr1 = genes[genes["gene_symbol"] == "DDR1"].iloc[0]
    assert ddr1["GSM1"] == (656.6 + 361.3) / 2
    assert ddr1["GSM2"] == (700.1 + 400.0) / 2
    assert len(genes) == 2  # DDR1 + RFC2, not 3 rows


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
