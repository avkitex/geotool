import pandas as pd
import pytest

from geotool import gene_symbol_mapping as gsm


def test_strip_version():
    assert gsm.strip_version("ENSG00000000003.18") == "ENSG00000000003"
    assert gsm.strip_version("ENST00000000233.10") == "ENST00000000233"
    assert gsm.strip_version("ENSG00000223972") == "ENSG00000223972"  # already unversioned


def test_canonical_id_unwraps_kallisto_salmon_composite_header():
    """Real GSE264630 shape: "ENST00000456328.2|ENSG00000223972.5|OTTHUMG...
    |OTTHUMT...|transcript_name|gene_name|length|biotype" -- extract just
    the leading ENST field."""
    composite = "ENST00000456328.2|ENSG00000223972.5|OTTHUMG00000000961.2|OTTHUMT00000362751.1|DDX11L1-202|DDX11L1|1657|processed_transcript"
    assert gsm.canonical_id(composite) == "ENST00000456328"


def test_canonical_id_plain_id_unaffected():
    assert gsm.canonical_id("ENSG00000000003.18") == "ENSG00000000003"
    assert gsm.canonical_id("TSPAN6") == "TSPAN6"


def test_undouble_repeated_symbol_ids_strips_majority_doubled_shape():
    """Real GSE174615 shape: RSEM gene_id column doubles the symbol as its
    own id -- "ACCSL_ACCSL" -- or, for a multi-copy gene family sharing one
    symbol, "<symbol>_<copy index>_<symbol>" ("5S_rRNA_10_5S_rRNA")."""
    ids = ["ACCSL_ACCSL", "5S_rRNA_10_5S_rRNA", "TSPAN6_TSPAN6"]
    result = gsm.undouble_repeated_symbol_ids(ids)
    assert list(result) == ["ACCSL", "5S_rRNA", "TSPAN6"]


def test_undouble_repeated_symbol_ids_leaves_minority_shape_unchanged():
    """Only unwraps when a majority of the values match the doubled shape --
    a matrix that's mostly plain symbols with one coincidentally
    self-repeating id must not have that id corrupted."""
    ids = ["TSPAN6", "TNMD", "A1BG", "FOO_FOO"]
    result = gsm.undouble_repeated_symbol_ids(ids)
    assert list(result) == ids


def test_undouble_repeated_symbol_ids_non_doubled_values_pass_through():
    """GSE174615's own non-doubled outliers (miRNA accessions, embedded-ENSG
    rows, clone IDs) must pass through unchanged even when the majority of
    the axis does unwrap."""
    ids = ["ACCSL_ACCSL", "TSPAN6_TSPAN6", "TNMD_TNMD", "MI0000060_hsa-let-7a-1"]
    result = gsm.undouble_repeated_symbol_ids(ids)
    assert list(result) == ["ACCSL", "TSPAN6", "TNMD", "MI0000060_hsa-let-7a-1"]


def make_reference(
    gene_to_symbol=None, transcript_to_symbol=None, gene_length=None,
) -> gsm.GencodeReference:
    gene_to_symbol = gene_to_symbol or {}
    transcript_to_symbol = transcript_to_symbol or {}
    known = frozenset(gene_to_symbol.values()) | frozenset(transcript_to_symbol.values())
    return gsm.GencodeReference("50", transcript_to_symbol, gene_to_symbol, gene_length or {}, known)


# --- load_gencode_reference ---------------------------------------------

def _write_tsv_gz(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def test_load_gencode_reference_full_set(tmp_path):
    base = tmp_path / "gencode50"
    _write_tsv_gz(base / "ensg2hugo_gencode_v50.tsv.gz", pd.DataFrame({
        "ID": ["ENSG00000000003.18", "ENSG00000000005.7"], "Gene": ["TSPAN6", "TNMD"],
    }))
    _write_tsv_gz(base / "clean_transcript_gene_symbol_v50.tsv.gz", pd.DataFrame({
        "transcript_id": ["ENST00000373020.9", "ENST00000373031.5"],
        "gene_id": ["ENSG00000000003.18", "ENSG00000000005.7"],
        "gene_symbol": ["TSPAN6", "TNMD"],
    }))
    _write_tsv_gz(base / "transcript_annotation_v50.tsv.gz", pd.DataFrame({
        "transcript_id": ["ENST00000373020.9", "ENST00000373031.5", "ENST00000zzzzz.1"],
        "gene_id": ["ENSG00000000003.18", "ENSG00000000003.18", "ENSG00000000005.7"],
        "transcript_length": [2206, 2100, 500],
        "included": [True, True, False],
    }))

    ref = gsm.load_gencode_reference("50", references_dir=tmp_path)

    assert ref.gene_to_symbol == {"ENSG00000000003": "TSPAN6", "ENSG00000000005": "TNMD"}
    assert ref.transcript_to_symbol == {"ENST00000373020": "TSPAN6", "ENST00000373031": "TNMD"}
    # median of [2206, 2100] (excluded-length 500 not counted, included=False)
    assert ref.gene_length == {"ENSG00000000003": 2153}
    assert "TSPAN6" in ref.known_symbols


def test_load_gencode_reference_degrades_gracefully_without_clean_or_length_files(tmp_path):
    """gencode32 shape: only the raw id2gene/ensg2hugo files exist."""
    base = tmp_path / "gencode32"
    _write_tsv_gz(base / "ensg2hugo_gencode_v32.tsv.gz", pd.DataFrame({
        "ID": ["ENSG00000000003.14"], "Gene": ["TSPAN6"],
    }))
    _write_tsv_gz(base / "id2gene_gencode_v32.tsv.gz", pd.DataFrame({
        "ID": ["ENST00000373020.8"], "Gene": ["TSPAN6"],
    }))

    ref = gsm.load_gencode_reference("32", references_dir=tmp_path)

    assert ref.gene_to_symbol == {"ENSG00000000003": "TSPAN6"}
    assert ref.transcript_to_symbol == {"ENST00000373020": "TSPAN6"}  # from the raw, unfiltered file
    assert ref.gene_length == {}  # no length data available for this version


# --- detect_identifier_type / locate_identifier_axis ---------------------

_REF = make_reference(
    gene_to_symbol={"ENSG00000000003": "TSPAN6", "ENSG00000000005": "TNMD"},
    transcript_to_symbol={"ENST00000373020": "TSPAN6", "ENST00000373031": "TNMD"},
    gene_length={"ENSG00000000003": 2200, "ENSG00000000005": 1200},
)


def test_detect_identifier_type_transcript():
    assert gsm.detect_identifier_type(["ENST00000373020.9", "ENST00000373031.5"], _REF) == "transcript"


def test_detect_identifier_type_composite_kallisto_header():
    """Real GSE264630 shape: pipe-delimited Kallisto/Salmon transcript
    FASTA headers -- still recognized as transcript IDs."""
    composite = "ENST00000373020.9|ENSG00000000003.18|OTTHUMG1|OTTHUMT1|TSPAN6-001|TSPAN6|2206|protein_coding"
    assert gsm.detect_identifier_type([composite, composite], _REF) == "transcript"


def test_detect_identifier_type_gene():
    assert gsm.detect_identifier_type(["ENSG00000000003.18", "ENSG00000000005.7"], _REF) == "gene"


def test_detect_identifier_type_symbol():
    assert gsm.detect_identifier_type(["TSPAN6", "TNMD"], _REF) == "symbol"


def test_detect_identifier_type_unknown():
    """Real GSE163305 shape: Cufflinks XLOC_ novel-locus IDs have no stable
    cross-reference to Ensembl at all -- must not be guessed at."""
    assert gsm.detect_identifier_type(["XLOC_000001", "XLOC_000002"], _REF) == "unknown"


def test_detect_identifier_type_symbol_below_min_id_match_fraction_still_recognized():
    """Real GSE273376 shape: a genuine, submitter-supplied gene-symbol list
    only ever partially overlaps GENCODE's own current nomenclature (~58%
    live-measured against real data) -- well below MIN_ID_MATCH_FRACTION
    (0.5, meant for exact ENST/ENSG regex matches) but should still clear
    the deliberately lower MIN_SYMBOL_MATCH_FRACTION."""
    ref = make_reference(gene_to_symbol={f"ENSG{i:011d}": f"GENE{i}" for i in range(10)})
    ids = [f"GENE{i}" for i in range(3)] + [f"UNMATCHED{i}" for i in range(7)]  # 30% known
    assert gsm.detect_identifier_type(ids, ref) == "symbol"


def test_detect_identifier_type_samples_randomly_not_just_the_first_n(monkeypatch):
    """Real bug found live: sampling literally the first N rows of a sorted
    file (e.g. alphabetical) can land in an unrepresentative run and
    misclassify an otherwise-clearly-symbol column as unknown. A large,
    unsorted-in-the-relevant-sense list where only the *tail* half is known
    symbols must still classify correctly once sampled from throughout."""
    monkeypatch.setattr(gsm, "_CLASSIFY_SAMPLE_SIZE", 100)
    ref = make_reference(gene_to_symbol={f"ENSG{i:011d}": f"GENE{i}" for i in range(300)})
    ids = [f"UNMATCHED{i}" for i in range(700)] + [f"GENE{i}" for i in range(300)]
    assert gsm.detect_identifier_type(ids, ref) == "symbol"


def test_locate_identifier_axis_prefers_index_when_recognizable():
    matrix = pd.DataFrame({"GSM1": [1.0, 2.0]}, index=["ENSG00000000003.18", "ENSG00000000005.7"])
    ids, id_type = gsm.locate_identifier_axis(matrix, _REF)
    assert id_type == "gene"
    assert list(ids) == ["ENSG00000000003.18", "ENSG00000000005.7"]


def test_locate_identifier_axis_finds_column_when_index_is_default():
    matrix = pd.DataFrame({"gene_id": ["ENSG00000000003.18", "ENSG00000000005.7"], "GSM1": [1.0, 2.0]})
    ids, id_type = gsm.locate_identifier_axis(matrix, _REF)
    assert id_type == "gene"


def test_locate_identifier_axis_prefers_gene_id_over_already_symbol_column():
    """Real GSE240145 shape: both "Gene_stable_ID" (ENSG) and "Gene_name"
    (already symbols) are present -- the canonical GENCODE mapping should
    win, for consistent naming across cohorts, not whatever the submitter's
    own symbol column happened to say."""
    matrix = pd.DataFrame({
        "Gene_stable_ID": ["ENSG00000000003", "ENSG00000000005"],
        "Gene_name": ["TSPAN6", "TNMD"],
        "GSM1": [1.0, 2.0],
    })
    ids, id_type = gsm.locate_identifier_axis(matrix, _REF)
    assert id_type == "gene"
    assert list(ids) == ["ENSG00000000003", "ENSG00000000005"]


def test_locate_identifier_axis_recognizes_doubled_rsem_symbols():
    """Real GSE174615 shape: without undoubling, "ACCSL_ACCSL" et al. match
    neither ENST/ENSG nor any known symbol, so the whole axis used to read
    as unrecognizable and the cohort was skipped outright."""
    matrix = pd.DataFrame(
        {"GSM1": [1.0, 2.0]}, index=["TSPAN6_TSPAN6", "TNMD_TNMD"],
    )
    ids, id_type = gsm.locate_identifier_axis(matrix, _REF)
    assert id_type == "symbol"
    assert list(ids) == ["TSPAN6", "TNMD"]


def test_locate_identifier_axis_none_when_nothing_recognizable():
    matrix = pd.DataFrame({"gene_id": ["XLOC_000001", "XLOC_000002"], "GSM1": [1.0, 2.0]})
    assert gsm.locate_identifier_axis(matrix, _REF) is None


# --- convert_to_gene_symbols ----------------------------------------------

def test_convert_to_gene_symbols_from_transcripts_sums_by_gene():
    """Two transcripts of the same gene (TSPAN6) must sum, not just rename."""
    matrix = pd.DataFrame(
        {"GSM1": [10.0, 5.0], "GSM2": [20.0, 8.0]},
        index=["ENST00000373020.9", "ENST00000new.1"],
    )
    ref = make_reference(
        transcript_to_symbol={"ENST00000373020": "TSPAN6", "ENST00000new": "TSPAN6"},
    )
    result, note = gsm.convert_to_gene_symbols(matrix, ref)
    assert result is not None
    assert len(result) == 1
    row = result.iloc[0]
    assert row["gene_symbol"] == "TSPAN6"
    assert row["GSM1"] == 15.0
    assert row["GSM2"] == 28.0
    assert "converted from transcript" in note


def test_convert_to_gene_symbols_from_composite_kallisto_headers():
    """Real GSE264630 shape: pipe-delimited transcript headers must still
    map and sum correctly, not fall through to "unrecognized"."""
    matrix = pd.DataFrame(
        {"GSM1": [10.0, 5.0]},
        index=[
            "ENST00000373020.9|ENSG00000000003.18|OTTHUMG1|OTTHUMT1|TSPAN6-001|TSPAN6|2206|protein_coding",
            "ENST00000373031.5|ENSG00000000003.18|OTTHUMG1|OTTHUMT2|TSPAN6-002|TSPAN6|1900|protein_coding",
        ],
    )
    ref = make_reference(transcript_to_symbol={"ENST00000373020": "TSPAN6", "ENST00000373031": "TSPAN6"})
    result, note = gsm.convert_to_gene_symbols(matrix, ref)
    assert result is not None
    assert list(result["gene_symbol"]) == ["TSPAN6"]
    assert result.iloc[0]["GSM1"] == 15.0


def test_convert_to_gene_symbols_from_genes_direct_mapping():
    matrix = pd.DataFrame(
        {"GSM1": [10.0, 20.0]}, index=["ENSG00000000003.18", "ENSG00000000005.7"]
    )
    result, note = gsm.convert_to_gene_symbols(matrix, _REF)
    assert set(result["gene_symbol"]) == {"TSPAN6", "TNMD"}
    assert "converted from gene" in note


def test_convert_to_gene_symbols_already_symbols_passthrough_still_collapses_duplicates():
    """Real risk called out by design: data that's nominally "already gene
    symbols" but wasn't actually pre-aggregated (duplicate symbol rows) must
    still be summed, not just accepted as-is."""
    matrix = pd.DataFrame({"GSM1": [10.0, 5.0]}, index=["TSPAN6", "TSPAN6"])
    result, note = gsm.convert_to_gene_symbols(matrix, _REF)
    assert len(result) == 1
    assert result.iloc[0]["GSM1"] == 15.0
    assert "already gene symbols" in note
    assert "collapsed 1 duplicate" in note


def test_convert_to_gene_symbols_drops_unmapped_rows():
    matrix = pd.DataFrame(
        {"GSM1": [10.0, 5.0]}, index=["ENSG00000000003.18", "ENSG00000nomatch.1"]
    )
    result, note = gsm.convert_to_gene_symbols(matrix, _REF)
    assert list(result["gene_symbol"]) == ["TSPAN6"]
    assert "1 unmapped row(s) dropped" in note


def test_convert_to_gene_symbols_refuses_negative_values():
    matrix = pd.DataFrame({"GSM1": [-1.0, 5.0]}, index=["ENSG00000000003.18", "ENSG00000000005.7"])
    result, note = gsm.convert_to_gene_symbols(matrix, _REF)
    assert result is None
    assert "negative values" in note


def test_convert_to_gene_symbols_refuses_unrecognized_identifiers():
    matrix = pd.DataFrame({"GSM1": [10.0, 5.0]}, index=["XLOC_000001", "XLOC_000002"])
    result, note = gsm.convert_to_gene_symbols(matrix, _REF)
    assert result is None
    assert "no recognizable" in note


def test_convert_to_gene_symbols_no_numeric_columns():
    matrix = pd.DataFrame({"gene_id": ["ENSG00000000003.18"], "notes": ["x"]})
    result, note = gsm.convert_to_gene_symbols(matrix, _REF)
    assert result is None
    assert "no numeric" in note


# --- compute_tpm -----------------------------------------------------------

def test_compute_tpm_sums_to_one_million_per_sample():
    matrix = pd.DataFrame({
        "gene_symbol": ["TSPAN6", "TNMD"],
        "GSM1": [100.0, 50.0],
        "GSM2": [10.0, 200.0],
    })
    result, note = gsm.compute_tpm(matrix, _REF)
    assert result is not None
    assert note == "TPM computed"
    for col in ("GSM1", "GSM2"):
        assert result[col].sum() == pytest.approx(1_000_000, rel=1e-6)


def test_compute_tpm_excludes_genes_without_length_data():
    matrix = pd.DataFrame({
        "gene_symbol": ["TSPAN6", "UNKNOWN_GENE"],
        "GSM1": [100.0, 50.0],
    })
    result, note = gsm.compute_tpm(matrix, _REF)
    assert list(result["gene_symbol"]) == ["TSPAN6"]
    assert "1 gene(s) without length data excluded" in note


def test_compute_tpm_none_when_reference_has_no_length_data():
    ref = make_reference(gene_to_symbol={"ENSG00000000003": "TSPAN6"}, gene_length={})
    matrix = pd.DataFrame({"gene_symbol": ["TSPAN6"], "GSM1": [100.0]})
    result, note = gsm.compute_tpm(matrix, ref)
    assert result is None
    assert "no gene-length data available" in note
