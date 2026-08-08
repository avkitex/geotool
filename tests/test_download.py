import gzip
import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from geotool import clinical_annotate, config, download, gene_symbol_mapping


def _big_gene_index(n: int = config.MIN_EXPECTED_RNASEQ_GENE_COUNT) -> list[str]:
    """Enough rows that a test's small handful of real values don't
    incidentally trip probe_mapping.check_gene_count's truncated-gene-list
    heuristic when that's not what the test is exercising."""
    return [f"GENE{i}" for i in range(n)]


@pytest.fixture(autouse=True)
def _fake_gene_reference(monkeypatch):
    """_content_verified_column_count's gene-identity check needs a real
    gene_symbol_mapping.GencodeReference; tests use synthetic "GENE{i}"
    identifiers (see _big_gene_index/_big_matrix_dataframe) rather than
    depending on the real, multi-MB-on-disk GENCODE tables -- this
    autouse fixture recognizes that exact convention (well beyond any
    n_genes used in this file) so every test gets it for free instead of
    each one loading/mocking a reference individually. A test exercising
    the negative case (a genuinely non-gene identifier axis) still gets a
    real rejection: this reference simply doesn't know that axis either."""
    known = frozenset(f"GENE{i}" for i in range(config.MIN_EXPECTED_RNASEQ_GENE_COUNT * 2))
    reference = gene_symbol_mapping.GencodeReference("test", {}, {}, {}, known)
    monkeypatch.setattr(download, "_default_gene_reference", lambda: reference)


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


def test_select_primary_expression_file_prioritizes_tpm_over_fpkm_and_cpm():
    paths = [Path("GSE1_counts.txt.gz"), Path("GSE1_CPM.txt.gz"), Path("GSE1_FPKM.csv.gz"), Path("GSE1_TPM.tsv.gz")]
    assert download.select_primary_expression_file(paths) == (Path("GSE1_TPM.tsv.gz"), "tpm")


def test_select_primary_expression_file_prioritizes_fpkm_over_cpm():
    paths = [Path("GSE1_CPM.txt.gz"), Path("GSE1_FPKM.csv.gz")]
    assert download.select_primary_expression_file(paths) == (Path("GSE1_FPKM.csv.gz"), "fpkm")


def test_select_primary_expression_file_recognizes_rpkm():
    """Real GSE246325 shape: "..._prmtRPKM.csv.gz" -- RPKM is FPKM's
    single-end-read equivalent, ranked alongside it."""
    paths = [Path("GSE1_CPM.txt.gz"), Path("GSE246325_prmtRPKM.csv.gz")]
    assert download.select_primary_expression_file(paths) == (Path("GSE246325_prmtRPKM.csv.gz"), "rpkm")


def test_select_primary_expression_file_recognizes_singular_count():
    """Real GSE273376 shape: "..._count_matrix.csv.gz" -- the old "counts"
    (plural-only) keyword missed this singular naming convention entirely."""
    paths = [Path("GSE273376_pancreas_cell_lines.AM9747.count_matrix.csv.gz")]
    assert download.select_primary_expression_file(paths) == (paths[0], "count")


def test_select_primary_expression_file_excludes_non_data_extensions():
    """Real GSE161706 shape: "..._dexseq_count.py.gz" is a compressed Python
    *script*, not data -- it must not be picked just because its name
    happens to contain "count"."""
    paths = [
        Path("GSE161706_dexseq_count.py.gz"),
        Path("GSE161706_DEXseq.R.gz"),
        Path("GSE161706_Processed_data_for_Table_S3_Figure_6.xlsx"),
    ]
    # The only actual data file has no recognizable unit keyword either --
    # correctly nothing is picked, not the script.
    assert download.select_primary_expression_file(paths) is None


def test_select_primary_expression_file_excludes_derived_comparison_files():
    """Real GSE194360/GSE194362 shape: "..._snp_counts_significance.csv.gz"
    matches the "count" unit keyword but is a differential-significance
    table derived from expression data, not a per-gene-per-sample matrix."""
    paths = [
        Path("GSE194360_Sample_A_vs_Control_snp_counts_significance.csv.gz"),
        Path("GSE194360_Sample_B_vs_Control_snp_counts_significance.csv.gz"),
    ]
    assert download.select_primary_expression_file(paths) is None


def test_select_primary_expression_file_excludes_original_backup_files():
    """A "<name>.original.tsv.gz" backup sitting next to the real file it
    was copied from (e.g. geotool.gene_symbol_mapping's in-place replacement
    convention) must never compete with it -- live bug: both matched the
    same "count" unit and tied, so whichever happened to come first in
    filesystem listing order won nondeterministically."""
    real = Path("GSE1_raw_counts.txt.gz")
    backup = Path("GSE1_raw_counts.original.tsv.gz")
    assert download.select_primary_expression_file([backup, real]) == (real, "count")
    assert download.select_primary_expression_file([real, backup]) == (real, "count")


def test_select_primary_expression_file_none_when_no_recognizable_unit():
    """Real GSE163305 shape: two rMATS splicing-analysis files and a
    differential-expression results table -- none is a gene-expression
    quantification matrix."""
    paths = [
        Path("GSE163305_D6.RI.MATS.JC.txt.gz"),
        Path("GSE163305_D6.SE.MATS.JC.txt.gz"),
        Path("GSE163305_GSK_vs_DMSO_D6.csv.gz"),
    ]
    assert download.select_primary_expression_file(paths) is None


def test_select_primary_expression_file_finds_fpkm_among_non_matrix_files():
    """The real GSE163305 supplementary file set: only the FPKM file is an
    actual expression matrix among 4 files."""
    paths = [
        Path("GSE163305_D6.RI.MATS.JC.txt.gz"),
        Path("GSE163305_D6.SE.MATS.JC.txt.gz"),
        Path("GSE163305_FPKM_6D_GSK6_DMSO.csv.gz"),
        Path("GSE163305_GSK_vs_DMSO_D6.csv.gz"),
    ]
    assert download.select_primary_expression_file(paths) == (Path("GSE163305_FPKM_6D_GSK6_DMSO.csv.gz"), "fpkm")


def test_select_primary_expression_file_excludes_gsm_named_per_sample_files():
    """Real GSE236498/GSE236499 shape: 12 "GSM*_gene_counts.txt.gz" files,
    one per sample, no combined matrix -- none of these may be picked as if
    one of them were the whole cohort's matrix, even though each matches the
    "counts" keyword."""
    paths = [
        Path("GSM7548847_MCF-7_WT1_gene_counts.txt.gz"),
        Path("GSM7548848_MCF-7_WT2_gene_counts.txt.gz"),
        Path("GSM7548849_MCF-7_WT3_gene_counts.txt.gz"),
    ]
    assert download.select_primary_expression_file(paths) is None


def test_select_primary_expression_file_still_finds_combined_matrix_among_gsm_named_files():
    paths = [
        Path("GSM7548847_MCF-7_WT1_gene_counts.txt.gz"),
        Path("GSM7548848_MCF-7_WT2_gene_counts.txt.gz"),
        Path("GSE236498_combined_TPM.txt.gz"),
    ]
    assert download.select_primary_expression_file(paths) == (Path("GSE236498_combined_TPM.txt.gz"), "tpm")


def test_check_rnaseq_expression_qc_flags_linear_scale_fpkm_matrix(tmp_path):
    """Real GSE163305 FPKM matrix shape: nonnegative, max value in the
    thousands. Source is .csv.gz (comma-separated), which needs converting
    to the guaranteed final .tsv.gz format -- see resolve_primary_expression_
    matrix's dedicated tests for that conversion behavior in isolation.
    resolve_primary_expression_matrix now auto-fixes linear-scale values
    (probe_mapping.normalize_expression_matrix) rather than just flagging
    them, so by the time check_expression_qc runs on the written file
    there's nothing left to report."""
    path = tmp_path / "GSE163305_FPKM_6D_GSK6_DMSO.csv.gz"
    genes = _big_gene_index()
    filler = [0.0] * (len(genes) - 2)
    pd.DataFrame(
        {"GSK6D_0": [0.0, 16096.1] + filler, "DMSO6D_0": [0.0, 12677.3] + filler}, index=genes
    ).to_csv(path, compression="gzip")

    primary_path, unit, notes = download.check_rnaseq_expression_qc([path], tmp_path)

    assert primary_path == tmp_path / "GSE163305_FPKM_6D_GSK6_DMSO.tsv.gz"
    assert unit == "fpkm"
    assert notes == []
    written = pd.read_csv(primary_path, sep="\t", compression="gzip")
    assert written["GSK6D_0"].max() == pytest.approx(np.log2(16096.1 + 1), abs=1e-3)


def test_check_rnaseq_expression_qc_flags_negative_values(tmp_path):
    """The RNA-seq-specific risk called out by design: log2(x) applied
    without a +1 pseudocount goes negative for x in (0, 1)."""
    path = tmp_path / "GSE1_TPM.csv.gz"
    genes = _big_gene_index()
    pd.DataFrame({"GSM1": [-1.2, 3.0] + [0.0] * (len(genes) - 2)}, index=genes).to_csv(path, compression="gzip")

    primary_path, unit, notes = download.check_rnaseq_expression_qc([path], tmp_path)

    assert unit == "tpm"
    assert len(notes) == 1
    assert "negative value" in notes[0]


def test_check_rnaseq_expression_qc_marks_truncated_gene_list(tmp_path):
    """Real GSE197728 shape: only genes with FPKM > 10 were reported (7833
    rows there; a small count here for a fast test) -- the primary file
    gets renamed with a ".truncated" marker since there's no way to
    auto-fix genes that were never published."""
    path = tmp_path / "GSE197728_counts.csv.gz"
    genes = [f"GENE{i}" for i in range(100)]
    pd.DataFrame({"GSM1": range(100), "GSM2": range(100)}, index=genes).to_csv(path, compression="gzip")

    primary_path, unit, notes = download.check_rnaseq_expression_qc([path], tmp_path)

    assert primary_path == tmp_path / "GSE197728_counts.truncated.tsv.gz"
    assert primary_path.exists()
    assert not (tmp_path / "GSE197728_counts.tsv.gz").exists()
    assert len(notes) == 1
    assert "only 100 genes" in notes[0] and "filtered/truncated" in notes[0]


def test_check_rnaseq_expression_qc_overwrites_stale_truncated_file_from_earlier_run(tmp_path):
    """Real GSE197728 shape: a repeat --force run re-resolves the primary
    file back to its plain (non-.truncated) name, but the .truncated.tsv.gz
    destination from an *earlier* run is still sitting on disk -- must
    overwrite it, not fail. Live-broke on Windows before Path.replace()
    (Path.rename() only overwrites an existing destination on POSIX)."""
    path = tmp_path / "GSE197728_counts.csv.gz"
    genes = [f"GENE{i}" for i in range(100)]
    pd.DataFrame({"GSM1": range(100), "GSM2": range(100)}, index=genes).to_csv(path, compression="gzip")
    stale = tmp_path / "GSE197728_counts.truncated.tsv.gz"
    stale.write_bytes(b"stale content from an earlier run")

    primary_path, unit, notes = download.check_rnaseq_expression_qc([path], tmp_path)

    assert primary_path == stale
    assert primary_path.read_bytes() != b"stale content from an earlier run"
    assert len(notes) == 1


def test_check_rnaseq_expression_qc_does_not_double_mark_already_truncated_file(tmp_path):
    path = tmp_path / "GSE1_counts.truncated.tsv.gz"
    genes = [f"GENE{i}" for i in range(100)]
    pd.DataFrame({"GSM1": range(100)}, index=genes).to_csv(path, sep="\t", compression="gzip")

    primary_path, unit, notes = download.check_rnaseq_expression_qc([path], tmp_path)

    assert primary_path == path  # not renamed again to ".truncated.truncated.tsv.gz"
    assert len(notes) == 1


def test_check_rnaseq_expression_qc_returns_nothing_when_no_matrix_file(tmp_path):
    paths = [tmp_path / "some_de_table.csv.gz"]
    primary_path, unit, notes = download.check_rnaseq_expression_qc(paths, tmp_path)
    assert primary_path is None
    assert unit is None
    assert notes == []


def test_check_rnaseq_expression_qc_skips_featurecounts_comment_line(tmp_path):
    """Real GSE264630 shape: featureCounts writes a leading '# Program:
    featureCounts ...' line before the real header row. Without skipping it,
    that comment line is read as the header, and the real header row
    (Geneid, Chr, Start, ..., sample columns) is misread as a data row --
    silently destroying every column name and shifting every value down one
    row. Live-verified: this corrupted a real cohort's primary file in place
    before comment="#" was added to _load_expression_file_for_qc."""
    path = tmp_path / "GSE264630_counts.txt.gz"
    genes = _big_gene_index()
    lines = ['# Program:featureCounts v2.0.1; Command:"featureCounts" "-a" "ref.gtf" "-o" "counts.txt" ...']
    lines.append("Geneid\tChr\tStart\tEnd\tStrand\tLength\tGSM1\tGSM2")
    for i, gene in enumerate(genes):
        lines.append(f"{gene}\t1\t{i}\t{i + 1}\t+\t100\t{i}\t{i * 2}")
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")

    primary_path, unit, notes = download.check_rnaseq_expression_qc([path], tmp_path)

    assert unit == "count"
    written = pd.read_csv(primary_path, sep="\t")
    assert list(written.columns) == ["Geneid", "GSM1", "GSM2"]
    assert written["Geneid"].tolist() == genes
    # values are log2(x + 1)-transformed by normalize_expression_matrix -- the
    # point of this test is the columns/rows survived intact, not the scale.
    assert written["GSM1"].iloc[10] == pytest.approx(np.log2(10 + 1), abs=1e-3)


def test_check_rnaseq_expression_qc_notes_unparseable_file(tmp_path):
    path = tmp_path / "broken_tpm.csv.gz"
    path.write_bytes(b"not actually gzip-compressed data")
    primary_path, unit, notes = download.check_rnaseq_expression_qc([path], tmp_path)
    # Couldn't be converted either -- falls back to reporting the original.
    assert primary_path == path
    assert unit == "tpm"
    assert len(notes) == 1
    assert "could not parse" in notes[0]


def test_check_rnaseq_expression_qc_reads_xlsx_files(tmp_path):
    """Real shape: several PRMT5/MTAP cohorts (e.g. GSE310927, GSE277490)
    publish their combined matrix as an .xlsx file rather than a
    delimited/gzipped text file -- converted to .tsv.gz, per the guaranteed
    final format. Linear-scale values get auto-fixed (see the .csv.gz test
    above), so there's nothing left for check_expression_qc to flag."""
    path = tmp_path / "GSE310927_L3.6_EPZ_Prex_Combo_CPM.xlsx"
    genes = _big_gene_index()
    filler = [0.0] * (len(genes) - 2)
    pd.DataFrame(
        {"GSM1": [0.0, 999.0] + filler, "GSM2": [1.0, 500.0] + filler}, index=genes
    ).to_excel(path)

    primary_path, unit, notes = download.check_rnaseq_expression_qc([path], tmp_path)

    assert primary_path == tmp_path / "GSE310927_L3.6_EPZ_Prex_Combo_CPM.tsv.gz"
    assert unit == "cpm"
    assert notes == []
    written = pd.read_csv(primary_path, sep="\t", compression="gzip")
    assert written["GSM1"].max() == pytest.approx(np.log2(999.0 + 1), abs=1e-3)


# --- select_primary_expression_file_by_content: content-based fallback for a
# real matrix whose filename carries no recognizable unit keyword at all ---

def _big_matrix_dataframe(n_genes: int, n_samples: int) -> pd.DataFrame:
    """A synthetic matrix big enough (_MIN_MATRIX_ROWS) to pass the
    "is this even plausibly a real expression matrix" row-count sanity
    check, with a caller-controlled sample-column count for the
    column-count-vs-n_samples verification under test."""
    return pd.DataFrame(
        {f"GSM{i}": [1.0] * n_genes for i in range(n_samples)},
        index=[f"GENE{i}" for i in range(n_genes)],
    )


def test_select_primary_expression_file_by_content_verifies_sole_candidate(tmp_path):
    """Real GSE253260 shape: sole file "..._BACAP.rawct.tsv.gz" -- "rawct"
    doesn't literally contain "count", so the filename-only path finds
    nothing, but its column count (60671 genes x 397 samples) matches the
    cohort's real sample count."""
    path = tmp_path / "GSE253260_BACAP.rawct.tsv.gz"
    _big_matrix_dataframe(1200, 397).to_csv(path, sep="\t", compression="gzip")

    result = download.select_primary_expression_file_by_content([path], n_samples=397)

    assert result == ([path], "unknown")


def test_select_primary_expression_file_by_content_rejects_table_excerpt(tmp_path):
    """Real GSE161706 shape: the sole remaining file after excluding script
    files, "..._Processed_data_for_Table_S3_Figure_6.xlsx", parses to a
    single descriptive text cell -- 0 data rows -- not the whole-cohort
    matrix, even though it's the only candidate left. Must stay rejected
    even with n_samples given, exactly like the filename-only path already
    (correctly) rejects it."""
    path = tmp_path / "GSE161706_Processed_data_for_Table_S3_Figure_6.xlsx"
    pd.DataFrame({"note": ["a description sentence, not gene data"]}).to_excel(path, index=False)

    assert download.select_primary_expression_file_by_content([path], n_samples=12) is None


def test_select_primary_expression_file_by_content_rejects_column_count_mismatch(tmp_path):
    """A real, big-enough matrix whose column count has nothing to do with
    the cohort's sample count must not be picked just for being the sole
    remaining candidate."""
    path = tmp_path / "GSE1_some_other_cohorts_matrix.tsv.gz"
    _big_matrix_dataframe(1200, 4).to_csv(path, sep="\t", compression="gzip")

    assert download.select_primary_expression_file_by_content([path], n_samples=397) is None


def test_select_primary_expression_file_by_content_sums_multiple_candidates(tmp_path):
    """Real GSE293744 shape: two files, "Matrix-File1.txt.gz" (12 sample
    columns) and "Matrix-File2.txt.gz" (36), neither alone matches the
    cohort's 48 samples, but together they do -- two real batches, not
    noise, live-verified against the real files (13/37 total columns
    including one gene-ID column each, 12+36=48)."""
    path1 = tmp_path / "GSE293744_Matrix-File1.txt.gz"
    path2 = tmp_path / "GSE293744_Matrix-File2.txt.gz"
    _big_matrix_dataframe(1200, 12).to_csv(path1, sep="\t", compression="gzip")
    _big_matrix_dataframe(1200, 36).to_csv(path2, sep="\t", compression="gzip")

    result = download.select_primary_expression_file_by_content([path1, path2], n_samples=48)

    assert result is not None
    assert set(result[0]) == {path1, path2}
    assert result[1] == "unknown"


def test_select_primary_expression_file_by_content_gives_up_beyond_five_candidates():
    """More than 5 remaining candidates is too large a pool to confidently
    sum an arbitrary subset against the sample count -- summing some subset
    to hit the target by chance becomes a real risk rather than a signal."""
    paths = [Path(f"GSE1_batch{i}.tsv.gz") for i in range(6)]
    assert download.select_primary_expression_file_by_content(paths, n_samples=48) is None


# --- _content_verified_column_count: gene-identity + non-expression-genomic-
# file rejection, on top of the shape checks above ---

def test_content_verified_column_count_rejects_non_gene_identifier_axis(tmp_path):
    """Real GSE236496 shape: a ChIP-Seq peak-calls table (chr/start/ned/
    state/gene_chr/gene_start/gene_end/gene_id/gene_name/strand columns,
    thousands of rows) whose column count happened to fall within
    _matches_sample_count's tolerance of that cohort's real sample count --
    shape alone would have wrongly accepted it; its identifier axis (here,
    'chr') was never a real gene/transcript in the first place."""
    path = tmp_path / "GSE236496_Peak_calls.tsv.gz"
    n_rows = 2000
    pd.DataFrame({
        "chr": [f"chr{(i % 24) + 1}" for i in range(n_rows)],
        "start": range(n_rows),
        "end": range(1, n_rows + 1),
        "gene_id": [f"ENSG{i:011d}" for i in range(n_rows)],
        "gene_name": [f"GENE{i}" for i in range(n_rows)],
        "strand": ["+"] * n_rows,
    }).to_csv(path, sep="\t", index=False, compression="gzip")

    assert download._content_verified_column_count(path) is None


def test_content_verified_column_count_rejects_maf_shaped_file_despite_real_gene_symbols(tmp_path):
    """The harder case a gene-identity check alone can't catch: a MAF
    (Mutation Annotation Format) file's Hugo_Symbol column carries real
    HUGO gene symbols, so it would pass gene-identity verification on its
    own -- it's the MAF-specific column vocabulary (Variant_Classification,
    Tumor_Sample_Barcode, ...) that must independently reject it."""
    path = tmp_path / "GSE1_mutations.tsv.gz"
    n_rows = 2000
    pd.DataFrame({
        "Hugo_Symbol": [f"GENE{i}" for i in range(n_rows)],
        "Chromosome": [f"chr{(i % 24) + 1}" for i in range(n_rows)],
        "Start_Position": range(n_rows),
        "End_Position": range(1, n_rows + 1),
        "Variant_Classification": ["Missense_Mutation"] * n_rows,
        "Variant_Type": ["SNP"] * n_rows,
        "Reference_Allele": ["A"] * n_rows,
        "Tumor_Seq_Allele1": ["A"] * n_rows,
        "Tumor_Seq_Allele2": ["T"] * n_rows,
        "Tumor_Sample_Barcode": [f"Sample{i % 20}" for i in range(n_rows)],
    }).to_csv(path, sep="\t", index=False, compression="gzip")

    assert download._content_verified_column_count(path) is None


def test_content_verified_column_count_accepts_real_gene_matrix_with_incidental_start_column(tmp_path):
    """A real gene expression matrix that happens to have one column named
    like a BED/VCF field (e.g. a sample literally named "start") must not
    be rejected -- _MIN_SIGNATURE_KEYWORD_MATCHES requires several matching
    columns from the same signature, not just one incidental hit."""
    path = tmp_path / "GSE1_counts.tsv.gz"
    _big_matrix_dataframe(1200, 5).rename(columns={"GSM0": "start"}).to_csv(path, sep="\t", compression="gzip")

    assert download._content_verified_column_count(path) == 5


def test_looks_like_non_expression_genomic_file_detects_each_signature():
    assert download._looks_like_non_expression_genomic_file(["chr", "start", "end", "gene_id"])
    assert download._looks_like_non_expression_genomic_file(["#CHROM", "POS", "REF", "ALT", "QUAL", "FILTER"])
    assert download._looks_like_non_expression_genomic_file(
        ["Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode"]
    )


def test_looks_like_non_expression_genomic_file_ignores_incidental_single_match():
    assert not download._looks_like_non_expression_genomic_file(["gene_id", "start", "SAMPLE1", "SAMPLE2"])


def test_check_rnaseq_expression_qc_uses_content_verification_when_no_unit_keyword(tmp_path):
    """Integration: the whole check_rnaseq_expression_qc path picks up a
    real matrix by content when resolve_primary_expression_matrix's
    filename-only path finds nothing, given n_samples."""
    path = tmp_path / "GSE172356_PDA_gene_expression_matrix.txt.gz"
    _big_matrix_dataframe(config.MIN_EXPECTED_RNASEQ_GENE_COUNT, 62).to_csv(path, sep="\t", compression="gzip")

    primary_path, unit, notes = download.check_rnaseq_expression_qc([path], tmp_path, n_samples=62)

    assert primary_path == path
    assert unit == "unknown"


def test_check_rnaseq_expression_qc_notes_multi_file_sum_without_picking_primary(tmp_path):
    """Real GSE131050 shape: two files, neither individually the whole
    matrix, but their combined column count matches the sample count --
    reported as an informational note pointing at both files rather than
    silently guessing one of them is "the" primary (neither is)."""
    path1 = tmp_path / "GSE131050_PurIST_Linehan_seq.tsv.gz"
    path2 = tmp_path / "GSE131050_PurIST_Yeh_seq.tsv.gz"
    _big_matrix_dataframe(1200, 66).to_csv(path1, sep="\t", compression="gzip")
    _big_matrix_dataframe(1200, 125).to_csv(path2, sep="\t", compression="gzip")

    primary_path, unit, notes = download.check_rnaseq_expression_qc([path1, path2], tmp_path, n_samples=191)

    assert primary_path is None
    assert unit is None
    assert len(notes) == 1
    assert path1.name in notes[0] and path2.name in notes[0]
    assert "191" in notes[0]


def test_check_rnaseq_expression_qc_still_rejects_no_matrix_file_with_n_samples(tmp_path):
    """n_samples being given must not turn a genuinely-not-a-matrix file
    into a false positive -- same real GSE161706 shape as the filename-only
    regression test, but this time exercised with content verification
    active (n_samples given) to confirm it doesn't override the row-count
    floor."""
    path = tmp_path / "GSE1_some_table.csv.gz"
    pd.DataFrame({"note": ["not gene data"]}).to_csv(path, compression="gzip")

    primary_path, unit, notes = download.check_rnaseq_expression_qc([path], tmp_path, n_samples=12)

    assert primary_path is None
    assert unit is None


# --- resolve_primary_expression_matrix: the guaranteed-.tsv.gz-output fix ---

def _tsv_gz_dataframe():
    return pd.DataFrame({"GSM1": [1.0, 2.0], "GSM2": [3.0, 4.0]}, index=["GENE1", "GENE2"])


def test_resolve_primary_expression_matrix_leaves_already_tsv_gz_untouched(tmp_path):
    """A plain, already-conformant .tsv.gz file is returned as-is -- no new
    file written, nothing re-parsed/rewritten unnecessarily."""
    path = tmp_path / "GSE1_TPM.tsv.gz"
    _tsv_gz_dataframe().to_csv(path, sep="\t", compression="gzip")

    result = download.resolve_primary_expression_matrix([path], tmp_path)

    assert result == (path, "tpm")
    assert list(tmp_path.iterdir()) == [path]  # nothing new written


def test_resolve_primary_expression_matrix_converts_csv_to_tsv_gz(tmp_path):
    path = tmp_path / "GSE1_FPKM.csv.gz"
    _tsv_gz_dataframe().to_csv(path, compression="gzip")

    dest, unit = download.resolve_primary_expression_matrix([path], tmp_path)

    assert dest == tmp_path / "GSE1_FPKM.tsv.gz"
    assert unit == "fpkm"
    result = pd.read_csv(dest, sep="\t", index_col=0)
    assert result.loc["GENE1", "GSM1"] == 1.0


def test_resolve_primary_expression_matrix_converts_xlsx_to_tsv_gz(tmp_path):
    """Real shape: GSE310927/GSE277490/GSE197728/GSE161706 publish Excel."""
    path = tmp_path / "GSE1_CPM.xlsx"
    _tsv_gz_dataframe().to_excel(path)

    dest, unit = download.resolve_primary_expression_matrix([path], tmp_path)

    assert dest == tmp_path / "GSE1_CPM.tsv.gz"
    assert unit == "cpm"
    assert dest.name.lower().endswith(".tsv.gz")
    result = pd.read_csv(dest, sep="\t")
    assert result["GSM1"].tolist() == [1.0, 2.0]


def test_resolve_primary_expression_matrix_extracts_matching_member_from_zip(tmp_path):
    """Real-shaped scenario: a combined matrix published as a .zip archive
    rather than a standalone file."""
    zip_path = tmp_path / "GSE1_supplementary.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("GSE1_TPM_matrix.csv", _tsv_gz_dataframe().to_csv())
        zf.writestr("readme.txt", "not a matrix")

    dest, unit = download.resolve_primary_expression_matrix([zip_path], tmp_path)

    assert dest == tmp_path / "GSE1_TPM_matrix.tsv.gz"
    assert unit == "tpm"
    result = pd.read_csv(dest, sep="\t", index_col=0)
    assert result.loc["GENE1", "GSM1"] == 1.0


def test_resolve_primary_expression_matrix_extracts_matching_member_from_tar(tmp_path):
    """Real-shaped scenario: e.g. a "..._RAW.tar" that happens to also
    contain the one real combined matrix alongside unrelated per-sample
    fragments."""
    tar_path = tmp_path / "GSE1_RAW.tar"
    csv_bytes = _tsv_gz_dataframe().to_csv().encode()
    with tarfile.open(tar_path, "w") as tf:
        info = tarfile.TarInfo(name="GSE1_FPKM_combined.csv")
        info.size = len(csv_bytes)
        tf.addfile(info, io.BytesIO(csv_bytes))
        gsm_bytes = b"gene,value\nGENE1,5\n"
        info2 = tarfile.TarInfo(name="GSM1234567_sample.csv")
        info2.size = len(gsm_bytes)
        tf.addfile(info2, io.BytesIO(gsm_bytes))

    dest, unit = download.resolve_primary_expression_matrix([tar_path], tmp_path)

    assert dest == tmp_path / "GSE1_FPKM_combined.tsv.gz"
    assert unit == "fpkm"


def test_resolve_primary_expression_matrix_none_when_tar_only_has_per_sample_members(tmp_path):
    """Real GSE108651/GSE215847/GSE236498/GSE236499/GSE286560 shape:
    "..._RAW.tar" bundling only GSM-named per-sample fragments, no combined
    matrix -- correctly nothing is extracted or picked."""
    tar_path = tmp_path / "GSE1_RAW.tar"
    with tarfile.open(tar_path, "w") as tf:
        for i in range(1, 4):
            data = f"gene,count\nGENE1,{i}\n".encode()
            info = tarfile.TarInfo(name=f"GSM100000{i}_sample_counts.csv")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

    assert download.resolve_primary_expression_matrix([tar_path], tmp_path) is None


def test_resolve_primary_expression_matrix_extracts_already_tsv_member_without_reparsing(tmp_path):
    """An archive member that's already tab-separated is extracted verbatim
    (byte copy), not round-tripped through pandas."""
    zip_path = tmp_path / "GSE1_supplementary.zip"
    original_bytes = b"gene\tGSM1\tGSM2\nGENE1\t1.5\t2.5\n"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("GSE1_counts.tsv", original_bytes)

    dest, unit = download.resolve_primary_expression_matrix([zip_path], tmp_path)

    assert dest == tmp_path / "GSE1_counts.tsv"
    assert unit == "count"
    assert dest.read_bytes() == original_bytes


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

    expr_path, qc_notes = download.build_and_map_expression_matrix(gse, tmp_path)

    assert (tmp_path / "probe_matrix.tsv.gz").exists()
    assert expr_path == tmp_path / "expression.tsv.gz"
    # entrez_id already did its job as a multi-platform merge key (see
    # normalize_expression_matrix) -- dropped from the persisted file so it
    # can't be mistaken for a sample column by anything reading it back.
    assert len(qc_notes) == 1 and "entrez_id" in qc_notes[0]
    genes = pd.read_csv(expr_path, sep="\t")
    assert set(genes["gene_symbol"]) == {"DDR1", "RFC2"}
    assert "entrez_id" not in genes.columns


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


def make_agilent_two_channel_gse_with_reference(gse_id="GSE_AGILENT_REF", n_samples=6):
    """A common-reference-design fixture: channel 1 is a fixed reference RNA
    (metadata-labeled, constant raw values across samples) and channel 2 is
    the actual biological sample (metadata-labeled, values that vary per
    sample) -- shaped after the real GSE50470/GSE21997/GSE22049 pattern that
    motivated probe_mapping.detect_reference_channel.
    """
    metadata = {"geo_accession": [gse_id], "title": ["A reference-design two-channel series"], "summary": ["s"]}
    gsms = {}
    for i in range(1, n_samples + 1):
        gsms[f"GSM{i}"] = FakeGSM(
            {
                "title": [f"s{i}"], "geo_accession": [f"GSM{i}"], "platform_id": ["GPL2011"],
                "organism_ch1": ["Homo sapiens"], "channel_count": ["2"],
                "source_name_ch1": ["Human Universal Reference"],
                "characteristics_ch1": ["reference: Human Universal Reference"],
                "source_name_ch2": [f"Tumor sample {i}"],
                "characteristics_ch2": [f"tissue: Breast Cancer {i}"],
            },
            table=pd.DataFrame({
                "ID_REF": ["1007_s_at", "1053_at"],
                "ch1 Intensity": [100.0, 100.0],
                "ch2 Intensity": [50.0 * i, 5.0 * i],
                "VALUE": [1.0, 1.0],
            }),
        )
    gpls = {"GPL2011": FakeGPL({"title": ["Agilent two-color array"], "technology": ["in situ oligonucleotide"], "manufacturer": ["Agilent Technologies"]})}
    return FakeGSE(metadata, gsms, gpls)


def test_build_and_map_channel_expression_matrices_writes_signal_and_reference_when_confident(monkeypatch, tmp_path):
    gse = make_agilent_two_channel_gse_with_reference()
    fake_map = pd.DataFrame([
        {"probe_id": "1007_s_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
        {"probe_id": "1053_at", "gene_symbol": "RFC2", "entrez_id": "5982", "source": "direct_columns"},
    ])
    monkeypatch.setattr(download.probe_mapping, "get_or_build_probe_gene_map", lambda gpl_id: fake_map)

    channel_paths, channel_roles = download.build_and_map_channel_expression_matrices(gse, tmp_path, _AGILENT_PLATFORM_DETAILS)

    assert channel_roles["method"] == "metadata+variance"
    assert channel_roles["reference_channel"] == 1
    assert channel_roles["signal_channel"] == 2
    assert (tmp_path / "channel_signal_expression.tsv.gz").exists()
    assert (tmp_path / "channel_reference_expression.tsv.gz").exists()
    signal = pd.read_csv(tmp_path / "channel_signal_expression.tsv.gz", sep="\t")
    reference = pd.read_csv(tmp_path / "channel_reference_expression.tsv.gz", sep="\t")
    assert signal.equals(pd.read_csv(channel_paths[2], sep="\t"))
    assert reference.equals(pd.read_csv(channel_paths[1], sep="\t"))


def test_build_and_map_channel_expression_matrices_writes_both_channels(monkeypatch, tmp_path):
    gse = make_agilent_two_channel_gse()
    fake_map = pd.DataFrame([
        {"probe_id": "1007_s_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
        {"probe_id": "1053_at", "gene_symbol": "RFC2", "entrez_id": "5982", "source": "direct_columns"},
    ])
    monkeypatch.setattr(download.probe_mapping, "get_or_build_probe_gene_map", lambda gpl_id: fake_map)

    channel_paths, channel_roles = download.build_and_map_channel_expression_matrices(gse, tmp_path, _AGILENT_PLATFORM_DETAILS)

    assert set(channel_paths.keys()) == {1, 2}
    assert channel_paths[1] == tmp_path / "channel1_expression.tsv.gz"
    assert channel_paths[2] == tmp_path / "channel2_expression.tsv.gz"
    assert (tmp_path / "channel1_probe_matrix.tsv.gz").exists()
    assert (tmp_path / "channel2_probe_matrix.tsv.gz").exists()
    assert channel_roles["method"] in ("ambiguous", "metadata", "variance", "metadata+variance")

    channel1 = pd.read_csv(channel_paths[1], sep="\t")
    channel2 = pd.read_csv(channel_paths[2], sep="\t")
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
    assert result == ({}, {})


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
    assert result == ({}, {})


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
    # make_agilent_two_channel_gse has no reference-design metadata hints, but
    # its hardcoded values do happen to clear the variance-gap threshold even
    # with just 2 samples -- exercises the variance-only path (no metadata
    # hint at all) alongside the dedicated, more realistic fixture in
    # test_download_cohort_persists_and_reuses_channel_roles.
    assert result["channel_roles"]["method"] == "variance"
    assert (tmp_path / "GSE_AGILENT" / "channel_roles.json").exists()


def test_download_cohort_persists_and_reuses_channel_roles(monkeypatch, tmp_path):
    gse = make_agilent_two_channel_gse_with_reference()
    monkeypatch.setattr(download.geo_fetch, "fetch_series", lambda gse_id: gse)
    monkeypatch.setattr(
        download.clinical_annotate, "plan_column_mapping", lambda samples, model=None: clinical_annotate.ColumnMappingPlan()
    )
    fake_map = pd.DataFrame([
        {"probe_id": "1007_s_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
        {"probe_id": "1053_at", "gene_symbol": "RFC2", "entrez_id": "5982", "source": "direct_columns"},
    ])
    monkeypatch.setattr(download.probe_mapping, "get_or_build_probe_gene_map", lambda gpl_id: fake_map)

    result = download.download_cohort("GSE_AGILENT_REF", series_dir=tmp_path)

    assert result["channel_roles"] == {
        "reference_channel": 1, "signal_channel": 2, "method": "metadata+variance", "notes": "",
    }
    assert result["channel_signal_expression_path"] == str(tmp_path / "GSE_AGILENT_REF" / "channel_signal_expression.tsv.gz")
    assert result["channel_reference_expression_path"] == str(tmp_path / "GSE_AGILENT_REF" / "channel_reference_expression.tsv.gz")
    assert (tmp_path / "GSE_AGILENT_REF" / "channel_roles.json").exists()

    # Reuse path: re-derived purely from files on disk, no re-fetch.
    monkeypatch.setattr(download.geo_fetch, "fetch_series", _raise)
    cached = download.download_cohort("GSE_AGILENT_REF", series_dir=tmp_path)
    assert cached["channel_roles"] == result["channel_roles"]
    assert cached["channel_signal_expression_path"] == result["channel_signal_expression_path"]
    assert cached["channel_reference_expression_path"] == result["channel_reference_expression_path"]


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


def test_download_cohort_skips_clinical_annotate_llm_call_by_default(monkeypatch, tmp_path):
    """The one unconditional Claude call in the whole download path (needs
    ANTHROPIC_API_KEY) -- must not fire unless explicitly requested via
    clinical_annotate_flag=True, so a bare download never requires a key."""
    gse = make_rnaseq_gse()
    monkeypatch.setattr(download.geo_fetch, "fetch_series", lambda gse_id: gse)

    def _raise(*a, **k):
        raise AssertionError("plan_column_mapping must not be called when clinical_annotate_flag=False")

    monkeypatch.setattr(download.clinical_annotate, "plan_column_mapping", _raise)

    class _NoopResponse:
        content = b"gene\tcount\n"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(download.requests, "get", lambda *a, **k: _NoopResponse())

    result = download.download_cohort("GSE_RNASEQ", series_dir=tmp_path)  # clinical_annotate_flag defaults to False

    assert (tmp_path / "GSE_RNASEQ" / "annotation.tsv").exists()
    assert result["assay_types"] == ["bulk_rnaseq"]


def test_download_cohort_calls_clinical_annotate_when_flag_enabled(monkeypatch, tmp_path):
    gse = make_rnaseq_gse()
    monkeypatch.setattr(download.geo_fetch, "fetch_series", lambda gse_id: gse)

    calls = []

    def _fake_plan(samples, model=None):
        calls.append(samples)
        return clinical_annotate.ColumnMappingPlan()

    monkeypatch.setattr(download.clinical_annotate, "plan_column_mapping", _fake_plan)

    class _NoopResponse:
        content = b"gene\tcount\n"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(download.requests, "get", lambda *a, **k: _NoopResponse())

    download.download_cohort("GSE_RNASEQ", series_dir=tmp_path, clinical_annotate_flag=True)

    assert len(calls) == 1


def test_download_cohort_reports_rnaseq_expression_qc_and_reuses_from_cache(monkeypatch, tmp_path):
    """Real GSE163305 shape: an FPKM matrix (linear-scale) is the only
    genuine expression file among its supplementary files.
    resolve_primary_expression_matrix now auto-fixes linear-scale values
    (probe_mapping.normalize_expression_matrix) rather than just flagging
    them, so this ends up with a clean "ok" status once written -- see
    test_check_rnaseq_expression_qc_flags_linear_scale_fpkm_matrix for the
    same fix in isolation."""
    gse = make_rnaseq_gse()
    gse.metadata["supplementary_file"].append("ftp://example.com/GSE_RNASEQ_notes.txt")
    monkeypatch.setattr(download.geo_fetch, "fetch_series", lambda gse_id: gse)
    monkeypatch.setattr(
        download.clinical_annotate, "plan_column_mapping", lambda samples, model=None: clinical_annotate.ColumnMappingPlan()
    )

    genes = _big_gene_index()
    fpkm_bytes = pd.DataFrame(
        {"GSM1": [0.0, 16096.1] + [0.0] * (len(genes) - 2)}, index=genes
    ).to_csv().encode()
    fpkm_gz = gzip.compress(fpkm_bytes)

    def fake_get(url, timeout=None):
        class _Resp:
            content = fpkm_gz if "counts" in url else b"unrelated notes file"

            def raise_for_status(self):
                pass

        return _Resp()

    monkeypatch.setattr(download.requests, "get", fake_get)

    result = download.download_cohort("GSE_RNASEQ_QC", series_dir=tmp_path)

    assert result["primary_expression_unit"] == "count"
    assert result["primary_expression_file"] == str(tmp_path / "GSE_RNASEQ_QC" / "expression" / "GSE_RNASEQ_counts.tsv.gz")
    assert "expression_qc_notes" not in result  # nothing left to report -- already fixed
    assert result["expression_status"] == clinical_annotate.EXPRESSION_STATUS_OK

    written = pd.read_csv(result["primary_expression_file"], sep="\t", compression="gzip")
    assert written["GSM1"].max() == pytest.approx(np.log2(16096.1 + 1), abs=1e-3)

    annotation = pd.read_csv(tmp_path / "GSE_RNASEQ_QC" / "annotation.tsv", sep="\t")
    assert (annotation["expression_status"] == clinical_annotate.EXPRESSION_STATUS_OK).all()

    monkeypatch.setattr(download.geo_fetch, "fetch_series", _raise)
    cached = download.download_cohort("GSE_RNASEQ_QC", series_dir=tmp_path)
    assert cached["primary_expression_file"] == result["primary_expression_file"]
    assert cached["primary_expression_unit"] == result["primary_expression_unit"]
    assert cached["expression_status"] == result["expression_status"]


def test_download_cohort_flags_no_expression_matrix_when_only_non_matrix_files_published(monkeypatch, tmp_path):
    """Real GSE108651 shape: the only per-sample supplementary files are a
    Cuffdiff differential-expression table and an rMATS splicing-analysis
    spreadsheet -- neither is a raw/normalized expression matrix, so nothing
    matches select_primary_expression_file's unit keywords at all."""
    gse = make_rnaseq_gse()
    gse.metadata["supplementary_file"] = [
        "ftp://example.com/GSM1_gene_exp.diff.gz",
        "ftp://example.com/GSM1_novel_filtered_rMATS.xlsx",
    ]
    monkeypatch.setattr(download.geo_fetch, "fetch_series", lambda gse_id: gse)
    monkeypatch.setattr(
        download.clinical_annotate, "plan_column_mapping", lambda samples, model=None: clinical_annotate.ColumnMappingPlan()
    )

    class _NoopResponse:
        content = b"irrelevant"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(download.requests, "get", lambda *a, **k: _NoopResponse())

    result = download.download_cohort("GSE_NO_MATRIX", series_dir=tmp_path)

    assert "primary_expression_file" not in result
    assert result["expression_status"] == clinical_annotate.EXPRESSION_STATUS_NO_MATRIX
    # No sidecar either -- nothing was found or flagged to write.
    assert not (tmp_path / "GSE_NO_MATRIX" / "expression_qc.json").exists()

    annotation = pd.read_csv(tmp_path / "GSE_NO_MATRIX" / "annotation.tsv", sep="\t")
    assert (annotation["expression_status"] == clinical_annotate.EXPRESSION_STATUS_NO_MATRIX).all()

    monkeypatch.setattr(download.geo_fetch, "fetch_series", _raise)
    cached = download.download_cohort("GSE_NO_MATRIX", series_dir=tmp_path)
    assert cached["expression_status"] == clinical_annotate.EXPRESSION_STATUS_NO_MATRIX


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


def test_cached_result_returns_none_when_not_downloaded(tmp_path):
    assert download._cached_result("GSE_NOPE", tmp_path / "GSE_NOPE") is None


def _raise(*args, **kwargs):
    raise AssertionError("should not be called on a reuse path")


def test_download_cohort_reuses_existing_download_without_refetching(monkeypatch, tmp_path):
    gse = make_microarray_gse()
    fetch_calls = []
    monkeypatch.setattr(download.geo_fetch, "fetch_series", lambda gse_id: (fetch_calls.append(gse_id), gse)[1])
    monkeypatch.setattr(
        download.clinical_annotate, "plan_column_mapping", lambda samples, model=None: clinical_annotate.ColumnMappingPlan()
    )
    fake_map = pd.DataFrame([
        {"probe_id": "1007_s_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
        {"probe_id": "1053_at", "gene_symbol": "RFC2", "entrez_id": "5982", "source": "direct_columns"},
    ])
    monkeypatch.setattr(download.probe_mapping, "get_or_build_probe_gene_map", lambda gpl_id: fake_map)

    first = download.download_cohort("GSE_ARRAY", series_dir=tmp_path)
    assert len(fetch_calls) == 1

    # Any of these being called on the reuse path would fail the test.
    monkeypatch.setattr(download.geo_fetch, "fetch_series", _raise)
    monkeypatch.setattr(download.clinical_annotate, "plan_column_mapping", _raise)
    monkeypatch.setattr(download.probe_mapping, "get_or_build_probe_gene_map", _raise)

    second = download.download_cohort("GSE_ARRAY", series_dir=tmp_path)

    assert second["expression_path"] == first["expression_path"]
    assert second["annotation_path"] == first["annotation_path"]
    assert second["assay_types"] == first["assay_types"]


def test_download_cohort_force_redoes_even_when_already_downloaded(monkeypatch, tmp_path):
    gse = make_microarray_gse()
    fetch_calls = []
    monkeypatch.setattr(download.geo_fetch, "fetch_series", lambda gse_id: (fetch_calls.append(gse_id), gse)[1])
    monkeypatch.setattr(
        download.clinical_annotate, "plan_column_mapping", lambda samples, model=None: clinical_annotate.ColumnMappingPlan()
    )
    fake_map = pd.DataFrame([
        {"probe_id": "1007_s_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
        {"probe_id": "1053_at", "gene_symbol": "RFC2", "entrez_id": "5982", "source": "direct_columns"},
    ])
    monkeypatch.setattr(download.probe_mapping, "get_or_build_probe_gene_map", lambda gpl_id: fake_map)

    download.download_cohort("GSE_ARRAY", series_dir=tmp_path)
    assert len(fetch_calls) == 1

    download.download_cohort("GSE_ARRAY", series_dir=tmp_path, force=True)
    assert len(fetch_calls) == 2


def test_download_cohort_backfills_only_missing_rma_without_redoing_annotation(monkeypatch, tmp_path):
    gse = make_microarray_gse()
    fetch_calls = []
    monkeypatch.setattr(download.geo_fetch, "fetch_series", lambda gse_id: (fetch_calls.append(gse_id), gse)[1])
    monkeypatch.setattr(
        download.clinical_annotate, "plan_column_mapping", lambda samples, model=None: clinical_annotate.ColumnMappingPlan()
    )
    fake_map = pd.DataFrame([
        {"probe_id": "1007_s_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
        {"probe_id": "1053_at", "gene_symbol": "RFC2", "entrez_id": "5982", "source": "direct_columns"},
    ])
    monkeypatch.setattr(download.probe_mapping, "get_or_build_probe_gene_map", lambda gpl_id: fake_map)

    download.download_cohort("GSE_ARRAY", series_dir=tmp_path)  # no --rma yet
    assert len(fetch_calls) == 1
    assert not (tmp_path / "GSE_ARRAY" / "expression_rma.tsv.gz").exists()

    # Would fail the test if the reuse+rma-backfill path redid the submitter-
    # value matrix or the clinical_annotate LLM call.
    monkeypatch.setattr(download.clinical_annotate, "plan_column_mapping", _raise)
    monkeypatch.setattr(
        download, "download_cel_files", lambda gse, out_dir: {"GSM1": tmp_path / "a.CEL", "GSM2": tmp_path / "b.CEL"}
    )
    fake_rma_matrix = pd.DataFrame({"GSM1": [1.0, 2.0], "GSM2": [3.0, 4.0]}, index=["1007_s_at", "1053_at"])
    monkeypatch.setattr(download.renormalize, "run_rma", lambda cel_files, gpl_id: fake_rma_matrix)

    result = download.download_cohort("GSE_ARRAY", series_dir=tmp_path, rma=True)

    assert len(fetch_calls) == 2  # re-fetched, just for CEL download/RMA
    assert result["expression_rma_path"] is not None
    assert (tmp_path / "GSE_ARRAY" / "expression_rma.tsv.gz").exists()


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
    # aggregate_probes_to_genes already auto-log2-transforms microarray values,
    # so a clean matrix comes out "ok" here (unlike a raw RNA-seq FPKM/counts
    # file, which is never transformed since it isn't ours to mutate).
    assert result["expression_status"] == clinical_annotate.EXPRESSION_STATUS_OK
    annotation = pd.read_csv(tmp_path / "GSE_ARRAY" / "annotation.tsv", sep="\t")
    assert (annotation["expression_status"] == clinical_annotate.EXPRESSION_STATUS_OK).all()


def test_download_cohort_rejects_non_human_organism(monkeypatch, tmp_path):
    gse = make_microarray_gse()
    for gsm in gse.gsms.values():
        gsm.metadata["organism_ch1"] = ["Mus musculus"]
    monkeypatch.setattr(download.geo_fetch, "fetch_series", lambda gse_id: gse)

    with pytest.raises(download.UnsupportedCohortError, match="Mus musculus"):
        download.download_cohort("GSE_MOUSE", series_dir=tmp_path)

    assert not (tmp_path / "GSE_MOUSE" / "annotation.tsv").exists()


def make_mixed_mrna_and_cna_gse():
    """One mRNA platform (GPL96, allowed) plus a CNA/SNP platform (GPL_CNA,
    rejected) on the same series -- the exact shape of GSE32688 (mRNA + CNA +
    miRNA combined), just with only the CNA half added on top of the plain
    microarray fixture."""
    gse = make_microarray_gse()
    gse.gsms["GSM_CNA"] = FakeGSM(
        {
            "title": ["cna1"], "geo_accession": ["GSM_CNA"], "platform_id": ["GPL_CNA"],
            "organism_ch1": ["Homo sapiens"], "characteristics_ch1": [],
        },
        table=pd.DataFrame({"ID_REF": [1, 2], "VALUE": [0.1, -0.2]}),
    )
    gse.gpls["GPL_CNA"] = FakeGPL({
        "title": ["[GenomeWideSNP_6] Affymetrix Genome-Wide Human SNP 6.0 Array"],
        "technology": ["in situ oligonucleotide"], "manufacturer": ["Affymetrix"],
        "data_row_count": ["900000"],
    })
    return gse


def test_download_cohort_skips_unsupported_platform_but_keeps_supported_one(monkeypatch, tmp_path, capsys):
    gse = make_mixed_mrna_and_cna_gse()
    monkeypatch.setattr(download.geo_fetch, "fetch_series", lambda gse_id: gse)
    monkeypatch.setattr(
        download.clinical_annotate, "plan_column_mapping", lambda samples, model=None: clinical_annotate.ColumnMappingPlan()
    )
    fake_map = pd.DataFrame([
        {"probe_id": "1007_s_at", "gene_symbol": "DDR1", "entrez_id": "780", "source": "direct_columns"},
        {"probe_id": "1053_at", "gene_symbol": "RFC2", "entrez_id": "5982", "source": "direct_columns"},
    ])
    monkeypatch.setattr(download.probe_mapping, "get_or_build_probe_gene_map", lambda gpl_id: fake_map)

    result = download.download_cohort("GSE_MIXED", series_dir=tmp_path)

    assert "GPL_CNA (cna array" in capsys.readouterr().out
    samples = pd.read_csv(tmp_path / "GSE_MIXED" / "samples.tsv", sep="\t")
    assert set(samples["gsm_id"]) == {"GSM1", "GSM2"}  # GSM_CNA dropped
    genes = pd.read_csv(result["expression_path"], sep="\t")
    assert set(genes["gene_symbol"]) == {"DDR1", "RFC2"}


def test_download_cohort_fails_when_every_platform_is_unsupported(monkeypatch, tmp_path):
    gse = make_mixed_mrna_and_cna_gse()
    del gse.gsms["GSM1"]
    del gse.gsms["GSM2"]
    del gse.gpls["GPL96"]
    monkeypatch.setattr(download.geo_fetch, "fetch_series", lambda gse_id: gse)

    with pytest.raises(download.UnsupportedCohortError, match="GPL_CNA"):
        download.download_cohort("GSE_CNA_ONLY", series_dir=tmp_path)

    assert not (tmp_path / "GSE_CNA_ONLY" / "annotation.tsv").exists()


def test_download_cohort_rejects_low_density_microarray_platform(monkeypatch, tmp_path):
    gse = make_microarray_gse()
    gse.gpls["GPL96"].metadata["data_row_count"] = ["500"]
    monkeypatch.setattr(download.geo_fetch, "fetch_series", lambda gse_id: gse)

    with pytest.raises(download.UnsupportedCohortError, match="500"):
        download.download_cohort("GSE_OLD_ARRAY", series_dir=tmp_path)


def test_resolve_download_targets_returns_cached_id_with_zero_fetches(monkeypatch, tmp_path):
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
    download.download_cohort("GSE_ARRAY", series_dir=tmp_path)  # populate the cache

    monkeypatch.setattr(download.geo_fetch, "fetch_series", _raise)
    monkeypatch.setattr(download.geo_fetch, "resolve_leaf_series_ids", _raise)

    assert download.resolve_download_targets("GSE_ARRAY", series_dir=tmp_path) == ["GSE_ARRAY"]


def test_resolve_download_targets_expands_uncached_superseries(monkeypatch, tmp_path):
    monkeypatch.setattr(download.geo_fetch, "resolve_leaf_series_ids", lambda gse_id: ["GSE101", "GSE102"])
    monkeypatch.setattr(
        download.geo_fetch, "find_superseries_orphans",
        lambda gse_id, leaf_ids: {"orphaned_gsm_ids": [], "orphaned_supplementary_files": []},
    )
    assert download.resolve_download_targets("GSE100", series_dir=tmp_path) == ["GSE101", "GSE102"]


def test_resolve_download_targets_writes_superseries_marker_with_orphans(monkeypatch, tmp_path):
    """The marker records the subseries it expanded to plus
    find_superseries_orphans' result, so both survive past this one CLI run
    -- see build_prmt5_cohort_annotations.py's use of it, and _cached_result's
    guard below."""
    monkeypatch.setattr(download.geo_fetch, "resolve_leaf_series_ids", lambda gse_id: ["GSE101", "GSE102"])
    monkeypatch.setattr(
        download.geo_fetch, "find_superseries_orphans",
        lambda gse_id, leaf_ids: {"orphaned_gsm_ids": ["GSM_X"], "orphaned_supplementary_files": []},
    )

    download.resolve_download_targets("GSE100", series_dir=tmp_path)

    marker = json.loads((tmp_path / "GSE100" / "superseries.json").read_text())
    assert marker == {
        "subseries": ["GSE101", "GSE102"],
        "orphaned_gsm_ids": ["GSM_X"],
        "orphaned_supplementary_files": [],
    }


def test_resolve_download_targets_does_not_write_marker_for_non_superseries(monkeypatch, tmp_path):
    monkeypatch.setattr(download.geo_fetch, "resolve_leaf_series_ids", lambda gse_id: [gse_id])
    download.resolve_download_targets("GSE1", series_dir=tmp_path)
    assert not (tmp_path / "GSE1" / "superseries.json").exists()


def test_cached_result_ignores_stale_annotation_when_superseries_marker_exists(tmp_path):
    """A GSE id can end up with a real annotation.tsv/series.tsv left over
    from before it was ever recognized as a SuperSeries (a real, live
    situation: GSE236500 already had stale output from an earlier, unrelated
    run). Once superseries.json exists, that stale pair must never be
    trusted as "already downloaded" again, even without --force."""
    out_dir = tmp_path / "GSE236500"
    out_dir.mkdir()
    pd.DataFrame([{"platform_details": "[]"}]).to_csv(out_dir / "series.tsv", sep="\t", index=False)
    pd.DataFrame([{"gsm_id": "GSM1"}]).to_csv(out_dir / "annotation.tsv", sep="\t", index=False)
    (out_dir / "superseries.json").write_text(json.dumps({"subseries": ["GSE236496"]}))

    assert download._cached_result("GSE236500", out_dir) is None
