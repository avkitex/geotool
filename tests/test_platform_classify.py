import pandas as pd
import pytest

from geotool import platform_classify


@pytest.mark.parametrize(
    "technology,title,expected",
    [
        ("high-throughput sequencing", "Illumina NovaSeq X Plus (Homo sapiens)", "bulk_rnaseq"),
        ("high-throughput sequencing", "10x Genomics single cell 3' (Homo sapiens)", "scrnaseq"),
        ("in situ oligonucleotide", "[HG-U133_Plus_2] Affymetrix Human Genome U133 Plus 2.0 Array", "microarray"),
        ("spotted DNA/cDNA", "Some two-color cDNA array", "microarray"),
        ("", "some unrelated platform", "other"),
    ],
)
def test_classify_assay_type(technology, title, expected):
    assert platform_classify.classify_assay_type(technology, title) == expected


@pytest.mark.parametrize(
    "manufacturer,title,expected",
    [
        ("Affymetrix", "[HG-U133_Plus_2] Affymetrix Human Genome U133 Plus 2.0 Array", "affymetrix"),
        ("", "Illumina HumanHT-12 V4.0", "illumina"),
        ("Agilent Technologies", "Agilent-014850 Whole Human Genome", "agilent"),
        ("", "Some obscure spotted array", "other"),
    ],
)
def test_classify_vendor(manufacturer, title, expected):
    assert platform_classify.classify_vendor(manufacturer, title) == expected


@pytest.mark.parametrize(
    "data_row_count,expected",
    [
        (20000, "full_transcriptome"),
        ("54675", "full_transcriptome"),
        (12000, "full_transcriptome"),
        (11999, "limited"),
        (500, "limited"),
        (0, "unknown"),
        ("", "unknown"),
        (None, "unknown"),
        ("not-a-number", "unknown"),
    ],
)
def test_classify_coverage(data_row_count, expected):
    assert platform_classify.classify_coverage(data_row_count) == expected


def test_classify_platform_microarray_gets_vendor_and_coverage():
    result = platform_classify.classify_platform(
        "GPL570",
        {
            "title": ["[HG-U133_Plus_2] Affymetrix Human Genome U133 Plus 2.0 Array"],
            "technology": ["in situ oligonucleotide"],
            "manufacturer": ["Affymetrix"],
            "data_row_count": ["54675"],
        },
    )
    assert result == {
        "gpl_id": "GPL570",
        "assay_type": "microarray",
        "vendor": "affymetrix",
        "coverage": "full_transcriptome",
        "content": "mrna",
        "data_row_count": 54675,
    }


def test_classify_platform_rnaseq_has_no_vendor_or_coverage():
    result = platform_classify.classify_platform(
        "GPL34284",
        {"title": ["Illumina NovaSeq X Plus (Homo sapiens)"], "technology": ["high-throughput sequencing"]},
    )
    assert result["assay_type"] == "bulk_rnaseq"
    assert result["vendor"] is None
    assert result["coverage"] is None
    assert result["content"] is None
    assert result["data_row_count"] is None


def test_classify_platform_accepts_esummary_docsum_shape():
    """esummary docsums use 'ptechtype' instead of 'technology' and plain strings, not lists."""
    result = platform_classify.classify_platform(
        "GPL570", {"title": "Affymetrix Human Genome U133 Plus 2.0 Array", "ptechtype": "in situ oligonucleotide"}
    )
    assert result["assay_type"] == "microarray"
    assert result["vendor"] == "affymetrix"


@pytest.mark.parametrize(
    "sample_metadata,expected",
    [
        ({"library_selection": ["PolyA"]}, "polyA"),
        ({"library_selection": ["Oligo-dT"]}, "polyA"),
        ({"library_selection": ["RANDOM"]}, "total_rna"),
        ({"library_selection": ["Hybrid Selection"]}, "exome_capture"),
        ({"library_selection": ["ChIP"]}, "other"),
        ({"library_selection": ["unspecified"], "extract_protocol_ch1": ["Ribo-Zero rRNA depletion was performed"]}, "total_rna"),
        ({"library_selection": [], "extract_protocol_ch1": ["Poly-A selection using oligo-dT beads"]}, "polyA"),
        ({"library_selection": [], "extract_protocol_ch1": ["Exome capture with a hybridization kit"]}, "exome_capture"),
        ({"library_selection": [], "extract_protocol_ch1": ["No relevant information here"]}, "unknown"),
    ],
)
def test_classify_rnaseq_library(sample_metadata, expected):
    assert platform_classify.classify_rnaseq_library(sample_metadata) == expected


def test_classify_scrna_platform_detects_10x():
    series_row = {"title": "scRNA-seq atlas", "overall_design": "10x Genomics Chromium droplet-based capture", "summary": ""}
    samples = pd.DataFrame({"description": [""]})
    assert platform_classify.classify_scrna_platform(series_row, samples) == "10x"


def test_classify_scrna_platform_detects_smartseq():
    series_row = {"title": "scRNA-seq atlas", "overall_design": "", "summary": "Libraries prepared using Smart-seq2 on single sorted cells"}
    samples = pd.DataFrame({"description": [""]})
    assert platform_classify.classify_scrna_platform(series_row, samples) == "smartseq"


def test_classify_scrna_platform_unknown_when_no_hint():
    series_row = {"title": "some series", "overall_design": "", "summary": ""}
    samples = pd.DataFrame({"description": [""]})
    assert platform_classify.classify_scrna_platform(series_row, samples) == "unknown"


@pytest.mark.parametrize(
    "title,expected",
    [
        # Real platforms from GSE32688 -- the exact combined mRNA+CNA+miRNA series
        # that motivated this classification.
        ("[HG-U133_Plus_2] Affymetrix Human Genome U133 Plus 2.0 Array", "mrna"),
        ("[GenomeWideSNP_6] Affymetrix Genome-Wide Human SNP 6.0 Array", "cna"),
        ("miRCURY LNA microRNA Array, v.11.0 - hsa, mmu & rno", "mirna"),
        ("Agilent-019118 Human miRNA Microarray 2.0 G4470B (miRNA ID version)", "mirna"),
        ("Agilent-014950 Human Genome CGH Microarray 244A", "cna"),
        ("Affymetrix Genome-Wide Human SNP Array 5.0", "cna"),
        ("Arraystar Human LncRNA Microarray V2.0", "lncrna"),
        # Combined chips explicitly mentioning mRNA content stay "mrna", not
        # mirna/lncrna -- the "combined is fine" case.
        ("Agilent-062918 Human LncRNA + mRNA Array", "mrna"),
        ("Arraystar Human LncRNA and mRNA Expression Microarray V4.0", "mrna"),
        ("Illumina HumanHT-12 V4.0 expression beadchip", "mrna"),
    ],
)
def test_classify_array_content(title, expected):
    assert platform_classify.classify_array_content(title) == expected


@pytest.mark.parametrize(
    "detail,expected_ok",
    [
        ({"assay_type": "bulk_rnaseq"}, True),  # not gated at all
        ({"assay_type": "microarray", "content": "mrna", "data_row_count": 54675}, True),
        ({"assay_type": "microarray", "content": "mrna", "data_row_count": 8000}, True),  # boundary: kept
        ({"assay_type": "microarray", "content": "mrna", "data_row_count": 7999}, False),  # boundary: rejected
        ({"assay_type": "microarray", "content": "cna", "data_row_count": 900000}, False),
        ({"assay_type": "microarray", "content": "mirna", "data_row_count": 100}, False),
        ({"assay_type": "microarray", "content": "lncrna", "data_row_count": 30000}, False),
        ({"assay_type": "microarray", "content": "mrna", "data_row_count": None}, True),  # unknown count, benefit of the doubt
    ],
)
def test_platform_supported(detail, expected_ok):
    ok, reason = platform_classify.platform_supported(detail)
    assert ok is expected_ok
    if not ok:
        assert reason


def test_summarize_array_content_joins_dedupes_and_sorts():
    docsums = {
        "GPL570": {"title": "[HG-U133_Plus_2] Affymetrix Human Genome U133 Plus 2.0 Array", "ptechtype": "in situ oligonucleotide"},
        "GPL6801": {"title": "[GenomeWideSNP_6] Affymetrix Genome-Wide Human SNP 6.0 Array", "ptechtype": "in situ oligonucleotide"},
        "GPL7723": {"title": "miRCURY LNA microRNA Array, v.11.0 - hsa, mmu & rno", "ptechtype": "spotted oligonucleotide"},
    }
    result = platform_classify.summarize_array_content(docsums, ["GPL7723", "GPL570", "GPL6801"])
    assert result == "cna;mirna;mrna"


def test_summarize_array_content_skips_missing_and_non_microarray_platforms():
    docsums = {
        "GPL34284": {"title": "Illumina NovaSeq X Plus (Homo sapiens)", "ptechtype": "high-throughput sequencing"},
    }
    assert platform_classify.summarize_array_content(docsums, ["GPL34284", "GPL_UNKNOWN"]) == ""
