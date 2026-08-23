"""Probe -> gene mapping for microarray platforms.

Five mapping strategies, tried in order, all parsed straight from the
platform's own (always-current-when-fetched) annotation table -- no external
lookups:

1. Direct columns: most platforms carry a "Gene Symbol"/"ENTREZ_GENE_ID"-style
   column pair (e.g. GPL96). Multi-gene probes use " /// " to separate
   parallel-position values -- the first pair is taken as the probe's one
   canonical gene.
2. Packed "gene_assignment" text (transcript-cluster platforms, e.g. GPL17586
   / Affymetrix HTA-style arrays): repeated "id // symbol // description //
   location // entrez_id" groups joined by " /// ". Parsed with a regex
   rather than an Ensembl REST lookup, since the gene symbol and Entrez ID
   are already embedded in the text.
3. Brainarray-style custom-CDF platforms (e.g. GPL23432, a re-annotated
   Affymetrix HG-U133 Plus 2 using Brainarray's ENSG CDF): each probeset is
   already one gene, identified by an "ORF" column holding the Ensembl Gene
   ID directly (e.g. "ENSG00000000003"), with no Gene Symbol/Entrez columns
   and no gene_assignment text at all. There's no official gene symbol in the
   platform's own table for these (its "Description" column is the gene's
   full name, not a short symbol) -- the Ensembl Gene ID itself is stored as
   gene_symbol so there's still a real, non-empty key to group/merge on.
   Only recognized when ORF actually looks like an Ensembl gene ID, since
   older spotted-array platforms also use a column literally named "ORF" for
   unrelated identifier schemes.
4. "PrimarySequenceName" column (older spotted cDNA/oligo platforms, e.g.
   GPL7091): holds the gene symbol directly for probes that were annotated
   at submission time. Probes that weren't are left as a bare internal clone
   ID instead (seen live: "I_959282", "I_1000440") rather than a real
   symbol -- these are excluded (not stored as a fake gene symbol) via a
   syntactic "I_<digits>" pattern check, not a guess about gene identity.
5. Probe ID *is* the gene symbol (e.g. GPL20769, a "collapsed"/gene-level
   re-annotated two-channel Agilent design used by GSE71729): the platform's
   own "ID" column holds gene symbols directly ("A1BG", "TP53", ...) with no
   separate symbol/Entrez/ORF/PrimarySequenceName column at all -- there's no
   translation to do, the probe already *is* the gene. Recognized by checking
   for several _CANONICAL_HUMAN_GENE_SYMBOLS literally present among the ID
   column's own values, rather than guessing from ID shape/format alone (an
   opaque vendor probe-ID scheme -- "A_23_P100001", "1007_s_at", a purely
   numeric index -- could never coincidentally match several of these).

Every probe gets at most one gene (source="unmapped" if none of the above
found anything) -- that, plus caching the result once per platform in
get_or_build_probe_gene_map, is what makes "only one correspondence
probe->gene per platform" true.

Also handles two-channel (e.g. Agilent Cy3/Cy5) samples: build_probe_matrix
reads the precomputed VALUE ratio; build_channel_probe_matrices separately
builds each channel's own raw-intensity matrix for samples that publish
per-channel columns (see detect_channel_columns), as an *additional* signal
alongside the ratio. Neither channel1_expression.tsv.gz/channel2_expression.
tsv.gz is ever assumed to be "the real sample" or "the reference" -- both are
always written, just named by channel number -- but detect_reference_channel
makes a best-effort, confidence-gated guess (metadata text + cross-sample
variance) at which one actually is which, used by download.py to
*additionally* write channel_signal_expression.tsv.gz/channel_reference_
expression.tsv.gz copies when confident.

Every gene-level result aggregate_probes_to_genes produces is passed through
maybe_log2_transform: values are log2(x + 1)-transformed unless
needs_log2_transform's simple heuristic (any value over 50) says the data
already looks log2 scale -- e.g. a VALUE ratio, which can be negative, must
never be re-transformed, while raw linear-scale intensities (routinely in
the hundreds or thousands) need it to be comparable across platforms. This
happens at the gene-expression level, not the probe level, so
probe_matrix.tsv.gz / channelN_probe_matrix.tsv.gz stay exactly as submitted.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from geotool import config, geo_fetch, platform_classify

# Heuristic threshold for "does this matrix already look log2-transformed".
# Log2 expression values rarely exceed the low teens even for highly
# expressed genes, whereas untransformed linear-scale intensities (the
# common case for e.g. Affymetrix MAS5-style VALUE) routinely run into the
# hundreds or thousands -- a single value above this is enough to call the
# whole matrix "not yet log2".
_LOG2_ALREADY_TRANSFORMED_MAX = 50

PROBE_ID_COL = "ID"

_GENE_SYMBOL_COLUMNS = ["Gene Symbol", "GENE_SYMBOL", "Symbol", "gene_symbol", "GeneSymbol"]
_ENTREZ_ID_COLUMNS = ["ENTREZ_GENE_ID", "Entrez_Gene_ID", "GENE_ID", "entrez_gene_id", "EntrezGeneID"]

_FIELD_SEP_RE = re.compile(r"\s*//\s*")
_ENSEMBL_GENE_ID_RE = re.compile(r"^ENSG\d+$")
_CLONE_ID_PLACEHOLDER_RE = re.compile(r"^I_\d+$")

# A handful of near-universally-present human gene symbols -- common
# housekeeping genes and famous oncogenes/tumor suppressors that appear on
# essentially every human expression platform. Used only as a confidence
# check in parse_probe_id_as_symbol, not as a real gene symbol reference.
_CANONICAL_HUMAN_GENE_SYMBOLS = {
    "GAPDH", "ACTB", "TP53", "EGFR", "MYC", "KRAS", "BRCA1", "BRCA2", "PTEN",
    "VEGFA", "TNF", "IL6", "INS", "ALB", "HBB", "CDKN2A", "RB1", "APC", "PIK3CA",
}
_MIN_CANONICAL_SYMBOL_MATCHES = 5

# Two-channel samples (e.g. Agilent Cy3/Cy5 reference-design arrays) publish
# their raw per-channel intensities under wildly inconsistent column names
# across submitters -- each tuple here is a (channel1_column, channel2_column)
# pair seen live on real GEO platforms, tried in order and required together
# (a submitter who uses one naming convention uses it for both channels).
# Channel *number*, not dye, is what's paired -- channel 1 is always the
# green/Cy3 scanner channel and channel 2 always red/Cy5 (a scanner hardware
# fact); what actually varies per dye-swap replicate is which *biological
# sample* was labeled with which dye (recorded in characteristics_ch1/ch2),
# never which channel number corresponds to which dye. Most two-channel
# series don't expose per-channel columns at all (only a precomputed ratio
# in VALUE) -- there's nothing to split for those, by far the common case.
_CHANNEL_COLUMN_PAIRS = [
    ("ch1 Intensity", "ch2 Intensity"),
    ("CH1_SIGNAL", "CH2_SIGNAL"),
    ("CH1_SIGNAL_MEAN", "CH2_SIGNAL_MEAN"),
    ("CH1_MEAN_SIGNAL", "CH2_MEAN_SIGNAL"),
    ("CH1_MEAN", "CH2_MEAN"),  # GenePix-style, seen live on GPL7504 (GSE50470/GSE21997/GSE22049)
    ("Intensity_Cy3", "Intensity_Cy5"),
    ("gMedianSignal", "rMedianSignal"),
    ("gMeanSignal", "rMeanSignal"),
    ("gProcessedSignal", "rProcessedSignal"),
]


def _probe_id_column(annotation_df: pd.DataFrame) -> str:
    return PROBE_ID_COL if PROBE_ID_COL in annotation_df.columns else annotation_df.columns[0]


def _first_matching_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _first_value(cell) -> str | None:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None
    text = str(cell).strip()
    if not text or text == "---":
        return None
    return text.split(" /// ")[0].strip()


def parse_direct_columns(annotation_df: pd.DataFrame) -> pd.DataFrame | None:
    """Gene Symbol / ENTREZ_GENE_ID style columns (GPL96-style). None if absent."""
    symbol_col = _first_matching_column(annotation_df, _GENE_SYMBOL_COLUMNS)
    entrez_col = _first_matching_column(annotation_df, _ENTREZ_ID_COLUMNS)
    if symbol_col is None and entrez_col is None:
        return None

    probe_col = _probe_id_column(annotation_df)
    rows = []
    for _, row in annotation_df.iterrows():
        symbol = _first_value(row.get(symbol_col)) if symbol_col else None
        entrez = _first_value(row.get(entrez_col)) if entrez_col else None
        if symbol is None and entrez is None:
            continue
        rows.append({"probe_id": row[probe_col], "gene_symbol": symbol, "entrez_id": entrez, "source": "direct_columns"})
    return pd.DataFrame(rows, columns=["probe_id", "gene_symbol", "entrez_id", "source"])


def parse_gene_assignment(text) -> tuple[str | None, str | None]:
    """Parse one packed 'gene_assignment' cell:
    'id // symbol // description // location // entrez_id /// id // symbol // ...'
    Returns the first (gene_symbol, entrez_id) pair with a real symbol, or (None, None).
    """
    if not text or not isinstance(text, str):
        return None, None
    for sub_record in text.split(" /// "):
        fields = _FIELD_SEP_RE.split(sub_record.strip())
        if len(fields) < 2:
            continue
        symbol = fields[1].strip()
        if not symbol or symbol == "---":
            continue
        entrez_id = next((f.strip() for f in reversed(fields) if f.strip().isdigit()), None)
        return symbol, entrez_id
    return None, None


def parse_orf_ensembl_column(annotation_df: pd.DataFrame) -> pd.DataFrame | None:
    """Brainarray-style custom-CDF platforms (e.g. GPL23432): an "ORF" column
    holding the Ensembl Gene ID directly, one gene per probeset already.
    None if there's no ORF column, or its values don't actually look like
    Ensembl gene IDs (some older spotted-array platforms reuse the "ORF"
    column name for a different identifier scheme).
    """
    if "ORF" not in annotation_df.columns:
        return None
    values = annotation_df["ORF"].dropna().astype(str)
    if values.empty or not values.str.match(_ENSEMBL_GENE_ID_RE).all():
        return None

    probe_col = _probe_id_column(annotation_df)
    rows = []
    for _, row in annotation_df.iterrows():
        orf = _first_value(row.get("ORF"))
        if orf is None:
            continue
        rows.append({"probe_id": row[probe_col], "gene_symbol": orf, "entrez_id": None, "source": "ensembl_orf"})
    return pd.DataFrame(rows, columns=["probe_id", "gene_symbol", "entrez_id", "source"])


def parse_primary_sequence_name_column(annotation_df: pd.DataFrame) -> pd.DataFrame | None:
    """Older spotted cDNA/oligo platforms (e.g. GPL7091): a "PrimarySequenceName"
    column holds the gene symbol directly for probes annotated at submission
    time. Probes that weren't annotated are left as a bare internal clone ID
    instead (e.g. "I_959282") -- excluded via a syntactic pattern check
    rather than stored as a fake gene symbol.
    """
    if "PrimarySequenceName" not in annotation_df.columns:
        return None

    probe_col = _probe_id_column(annotation_df)
    rows = []
    for _, row in annotation_df.iterrows():
        symbol = _first_value(row.get("PrimarySequenceName"))
        if symbol is None or _CLONE_ID_PLACEHOLDER_RE.match(symbol):
            continue
        rows.append({"probe_id": row[probe_col], "gene_symbol": symbol, "entrez_id": None, "source": "primary_sequence_name"})
    return pd.DataFrame(rows, columns=["probe_id", "gene_symbol", "entrez_id", "source"])


def parse_probe_id_as_symbol(annotation_df: pd.DataFrame) -> pd.DataFrame | None:
    """Some platforms (e.g. GPL20769) use the gene symbol itself as the probe
    ID -- there's no separate probe->gene translation to do at all. Recognized
    by requiring several _CANONICAL_HUMAN_GENE_SYMBOLS literally present among
    the ID column's own values (an opaque vendor probe-ID scheme could never
    coincidentally match several of these), not a guess from ID shape alone.
    """
    probe_col = _probe_id_column(annotation_df)
    ids = annotation_df[probe_col].dropna().astype(str)
    if ids.isin(_CANONICAL_HUMAN_GENE_SYMBOLS).sum() < _MIN_CANONICAL_SYMBOL_MATCHES:
        return None
    return pd.DataFrame({
        "probe_id": ids, "gene_symbol": ids, "entrez_id": None, "source": "probe_id_is_symbol",
    })


def extract_probe_gene_table(annotation_df: pd.DataFrame) -> pd.DataFrame:
    """Per-platform probe->gene parser. Tries direct columns, then packed
    gene_assignment text, then a Brainarray-style ORF/Ensembl column, then a
    PrimarySequenceName column, then "the probe ID is already the gene
    symbol", else leaves every probe unmapped (never guessed).
    """
    direct = parse_direct_columns(annotation_df)
    if direct is not None and not direct.empty:
        return direct

    if "gene_assignment" in annotation_df.columns:
        probe_col = _probe_id_column(annotation_df)
        rows = []
        for _, row in annotation_df.iterrows():
            symbol, entrez = parse_gene_assignment(row.get("gene_assignment"))
            if symbol is None:
                continue
            rows.append({"probe_id": row[probe_col], "gene_symbol": symbol, "entrez_id": entrez, "source": "gene_assignment"})
        if rows:
            return pd.DataFrame(rows, columns=["probe_id", "gene_symbol", "entrez_id", "source"])

    orf = parse_orf_ensembl_column(annotation_df)
    if orf is not None and not orf.empty:
        return orf

    primary_sequence_name = parse_primary_sequence_name_column(annotation_df)
    if primary_sequence_name is not None and not primary_sequence_name.empty:
        return primary_sequence_name

    probe_id_as_symbol = parse_probe_id_as_symbol(annotation_df)
    if probe_id_as_symbol is not None and not probe_id_as_symbol.empty:
        return probe_id_as_symbol

    return pd.DataFrame(columns=["probe_id", "gene_symbol", "entrez_id", "source"])


def get_or_build_probe_gene_map(gpl_id: str, platforms_dir: Path | None = None) -> pd.DataFrame:
    """Cache wrapper: data/platforms/<GPL>/probe_gene_map.tsv, computed once
    per platform and reused by every series on it.

    probe_id is always cast to str before returning, on both the cache-hit
    and cache-miss path -- platforms whose probe IDs look purely numeric
    (e.g. GPL7091's "1", "2", ...) would otherwise come back as int64 on a
    cache miss (inherited untouched from the platform's own "ID" column) but
    str on a cache hit (pd.read_csv(..., dtype=str) below), silently
    breaking aggregate_probes_to_genes's index join depending on whichever
    happened to run first in a given process.
    """
    cache_path = (platforms_dir or config.PLATFORMS_DIR) / gpl_id / "probe_gene_map.tsv"
    if cache_path.exists():
        return pd.read_csv(cache_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])

    gpl = geo_fetch.fetch_platform(gpl_id)
    table = extract_probe_gene_table(gpl.table)
    table["probe_id"] = table["probe_id"].astype(str)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache_path, sep="\t", index=False)
    return table


def needs_log2_transform(matrix: pd.DataFrame) -> bool:
    """True if matrix looks like it hasn't been log2-transformed yet -- see
    _LOG2_ALREADY_TRANSFORMED_MAX for the reasoning. An empty/all-NaN matrix
    has nothing to check, so counts as "already fine" (False).
    """
    return bool((matrix > _LOG2_ALREADY_TRANSFORMED_MAX).to_numpy().any())


def maybe_log2_transform(matrix: pd.DataFrame) -> pd.DataFrame:
    """Apply log2(value + 1) to every value in a gene-level expression
    matrix, but only when needs_log2_transform says it isn't already log2
    scale -- data that's already log2 (e.g. a two-channel log-ratio, which
    can be negative) must never be transformed again.
    """
    if matrix.empty or not needs_log2_transform(matrix):
        return matrix
    return np.log2(matrix + 1)


# A whole-transcriptome (or even a modest targeted-panel) gene-level matrix
# always has far more features (rows) than samples (columns) -- thousands to
# tens of thousands of genes vs. tens to a few hundred samples. More columns
# than rows is a strong, submitter-naming-convention-independent signal that
# a matrix has samples in rows and features in columns instead of the
# expected orientation (this codebase's own build_probe_matrix/
# aggregate_probes_to_genes always produce genes-as-rows; a submitter's own
# RNA-seq file has no such guarantee).
_MIN_ROWS_FOR_ORIENTATION_CHECK = 10

# Recognized gene-identifier column names, highest priority first -- used to
# pick the one column normalize_expression_matrix keeps when a file has more
# than one identifier-shaped column (e.g. this codebase's own gene_symbol +
# entrez_id pair after a multi-platform merge, or a submitter's gene_id +
# Symbol + HGNC). Matched case-insensitively as a substring against the
# whole column name with separators normalized, so "Gene Symbol" and
# "gene_symbol" both match "gene_symbol" -- safe here because these are all
# specific/compound enough that no real gene *value* ever collides with one
# as a column *name*.
_ID_COLUMN_SUBSTRING_PRIORITY = [
    "gene_symbol", "symbol", "gene_id", "geneid", "ensembl", "probe_id", "probeid", "id_ref",
]

# Shorter, generic identifier-column names that are only safe to match
# *exactly* (not as a substring): a transposed matrix's gene names can
# legitimately be column headers like "GENE0".."GENE20000" (this codebase's
# own placeholder convention, and not unheard of as a real one), and
# "gene" in _ID_COLUMN_SUBSTRING_PRIORITY would wrongly match every one of
# those as "the identifier column" instead of letting the orientation check
# catch the real problem.
_ID_COLUMN_EXACT_PRIORITY = ["gene", "id", "name"]

# Column names that are recognizable submitter/annotation metadata rather
# than per-sample expression data -- dropped by normalize_expression_matrix
# whenever they aren't the one column chosen as the identifier. Covers
# alternate identifier columns (a file with both "gene_id" and "Symbol" only
# needs one), pure annotation columns from featureCounts/htseq-style files
# (seqnames/strand/biotype/etc., published alongside per-sample counts in
# the same table -- live example: GSE208732/GSE243584's PDAC RNA-seq count
# matrices), and derived/computed columns that look numeric like a sample
# but aren't one -- a fold-change or p-value column sitting next to the raw
# per-sample values it was computed from (live example: GSE197728's "12-0160
# fold change (ALA/Veh)", a ratio between two of its own sample columns).
# Substring-matched against column *names* (not values), so the same
# "GENE0" collision risk above doesn't apply here.
_METADATA_COLUMN_KEYWORDS = _ID_COLUMN_SUBSTRING_PRIORITY + [
    "entrez", "seqnames", "chr", "chromosome", "strand", "start", "end", "width",
    "biotype", "type_of_gene", "locus", "hgnc", "name", "length", "description", "annotation",
    "fold_change", "foldchange", "log2fc", "logfc", "p_value", "pvalue", "padj", "fdr", "q_value", "qvalue",
]

# Bare row-numbering columns -- e.g. a literal "#" column, seen live in
# GSE208732/GSE243584's PDAC RNA-seq files right after the gene symbol
# column -- are numeric and match no word-shaped keyword above, but are
# just as much a non-sample column as any of them. Matched exactly (not as
# a substring): unlike the keywords above, several of these are short
# enough that a substring match risks colliding with a real column name.
# "id" specifically: live example, GSE243850's own bare "ID" column (raw
# 1..N gene indices) once its real header is recovered from a misparsed
# title row (see the Unnamed:-N-columns fixup in normalize_expression_matrix)
# -- _pick_identifier_column already prefers a more specific match
# ("Gene symbol" via the "symbol" substring) when one exists, so a bare "id"
# reaching here at all means it lost that contest and is a redundant second
# identifier column, not a sample, even though its dtype (sequential
# integers) looks numeric like one.
_METADATA_COLUMN_EXACT_KEYWORDS = ["#", "index", "no", "no.", "rownum", "row_number", "row_num", "id"]


def _normalize_column_name(name) -> str:
    return re.sub(r"[\s_-]+", "_", str(name).strip().lower())


def _pick_identifier_column(columns) -> object:
    """The one column to keep as this matrix's row identifier: the highest-
    priority match in _ID_COLUMN_SUBSTRING_PRIORITY (substring), then
    _ID_COLUMN_EXACT_PRIORITY (exact name only), then the very first column
    if nothing matches -- covers a plain/unlabeled index column read back
    as "Unnamed: 0", or any other submitter naming convention not
    recognized at all. Positionally, whatever's first is still the
    identifier axis, not a second sample.
    """
    normalized = {col: _normalize_column_name(col) for col in columns}
    for keyword in _ID_COLUMN_SUBSTRING_PRIORITY:
        for col, norm in normalized.items():
            if keyword in norm:
                return col
    for keyword in _ID_COLUMN_EXACT_PRIORITY:
        for col, norm in normalized.items():
            if norm == keyword:
                return col
    return columns[0]


def check_expression_qc(matrix: pd.DataFrame) -> list[str]:
    """Sanity-check a *final* expression matrix (any numeric orientation) for
    three easy-to-miss problems. Called after normalize_expression_matrix
    in every download path, so in practice this mostly catches whatever
    that function couldn't safely auto-fix (negative values -- see below --
    are never auto-corrected, since there's no way to tell a mis-transformed
    matrix apart from a legitimate log-ratio one) or a matrix that never
    went through normalize_expression_matrix at all:

    1. Not log2-transformed -- informational, not necessarily wrong (most
       submitted RNA-seq FPKM/TPM/counts files are legitimately linear-scale)
       -- same heuristic as needs_log2_transform (any value over
       _LOG2_ALREADY_TRANSFORMED_MAX).
    2. Negative values -- never valid for a raw or log2(x+1)-transformed
       expression matrix. Most often means either log2(x) was applied
       *without* the +1 pseudocount (log2 of a value between 0 and 1 goes
       negative, log2(0) is -inf) -- a real risk specifically for RNA-seq,
       where 0-count genes are common -- or the file isn't actually a raw
       expression matrix at all (e.g. a log-fold-change column from a
       differential-expression results table).
    3. Looks transposed -- more columns than rows (see
       _MIN_ROWS_FOR_ORIENTATION_CHECK), i.e. probably samples-as-rows /
       features-as-columns instead of the expected genes-as-rows /
       samples-as-columns.
    4. No numeric column(s) at all despite having more than one column --
       for a real expression matrix this is the single worst sign possible,
       not "nothing to check": it almost always means the file's real
       header row got misparsed as data (a multi-row header, live example:
       GSE243850's "Raw counts and normalized read count" file -- pandas
       read a mostly-blank title row as the header, leaving the real header
       ["ID", "Gene symbol", "YS_2542338", ...] as the first data row, which
       poisons every sample column's dtype to object). check_rnaseq_expression
       _qc calls this directly on the loaded matrix, never through
       normalize_expression_matrix (whose own bail-out note for exactly this
       shape only ever reaches the microarray path) -- without this check,
       such a cohort's expression_status silently comes out "ok" even though
       geotool.rnaseq_finalize's own parser later fails outright on the same
       file ("list index out of range").

    Returns human-readable notes, empty if nothing stood out.
    """
    numeric = matrix.select_dtypes(include="number")
    if numeric.empty:
        if matrix.empty or matrix.shape[1] <= 1:
            return []
        return [
            f"no numeric column(s) found among {matrix.shape[1]} column(s) -- likely a misparsed/shifted "
            "header row (e.g. a multi-row header), not a genuine identifier-only matrix"
        ]

    notes = []
    max_value = numeric.max(numeric_only=True).max()
    if pd.notna(max_value) and max_value > _LOG2_ALREADY_TRANSFORMED_MAX:
        notes.append(f"linear-scale, not log2-transformed (max value {max_value:.1f})")

    n_negative = int((numeric < 0).to_numpy().sum())
    if n_negative:
        notes.append(
            f"{n_negative} negative value(s) found -- possible log2 transform without a "
            "+1 pseudocount, or this isn't a raw/log2 expression matrix"
        )

    n_rows, n_cols = matrix.shape
    if n_rows >= _MIN_ROWS_FOR_ORIENTATION_CHECK and n_cols > n_rows:
        notes.append(
            f"more columns ({n_cols}) than rows ({n_rows}) -- expected features (genes) as rows and "
            "samples as columns; this matrix may be transposed"
        )

    return notes


def check_gene_count(matrix: pd.DataFrame) -> list[str]:
    """RNA-seq-specific: a real full-transcriptome bulk RNA-seq gene-level
    table has tens of thousands of rows. Fewer than
    config.MIN_EXPECTED_RNASEQ_GENE_COUNT strongly suggests the submitter
    only published a filtered subset of genes -- live example: GSE197728's
    supplementary table, which only reports genes with FPKM > 10 (7833
    rows). There's no way to recover genes that were never published, so
    this is reported, not auto-fixed by normalize_expression_matrix.

    Not called for microarray matrices -- a platform's own probe/gene
    coverage is a fixed design property there (see
    platform_classify.classify_coverage), not evidence of filtering.
    """
    n_genes = len(matrix)
    if n_genes == 0 or n_genes >= config.MIN_EXPECTED_RNASEQ_GENE_COUNT:
        return []
    return [
        f"only {n_genes} genes (< {config.MIN_EXPECTED_RNASEQ_GENE_COUNT}) -- likely a "
        "filtered/truncated gene list, not the full transcriptome"
    ]


def normalize_expression_matrix(matrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The single place every *final* gene-level expression matrix -- any
    platform, any assay type -- passes through immediately before being
    written to disk, auto-fixing (not just reporting -- see
    check_expression_qc for the report-only checks that remain) the three
    problems real submitter files keep exhibiting:

    1. Extra non-sample columns -- e.g. a featureCounts-style RNA-seq count
       file's chromosome/strand/biotype annotation columns sitting right
       alongside per-sample counts, or this codebase's own gene_symbol +
       entrez_id pair once a multi-platform merge that needed both keys is
       done with them. Every column matching a known identifier/annotation
       name (see _METADATA_COLUMN_KEYWORDS) other than the one column
       _pick_identifier_column chooses is dropped -- a numeric column like
       "entrez_id" is metadata here too, not a second sample, even though
       its dtype alone can't tell it apart from a real sample column.
    2. Transposed orientation (samples as rows, genes as columns) --
       detected the same way check_expression_qc always has (more sample
       columns than gene rows, on a whole-transcriptome-sized matrix, see
       _MIN_ROWS_FOR_ORIENTATION_CHECK), fixed by transposing back.
    3. Linear-scale (not log2) values -- log2(x + 1), the same
       maybe_log2_transform already applied at the gene level for
       microarray data, now applied uniformly to every platform's final
       matrix here. Skipped if the matrix already has negative values (a
       real log-ratio dataset, or a previous bad transform) -- log2 of a
       negative number is undefined, and guessing which case it is isn't
       safe; check_expression_qc's negative-value note is what surfaces
       that instead.

    Returns (fixed_matrix, notes) describing what was actually changed --
    empty if nothing needed fixing. `matrix` is expected to have the row
    identifier as its first (or otherwise identifiable, see
    _pick_identifier_column) column, not as a pandas index -- the
    index=False convention every _write_matrix(..., index=False) call in
    this codebase already uses -- but a non-default index is recovered into
    a column first regardless (see below), rather than assumed away.
    """
    if matrix.empty:
        return matrix, []

    notes: list[str] = []
    # A file whose header row has one fewer field than its data rows (a
    # common real convention: no header label for the row-name column at
    # all) makes pandas auto-detect that first field as the index rather
    # than a column -- live example: GSE242915's raw counts file. Recovered
    # into a real column so it isn't mistaken for having no identifier
    # column at all (which would silently make the first *sample* column
    # look like the identifier instead).
    if not matrix.index.equals(pd.RangeIndex(len(matrix))):
        notes.append("gene identifier was a pandas index (no column header for it in the source file) -- recovered as a column")
        matrix = matrix.reset_index()
        if matrix.columns[0] == "index":
            matrix = matrix.rename(columns={"index": "gene_id"})

    # A spurious title/section-label row sitting above the real header --
    # most of its cells blank, so pandas auto-names those columns
    # "Unnamed: N" -- leaves the true header as the first *data* row
    # instead. Live example: GSE243850's "Raw counts and normalized read
    # count" file, whose literal first line is blank except for a lone
    # "Raw count" section label, with the real header (["ID", "Gene
    # symbol", "YS_2542338", ...]) one row below -- expression_status came
    # out "ok" anyway, since check_expression_qc had no numeric columns
    # left to check once every sample column's dtype got poisoned to
    # object by that stray header-shaped row (see check_expression_qc's
    # own no-numeric-columns note for the other half of that fix).
    # Detected by column *names* (mostly pandas' own placeholder), not
    # column count, so it can't collide with the pandas-index case above
    # (real column labels there, just one too few). Only ever shifts once:
    # if the row below still doesn't look like a real header, this isn't
    # that case, and the extra-columns bail-out further down is the safety
    # net for whatever's still wrong.
    unnamed_fraction = sum(bool(re.fullmatch(r"Unnamed: \d+", str(c))) for c in matrix.columns) / len(matrix.columns)
    if unnamed_fraction > 0.5 and len(matrix) > 1:
        new_header = [str(v) for v in matrix.iloc[0].tolist()]
        # Real column headers are unique; a repeated value (most obviously,
        # every cell holding the same placeholder) means this row doesn't
        # actually look like a header either -- skip the shift rather than
        # produce a matrix with duplicate column labels (matrix[col] below,
        # and every check downstream, assumes one Series per label, not a
        # sub-DataFrame). Left for the ordinary extra-columns bail-out
        # further down to refuse safely, same as before this fixup existed.
        if len(set(new_header)) == len(new_header):
            notes.append(
                f"header row looks blank/misparsed ({unnamed_fraction:.0%} of columns unnamed) -- a title/section-label "
                "row above the real header -- used the next row down as the real header instead"
            )
            matrix = matrix.iloc[1:].reset_index(drop=True)
            matrix.columns = new_header
            # Every column's dtype was already committed to `object` by
            # pandas when it first parsed the file (that stray string-valued
            # header row, mixed in with the numeric data rows below, forces
            # the whole column to object) -- removing the row doesn't
            # retroactively re-infer it, so a genuinely all-numeric sample
            # column would otherwise still look non-numeric to every check
            # below (starting with extra_cols' own is_numeric_dtype test)
            # and get dropped as "extra" right along with the real metadata
            # columns. Only ever promotes a column when *every* remaining
            # value parses cleanly -- anything that doesn't (a real text
            # column) is left alone.
            for col in matrix.columns:
                if not pd.api.types.is_numeric_dtype(matrix[col]):
                    coerced = pd.to_numeric(matrix[col], errors="coerce")
                    if coerced.notna().sum() == matrix[col].notna().sum():
                        matrix[col] = coerced

    columns = list(matrix.columns)
    id_col = _pick_identifier_column(columns)
    normalized_names = {col: _normalize_column_name(col) for col in columns}
    extra_cols = [
        col for col in columns
        if col != id_col
        and (
            not pd.api.types.is_numeric_dtype(matrix[col])
            or any(kw in normalized_names[col] for kw in _METADATA_COLUMN_KEYWORDS)
            or normalized_names[col] in _METADATA_COLUMN_EXACT_KEYWORDS
        )
    ]
    # Dropping to fewer than 2 remaining sample columns (from more than a
    # couple of candidates) means something deeper is wrong with how this
    # file was parsed -- not "a few annotation columns to prune" but likely
    # a multi-row header pandas read as data (live example: GSE243850's
    # "Raw counts and normalized read count" file, where the real header
    # was row 2, not row 1, so almost every column looked unrecognizable
    # and the sole "identifier" column that survived held row numbers, not
    # genes). Guessing further here risks silently destroying the file
    # instead of just failing to improve it, so this bails out with nothing
    # touched rather than dropping ~everything.
    if extra_cols and len(columns) > 3 and len(columns) - len(extra_cols) - 1 < 2:
        notes.append(
            f"{len(extra_cols)} of {len(columns) - 1} non-identifier columns look like metadata, leaving "
            "fewer than 2 recognizable sample columns -- matrix left unchanged, may need manual inspection "
            "(e.g. a multi-row header)"
        )
        return matrix, notes
    if extra_cols:
        notes.append(f"dropped {len(extra_cols)} extra non-sample column(s): {extra_cols}")
        matrix = matrix[[id_col] + [c for c in columns if c not in extra_cols and c != id_col]]

    sample_cols = [c for c in matrix.columns if c != id_col]
    n_rows, n_samples = matrix.shape[0], len(sample_cols)
    if n_rows >= _MIN_ROWS_FOR_ORIENTATION_CHECK and n_samples > n_rows:
        notes.append(
            f"transposed ({n_rows} rows, {n_samples} sample column(s)) -- samples were in rows, "
            "features in columns; transposed back"
        )
        matrix = matrix.set_index(id_col).transpose()
        matrix.index.name = "gene_id"
        matrix = matrix.reset_index()
        id_col = "gene_id"
        sample_cols = [c for c in matrix.columns if c != id_col]

    sample_values = matrix[sample_cols]
    if not sample_values.empty and (sample_values.to_numpy() < 0).any():
        return matrix, notes  # possible log-ratio data -- see check_expression_qc instead

    if needs_log2_transform(sample_values):
        max_value = sample_values.max(numeric_only=True).max()
        notes.append(f"linear-scale values (max {max_value:.1f}) -- applied log2(x + 1)")
        matrix = matrix.copy()
        matrix[sample_cols] = maybe_log2_transform(sample_values)

    return matrix, notes


def build_probe_matrix(gse) -> pd.DataFrame:
    """Probes x samples matrix from every fetched sample's own data table
    (ID_REF -> VALUE). Samples with no data table (e.g. RNA-seq, handled via
    download.py's supplementary-file path instead) are skipped. Values are
    left exactly as submitted -- see aggregate_probes_to_genes for the
    log2(x + 1) transform, applied at the gene-expression level rather than
    here at the probe level.
    """
    columns = {}
    for gsm_id, gsm in gse.gsms.items():
        table = getattr(gsm, "table", None)
        if table is None or table.empty or "ID_REF" not in table.columns or "VALUE" not in table.columns:
            continue
        indexed = table.set_index(table["ID_REF"].astype(str))
        values = pd.to_numeric(indexed["VALUE"], errors="coerce")
        columns[gsm_id] = values
    if not columns:
        return pd.DataFrame()
    return pd.DataFrame(columns)


def detect_channel_columns(table: pd.DataFrame) -> tuple[str, str] | None:
    """Return (channel1_column, channel2_column) if this sample's own data
    table exposes recognizable per-channel intensity columns (see
    _CHANNEL_COLUMN_PAIRS), else None -- callers should treat None as "this
    sample can't be split" rather than guessing at unfamiliar column names.
    """
    for ch1_col, ch2_col in _CHANNEL_COLUMN_PAIRS:
        if ch1_col in table.columns and ch2_col in table.columns:
            return ch1_col, ch2_col
    return None


def build_channel_probe_matrices(gse) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Probes x samples matrices for each channel of every two-channel sample
    whose own data table has detectable per-channel columns.

    Only those samples contribute a column here (channel_count must be "2"
    and detect_channel_columns must find a match) -- every other sample
    (single-channel, or two-channel but only publishing the VALUE ratio) is
    untouched and keeps going through the existing build_probe_matrix path
    unaffected. Both returned matrices share the same sample columns. Values
    are left exactly as submitted -- see aggregate_probes_to_genes for the
    log2(x + 1) transform, applied at the gene-expression level.
    """
    channel1_cols: dict[str, pd.Series] = {}
    channel2_cols: dict[str, pd.Series] = {}
    for gsm_id, gsm in gse.gsms.items():
        channel_count = platform_classify.as_str(gsm.metadata.get("channel_count")).strip()
        if channel_count != "2":
            continue
        table = getattr(gsm, "table", None)
        if table is None or table.empty or "ID_REF" not in table.columns:
            continue
        columns = detect_channel_columns(table)
        if columns is None:
            continue
        ch1_col, ch2_col = columns
        indexed = table.set_index("ID_REF")
        channel1_cols[gsm_id] = pd.to_numeric(indexed[ch1_col], errors="coerce")
        channel2_cols[gsm_id] = pd.to_numeric(indexed[ch2_col], errors="coerce")

    channel1_matrix = pd.DataFrame(channel1_cols) if channel1_cols else pd.DataFrame()
    channel2_matrix = pd.DataFrame(channel2_cols) if channel2_cols else pd.DataFrame()
    return channel1_matrix, channel2_matrix


# Text hint for a common-reference-design channel: the same (or near-same)
# reference material hybridized on every array, as opposed to the actual
# per-sample biological material. Checked per sample against that sample's
# own characteristics/source_name -- not assumed fixed across the whole
# series -- so a dye-swap design (reference alternates channel per replicate)
# is handled naturally rather than assumed away.
_REFERENCE_HINT_RE = re.compile(r"\breference\b|\bpool(ed)?\b", re.IGNORECASE)

# Metadata: a channel number is called "reference" only if this fraction of
# samples with a *clear* per-sample hint (one channel matches, the other
# doesn't) agree on it.
_MIN_METADATA_AGREEMENT = 0.9

# Variance: the reference channel should vary less across samples than the
# actual biological sample does (median per-probe variance of log2 values).
# A call is only made if the relative gap between channels clears this bar --
# live-validated against 3 real two-channel Agilent series with a confirmed
# (metadata-labeled) common reference design: relative gaps of 12%-66%, so
# 10% is a conservative floor that comfortably covers all of them while still
# discarding a noise-level (near-0%) gap as "ambiguous" rather than guessing.
_MIN_VARIANCE_RELATIVE_GAP = 0.10


def _channel_metadata_hint(gsm) -> int | None:
    """1 or 2 if exactly one of this sample's own ch1/ch2 characteristics +
    source_name mentions a reference/pool and the other doesn't; None if
    neither or both do (no clear per-sample signal).
    """
    def _text(channel: str) -> str:
        md = gsm.metadata
        return " ".join(md.get(f"source_name_{channel}", []) + md.get(f"characteristics_{channel}", []))

    ch1_hit = bool(_REFERENCE_HINT_RE.search(_text("ch1")))
    ch2_hit = bool(_REFERENCE_HINT_RE.search(_text("ch2")))
    if ch1_hit and not ch2_hit:
        return 1
    if ch2_hit and not ch1_hit:
        return 2
    return None


def detect_reference_channel(gse, channel1_matrix: pd.DataFrame, channel2_matrix: pd.DataFrame) -> dict:
    """Best-effort guess at which channel of a two-channel series is the
    fixed reference (vs. the channel actually carrying the biological
    sample) -- {"reference_channel", "signal_channel", "method", "notes"}.
    reference_channel/signal_channel are None and method is "ambiguous" when
    neither signal below is clear enough to call -- never guessed past that
    point, same "unknown rather than guess" spirit as the rest of this
    module (e.g. classify_scrna_platform).

    Two independent signals, live-validated against 3 real two-channel
    Agilent series with a metadata-confirmed common-reference design
    (GSE50470, GSE21997, GSE22049 -- all agreed on both signals):
    1. Metadata (_channel_metadata_hint, _MIN_METADATA_AGREEMENT): per-sample
       characteristics/source_name text.
    2. Cross-sample variance (_MIN_VARIANCE_RELATIVE_GAP): the reference
       channel is by design the same or near-same material on every array,
       so its values should vary less across samples than the actual
       biological sample's do.

    If both signals fire and agree, method is "metadata+variance" (highest
    confidence); if only one fires, that call stands alone; if they fire and
    disagree, the result is "ambiguous" (recorded in notes) rather than
    picking one arbitrarily.
    """
    result = {"reference_channel": None, "signal_channel": None, "method": "ambiguous", "notes": ""}
    if channel1_matrix.empty or channel2_matrix.empty:
        return result

    hints = [
        _channel_metadata_hint(gse.gsms[gsm_id]) for gsm_id in channel1_matrix.columns if gsm_id in gse.gsms
    ]
    clear_hints = [h for h in hints if h is not None]
    metadata_call = None
    if clear_hints:
        if clear_hints.count(1) / len(clear_hints) >= _MIN_METADATA_AGREEMENT:
            metadata_call = 1
        elif clear_hints.count(2) / len(clear_hints) >= _MIN_METADATA_AGREEMENT:
            metadata_call = 2

    ch1_var = np.log2(channel1_matrix.clip(lower=1)).var(axis=1, skipna=True).median()
    ch2_var = np.log2(channel2_matrix.clip(lower=1)).var(axis=1, skipna=True).median()
    variance_call = None
    if pd.notna(ch1_var) and pd.notna(ch2_var):
        lower_channel = 1 if ch1_var <= ch2_var else 2
        lower_var, higher_var = min(ch1_var, ch2_var), max(ch1_var, ch2_var)
        if higher_var > 0 and (higher_var - lower_var) / higher_var >= _MIN_VARIANCE_RELATIVE_GAP:
            variance_call = lower_channel

    if metadata_call and variance_call:
        if metadata_call == variance_call:
            result.update(reference_channel=metadata_call, signal_channel=3 - metadata_call, method="metadata+variance")
        else:
            result["notes"] = f"metadata says channel {metadata_call} is the reference, variance says channel {variance_call} -- disagree"
    elif metadata_call:
        result.update(reference_channel=metadata_call, signal_channel=3 - metadata_call, method="metadata")
    elif variance_call:
        result.update(reference_channel=variance_call, signal_channel=3 - variance_call, method="variance")

    return result


def aggregate_probes_to_genes(probe_matrix: pd.DataFrame, probe_gene_map: pd.DataFrame, agg: str = "mean") -> pd.DataFrame:
    """Probes x samples -> genes x samples.

    Groups every probe (or transcript, already resolved to its gene upstream
    in extract_probe_gene_table) mapping to the same gene and aggregates
    their values -- this single step is also what summarizes transcripts of
    the same gene before the gene-level value is produced. Unmapped probes
    are dropped. Genes are keyed by entrez_id when available, falling back
    to gene_symbol for platforms/probes that only carry a symbol.

    The resulting gene-level values are log2(x + 1)-transformed unless
    needs_log2_transform says they already look log2 scale (see
    maybe_log2_transform) -- done here, at the gene-expression level, rather
    than on the raw per-probe/per-channel values, so probe_matrix.tsv.gz and
    channelN_probe_matrix.tsv.gz stay exactly as submitted while every
    gene-level expression file (expression.tsv.gz, channelN_expression.tsv.gz)
    ends up on a comparable scale regardless of whether the platform/
    submitter's own values were raw or already log-transformed.
    """
    if probe_matrix.empty or probe_gene_map.empty:
        return pd.DataFrame(columns=["entrez_id", "gene_symbol"])

    # probe_matrix's index dtype comes from however GEOparse happened to
    # parse the sample's own ID_REF column -- int64 for platforms with
    # purely-numeric probe IDs (e.g. GPL7091's "1", "2", ...), str for most
    # others (e.g. "1007_s_at"). probe_gene_map's probe_id is always str
    # (get_or_build_probe_gene_map). Cast both sides so the join below can't
    # silently come back empty from a dtype mismatch alone.
    probe_matrix = probe_matrix.set_axis(probe_matrix.index.astype(str))

    mapped = probe_gene_map[probe_gene_map["source"] != "unmapped"].copy()
    mapped = mapped[mapped["gene_symbol"].notna() | mapped["entrez_id"].notna()]
    if mapped.empty:
        return pd.DataFrame(columns=["entrez_id", "gene_symbol"])

    mapped["gene_key"] = mapped["entrez_id"].where(mapped["entrez_id"].notna(), mapped["gene_symbol"])
    mapped["probe_id"] = mapped["probe_id"].astype(str)
    mapped = mapped.set_index("probe_id")

    joined = probe_matrix.join(mapped[["gene_key", "gene_symbol", "entrez_id"]], how="inner")
    if joined.empty:
        return pd.DataFrame(columns=["entrez_id", "gene_symbol"])

    sample_cols = list(probe_matrix.columns)
    grouped_values = maybe_log2_transform(joined.groupby("gene_key")[sample_cols].agg(agg))
    gene_labels = joined.groupby("gene_key")[["gene_symbol", "entrez_id"]].first()
    return gene_labels.join(grouped_values).reset_index(drop=True)
