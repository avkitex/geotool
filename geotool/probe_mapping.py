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
alongside the ratio -- neither channel is assumed to be "the real sample" or
"the reference", so both are just named by channel number.

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

_GENE_SYMBOL_COLUMNS = ["Gene Symbol", "GENE_SYMBOL", "Symbol", "gene_symbol"]
_ENTREZ_ID_COLUMNS = ["ENTREZ_GENE_ID", "Entrez_Gene_ID", "GENE_ID", "entrez_gene_id"]

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
