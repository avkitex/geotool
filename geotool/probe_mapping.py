"""Probe -> gene mapping for microarray platforms.

Two mapping strategies, tried in order, both parsed straight from the
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

Every probe gets at most one gene (source="unmapped" if neither strategy
found anything) -- that, plus caching the result once per platform in
get_or_build_probe_gene_map, is what makes "only one correspondence
probe->gene per platform" true.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from geotool import config, geo_fetch

PROBE_ID_COL = "ID"

_GENE_SYMBOL_COLUMNS = ["Gene Symbol", "GENE_SYMBOL", "Symbol", "gene_symbol"]
_ENTREZ_ID_COLUMNS = ["ENTREZ_GENE_ID", "Entrez_Gene_ID", "GENE_ID", "entrez_gene_id"]

_FIELD_SEP_RE = re.compile(r"\s*//\s*")


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


def extract_probe_gene_table(annotation_df: pd.DataFrame) -> pd.DataFrame:
    """Per-platform probe->gene parser. Tries direct columns, then packed
    gene_assignment text, else leaves every probe unmapped (never guessed).
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

    return pd.DataFrame(columns=["probe_id", "gene_symbol", "entrez_id", "source"])


def get_or_build_probe_gene_map(gpl_id: str, platforms_dir: Path | None = None) -> pd.DataFrame:
    """Cache wrapper: data/platforms/<GPL>/probe_gene_map.tsv, computed once
    per platform and reused by every series on it.
    """
    cache_path = (platforms_dir or config.PLATFORMS_DIR) / gpl_id / "probe_gene_map.tsv"
    if cache_path.exists():
        return pd.read_csv(cache_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])

    gpl = geo_fetch.fetch_platform(gpl_id)
    table = extract_probe_gene_table(gpl.table)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache_path, sep="\t", index=False)
    return table


def build_probe_matrix(gse) -> pd.DataFrame:
    """Probes x samples matrix from every fetched sample's own data table
    (ID_REF -> VALUE). Samples with no data table (e.g. RNA-seq, handled via
    download.py's supplementary-file path instead) are skipped.
    """
    columns = {}
    for gsm_id, gsm in gse.gsms.items():
        table = getattr(gsm, "table", None)
        if table is None or table.empty or "ID_REF" not in table.columns or "VALUE" not in table.columns:
            continue
        values = pd.to_numeric(table.set_index("ID_REF")["VALUE"], errors="coerce")
        columns[gsm_id] = values
    if not columns:
        return pd.DataFrame()
    return pd.DataFrame(columns)


def aggregate_probes_to_genes(probe_matrix: pd.DataFrame, probe_gene_map: pd.DataFrame, agg: str = "mean") -> pd.DataFrame:
    """Probes x samples -> genes x samples.

    Groups every probe (or transcript, already resolved to its gene upstream
    in extract_probe_gene_table) mapping to the same gene and aggregates
    their values -- this single step is also what summarizes transcripts of
    the same gene before the gene-level value is produced. Unmapped probes
    are dropped. Genes are keyed by entrez_id when available, falling back
    to gene_symbol for platforms/probes that only carry a symbol.
    """
    if probe_matrix.empty or probe_gene_map.empty:
        return pd.DataFrame(columns=["entrez_id", "gene_symbol"])

    mapped = probe_gene_map[probe_gene_map["source"] != "unmapped"].copy()
    mapped = mapped[mapped["gene_symbol"].notna() | mapped["entrez_id"].notna()]
    if mapped.empty:
        return pd.DataFrame(columns=["entrez_id", "gene_symbol"])

    mapped["gene_key"] = mapped["entrez_id"].where(mapped["entrez_id"].notna(), mapped["gene_symbol"])
    mapped = mapped.set_index("probe_id")

    joined = probe_matrix.join(mapped[["gene_key", "gene_symbol", "entrez_id"]], how="inner")
    if joined.empty:
        return pd.DataFrame(columns=["entrez_id", "gene_symbol"])

    sample_cols = list(probe_matrix.columns)
    grouped_values = joined.groupby("gene_key")[sample_cols].agg(agg)
    gene_labels = joined.groupby("gene_key")[["gene_symbol", "entrez_id"]].first()
    return gene_labels.join(grouped_values).reset_index(drop=True)
