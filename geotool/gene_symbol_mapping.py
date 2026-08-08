"""Convert an RNA-seq expression matrix's row identifiers (Ensembl transcript
or gene IDs) to HUGO/HGNC gene symbols, using the GENCODE reference tables
built by data/references/build_gencode_reference.py.

Pipeline (see convert_to_gene_symbols):
1. Locate the matrix's identifier axis (its index, or whichever non-numeric
   column looks like one) and classify it as Ensembl transcript (ENST...),
   Ensembl gene (ENSG...), already a gene symbol, or unrecognized (e.g. a
   Cufflinks "XLOC_..." novel-locus ID, or a custom repeat-annotation
   scheme) -- the last case is left alone, never guessed at.
2. Refuse outright if the matrix has any negative values: summing across
   possibly-log-transformed or otherwise non-additive data (e.g. a
   log-fold-change table) would be mathematically wrong, and GEO submitters
   overwhelmingly publish counts/TPM/FPKM/CPM in non-negative linear scale,
   so a negative value here is a real red flag, not routine noise.
3. Transcript-level: map each transcript to a gene symbol via the *clean*,
   filtered transcript->gene map (protein-coding, CCDS-backed, minimum
   length -- see build_gencode_reference.py), not the raw unfiltered one.
4. Gene-level: map Ensembl gene ID -> symbol directly.
5. Either way (including the already-a-symbol case): group by the resulting
   gene_symbol column and sum -- this is also what collapses duplicate rows
   (e.g. multiple transcripts that were relabeled with their gene's symbol
   but never actually aggregated, or two Ensembl gene IDs that happen to
   share one symbol) into one row per gene.

compute_tpm is a separate, optional final step for raw-count matrices only.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Literal, NamedTuple

import pandas as pd

from geotool import config

_VERSION_SUFFIX_RE = re.compile(r"\.\d+$")
_ENST_RE = re.compile(r"^ENST\d+$")
_ENSG_RE = re.compile(r"^ENSG\d+$")

# Kallisto/Salmon-style transcript FASTA headers pack several fields into one
# pipe-delimited string: "ENST00000456328.2|ENSG00000223972.5|OTTHUMG...|
# OTTHUMT...|transcript_name|gene_name|length|biotype" -- live example,
# GSE264630. _composite_leading_id extracts just the leading ENST/ENSG field
# so this classifies/maps like a normal identifier rather than "unknown".
_COMPOSITE_ID_RE = re.compile(r"^(ENS[TG]\d+(?:\.\d+)?)\|")

# Fraction of a sampled identifier axis that must match the ENST/ENSG regex
# before trusting the classification -- these are exact pattern matches, so
# a high bar is fine; a handful of stray malformed IDs shouldn't flip it.
MIN_ID_MATCH_FRACTION = 0.5

# A real, whole-transcriptome gene-symbol list only ever partially overlaps
# any *single* reference's own current nomenclature -- verified live against
# GSE273376's real (legitimate, submitter-supplied) gene_symbol column: only
# ~58% of its ~60k symbols are an exact literal match to GENCODE v50's own
# gene_symbol field (symbol aliases, withdrawn/renamed HGNC entries, and
# genuinely reference-specific gene sets are all expected and normal). A
# genuinely unrelated identifier scheme (Cufflinks XLOC_ loci, custom
# repeat-annotation IDs) scores a clean 0% in the same check, so this can
# stay far below MIN_ID_MATCH_FRACTION without risking a false positive.
MIN_SYMBOL_MATCH_FRACTION = 0.2

# Sampling every ID in a 50k-row matrix to classify its identifier axis is
# wasted work; a few hundred is already a robust majority-vote sample --
# must be a *random* sample, not the first N: real files are routinely
# sorted (e.g. alphabetically), and a contiguous slice of a sorted gene list
# is not representative of the whole (verified live: the first 500 rows of
# GSE273376's alphabetically-sorted, ~58%-matching symbol list scored a
# misleadingly low 31%, purely from landing in an unlucky alphabetic range).
_CLASSIFY_SAMPLE_SIZE = 500
_RANDOM_SEED = 0

IdentifierType = Literal["transcript", "gene", "symbol", "unknown"]


def strip_version(value) -> str:
    """"ENSG00000000003.18" -> "ENSG00000000003" -- Ensembl base IDs are
    stable across releases, only the trailing version increments, and a
    cohort's own GENCODE/Ensembl version rarely matches this reference's.
    """
    return _VERSION_SUFFIX_RE.sub("", str(value))


def canonical_id(value) -> str:
    """strip_version, plus first unwrapping a Kallisto/Salmon-style
    pipe-delimited composite header down to its leading ENST/ENSG field (see
    _COMPOSITE_ID_RE) -- the form a cohort's own *identifier axis* values
    come in. Reference-side IDs (already single, clean ENST/ENSG strings
    straight from a GENCODE table) only ever need strip_version, not this.
    """
    text = str(value)
    match = _COMPOSITE_ID_RE.match(text)
    return strip_version(match.group(1) if match else text)


# Some RSEM-based quantification pipelines double-encode a gene symbol as
# its own row id instead of emitting a plain symbol -- "ACCSL_ACCSL", or,
# for the Nth copy of a multi-copy gene family sharing one symbol,
# "<symbol>_<copy index>_<symbol>" ("5S_rRNA_10_5S_rRNA") -- live example,
# GSE174615, where this shape covers 93.7% of a 30,373-row matrix's raw
# IDs. Neither form is ENST/ENSG, nor a literal match against any known
# symbol (the doubled string itself isn't a real gene symbol), so unless
# unwrapped first the whole axis reads as "unknown" to
# detect_identifier_type and the cohort is skipped outright rather than
# read as the plain gene-symbol matrix it actually is.
_DOUBLED_SYMBOL_RE = re.compile(r"^(.+)_(?:\d+_)?\1$")


def undouble_repeated_symbol_ids(values) -> pd.Series:
    """Strip "SYMBOL_SYMBOL"/"SYMBOL_N_SYMBOL" doubling (see
    _DOUBLED_SYMBOL_RE) down to the single symbol, positionally aligned
    (fresh default RangeIndex, same as detect_identifier_type/
    convert_to_gene_symbols expect from an identifier axis). Only unwraps
    when a majority (>50%) of `values` actually match that shape -- same
    majority-vote guard the ENST/ENSG embedded-id and Kallisto/Salmon
    composite-header handling use elsewhere in this codebase -- so a matrix
    with a handful of coincidentally self-repeating IDs isn't corrupted.
    Values that don't match the shape pass through unchanged either way:
    e.g. GSE174615's own miRNA "MI0000060_hsa-let-7a-1" accessions,
    "ENSG00000198695_MT-ND6"-style embedded-ENSG rows, and
    "Em:AC005003.4_AC005003.4" clone IDs -- none of which are
    protein-coding genes the clean GENCODE reference would keep anyway.
    """
    series = pd.Series([str(v) for v in values])
    extracted = series.str.extract(_DOUBLED_SYMBOL_RE, expand=False)
    if extracted.notna().mean() > 0.5:
        return extracted.fillna(series)
    return series


class GencodeReference(NamedTuple):
    version: str
    transcript_to_symbol: dict[str, str]  # stripped ENST -> symbol (clean/filtered set only)
    gene_to_symbol: dict[str, str]  # stripped ENSG -> symbol (every annotated gene)
    gene_length: dict[str, int]  # stripped ENSG -> representative length in bp, for TPM; {} if unavailable
    known_symbols: frozenset[str]


def load_gencode_reference(version: str, references_dir: Path | None = None) -> GencodeReference:
    """Load a GENCODE release's reference tables (data/references/gencode<version>/,
    from build_gencode_reference.py). Only gencode50 currently ships the
    filtered "clean" transcript map and per-transcript lengths; an older
    release lacking them (e.g. gencode32) still loads for the gene-level
    (ENSG) path, just with an empty gene_length (no TPM) and the raw,
    unfiltered id2gene table standing in for the clean transcript map.
    """
    base = (references_dir or config.REFERENCES_DIR) / f"gencode{version}"

    ensg2hugo = pd.read_csv(base / f"ensg2hugo_gencode_v{version}.tsv.gz", sep="\t")
    gene_to_symbol = dict(zip(ensg2hugo["ID"].map(strip_version), ensg2hugo["Gene"]))

    clean_path = base / f"clean_transcript_gene_symbol_v{version}.tsv.gz"
    if clean_path.exists():
        clean = pd.read_csv(clean_path, sep="\t")
        transcript_to_symbol = dict(zip(clean["transcript_id"].map(strip_version), clean["gene_symbol"]))
    else:
        id2gene = pd.read_csv(base / f"id2gene_gencode_v{version}.tsv.gz", sep="\t")
        transcript_to_symbol = dict(zip(id2gene["ID"].map(strip_version), id2gene["Gene"]))

    gene_length: dict[str, int] = {}
    annotation_path = base / f"transcript_annotation_v{version}.tsv.gz"
    if annotation_path.exists():
        tx = pd.read_csv(annotation_path, sep="\t", usecols=["gene_id", "transcript_length", "included"])
        included = tx[tx["included"]]
        gene_length = (
            included.assign(gene_id=included["gene_id"].map(strip_version))
            .groupby("gene_id")["transcript_length"]
            .median()
            .round()
            .astype(int)
            .to_dict()
        )

    known_symbols = frozenset(gene_to_symbol.values()) | frozenset(transcript_to_symbol.values())
    return GencodeReference(version, transcript_to_symbol, gene_to_symbol, gene_length, known_symbols)


def _match_fraction(values: list[str], pattern: re.Pattern) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if pattern.match(v)) / len(values)


def detect_identifier_type(ids, reference: GencodeReference) -> IdentifierType:
    """Classify a *random* sample of `ids` (any iterable of raw identifier
    strings, capped at _CLASSIFY_SAMPLE_SIZE -- a whole-transcriptome matrix
    can have tens of thousands of rows) as Ensembl transcript, Ensembl gene,
    an already-known gene symbol, or unrecognized.
    """
    all_values = [str(v) for v in ids]
    if len(all_values) > _CLASSIFY_SAMPLE_SIZE:
        sample = random.Random(_RANDOM_SEED).sample(all_values, _CLASSIFY_SAMPLE_SIZE)
    else:
        sample = all_values

    stripped = [canonical_id(v) for v in sample]
    if _match_fraction(stripped, _ENST_RE) >= MIN_ID_MATCH_FRACTION:
        return "transcript"
    if _match_fraction(stripped, _ENSG_RE) >= MIN_ID_MATCH_FRACTION:
        return "gene"
    if sample:
        known_fraction = sum(1 for v in sample if v in reference.known_symbols) / len(sample)
        if known_fraction >= MIN_SYMBOL_MATCH_FRACTION:
            return "symbol"
    return "unknown"


_ID_TYPE_PRIORITY = {"transcript": 0, "gene": 1, "symbol": 2}


def locate_identifier_axis(matrix: pd.DataFrame, reference: GencodeReference) -> tuple[pd.Series, IdentifierType] | None:
    """Find the matrix's identifier axis -- its index if it's not a plain
    positional RangeIndex, or whichever non-numeric column looks like one --
    and classify it. A submitter's file sometimes carries more than one
    candidate (e.g. both "Gene_stable_ID" and "Gene_name"); transcript/gene
    IDs are preferred over an already-symbol column so every cohort maps
    through the same canonical GENCODE symbols rather than whatever ad hoc
    naming each submitter used. None if nothing recognizable was found.
    """
    candidates: list[tuple[pd.Series, IdentifierType]] = []

    if not isinstance(matrix.index, pd.RangeIndex):
        ids = undouble_repeated_symbol_ids(matrix.index)
        id_type = detect_identifier_type(ids, reference)
        if id_type != "unknown":
            candidates.append((ids, id_type))

    for col in matrix.columns:
        if pd.api.types.is_numeric_dtype(matrix[col]):
            continue
        ids = undouble_repeated_symbol_ids(matrix[col])
        id_type = detect_identifier_type(ids, reference)
        if id_type != "unknown":
            candidates.append((ids, id_type))

    if not candidates:
        return None
    return min(candidates, key=lambda c: _ID_TYPE_PRIORITY[c[1]])


def convert_to_gene_symbols(matrix: pd.DataFrame, reference: GencodeReference) -> tuple[pd.DataFrame | None, str]:
    """Convert `matrix` (any orientation genotool already guarantees: genes/
    transcripts as rows, samples as columns) to one row per HUGO gene symbol,
    linear-scale values summed across every transcript/row that maps to it.

    Returns (converted, note). converted is None when nothing could be done
    at all (no recognizable identifier axis, no numeric data, or negative
    values present) -- note always explains what happened either way.
    """
    numeric = matrix.select_dtypes(include="number")
    if numeric.empty:
        return None, "no numeric sample columns found"

    if (numeric < 0).to_numpy().any():
        return None, (
            "matrix has negative values -- refusing to sum across possibly-log-transformed "
            "or non-additive data"
        )

    located = locate_identifier_axis(matrix, reference)
    if located is None:
        return None, "no recognizable transcript/gene/symbol identifier found -- not converted"
    ids, id_type = located

    if id_type == "symbol":
        symbols = ids.astype(str)
        note_prefix = "already gene symbols"
    else:
        mapping = reference.transcript_to_symbol if id_type == "transcript" else reference.gene_to_symbol
        symbols = ids.astype(str).map(canonical_id).map(mapping)
        note_prefix = f"converted from {id_type} identifiers via GENCODE v{reference.version}"

    working = numeric.reset_index(drop=True).copy()
    working.insert(0, "gene_symbol", symbols.values)
    n_unmapped = int(working["gene_symbol"].isna().sum())
    working = working.dropna(subset=["gene_symbol"])

    n_before = len(working)
    collapsed = working.groupby("gene_symbol", as_index=False).sum(numeric_only=True)
    n_after = len(collapsed)

    note = note_prefix
    if n_unmapped:
        note += f"; {n_unmapped} unmapped row(s) dropped"
    if n_after < n_before:
        note += f"; collapsed {n_before - n_after} duplicate gene-symbol row(s) via sum"

    return collapsed, note


def compute_tpm(matrix: pd.DataFrame, reference: GencodeReference, gene_symbol_col: str = "gene_symbol") -> tuple[pd.DataFrame | None, str]:
    """TPM-normalize a raw-count, gene-symbol-indexed matrix (the output of
    convert_to_gene_symbols) using GENCODE's per-gene representative
    transcript length (reference.gene_length -- the median length among a
    gene's "clean"/included transcripts; see load_gencode_reference).
    "Where possible": a gene symbol geotool can't resolve back to an Ensembl
    gene ID with known length is simply excluded, not treated as an error --
    everything else still gets a real TPM value.
    """
    if not reference.gene_length:
        return None, f"no gene-length data available for GENCODE v{reference.version} -- TPM not computed"

    # A gene symbol maps to length via gene_to_symbol's own inverse -- build
    # it once, lazily, rather than requiring callers to pass it in.
    symbol_to_length: dict[str, int] = {}
    for ensg, symbol in reference.gene_to_symbol.items():
        length = reference.gene_length.get(ensg)
        if length is not None and symbol not in symbol_to_length:
            symbol_to_length[symbol] = length

    lengths = matrix[gene_symbol_col].map(symbol_to_length)
    known = lengths.notna()
    if not known.any():
        return None, "no gene-length data matched any gene symbol in this matrix -- TPM not computed"

    sample_cols = [c for c in matrix.columns if c != gene_symbol_col]
    rpk = matrix.loc[known, sample_cols].div(lengths[known] / 1000.0, axis=0)
    scale = rpk.sum(axis=0) / 1e6
    tpm = rpk.div(scale.replace(0, pd.NA), axis=1)

    result = matrix.loc[known, [gene_symbol_col]].copy()
    result[sample_cols] = tpm
    result = result.reset_index(drop=True)

    n_dropped = int((~known).sum())
    note = "TPM computed"
    if n_dropped:
        note += f"; {n_dropped} gene(s) without length data excluded"
    return result, note
