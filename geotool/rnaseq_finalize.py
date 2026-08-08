"""Finalize RNA-seq cohorts' gene-level expression matrices: convert to HUGO
gene symbols (via gene_symbol_mapping), restrict to the clean GENCODE
reference gene set (data/references/gencode<version>), then renormalize each
sample to a 1,000,000 composition (TPM-style) -- for every cohort under one
or more collection roots (e.g. data/pdac_cohorts, data/mtap_prmt5_cohorts)
that has a resolved primary expression matrix (geotool.download's own
expression_qc.json, written for every RNA-seq cohort it successfully
resolved a file for).

Writes <collection_root>/<GSE>/expression_final.tsv.gz -- the actual
analysis-ready matrix (HUGO gene symbols, clean GENCODE gene set only,
TPM-renormalized, log2(x+1)).

Row-identifier detection, ENST/ENSG->gene-symbol mapping, and duplicate-row
aggregation are all delegated to geotool.gene_symbol_mapping (built on this
same reference), not reimplemented here -- it already handles the ENST/
ENSG/symbol cascade robustly (including submitter oddities like Kallisto/
Salmon pipe-delimited composite headers, and RSEM's doubled-symbol ids).
This module adds only what that shared, general-purpose converter
deliberately doesn't do: restricting the *result* to the clean, protein-
coding/CCDS-filtered gene set regardless of which ID scheme the input used
(gene_symbol_mapping's ENSG/symbol paths intentionally map against its
*full* gene_to_symbol/known_symbols -- broader than this reference's clean
set -- since it's meant as a general-purpose converter, not a fixed
inclusion policy), the raw-count-vs-already-normalized branch (compute_tpm
vs a plain sum-to-1e6 rescale -- rescaling already-length-normalized TPM/
FPKM/RPKM to sum 1e6 *is* the standard FPKM/RPKM->TPM conversion), and the
per-cohort file resolution / skip reporting.

Quantification unit comes from expression_qc.json's primary_expression_unit
(tpm/fpkm/rpkm need no length step; count/cpm do, via compute_tpm's
per-gene median included-transcript length). A cohort missing an
expression_qc.json entirely, or a multi-file cohort (unit present but
null), is skipped and reported, never guessed at -- there's no single
resolved matrix to guess about in the first place. A matrix whose row
identifiers can't be resolved to any gene at all (e.g. Cufflinks XLOC_
novel-locus ids with no stable cross-reference) is reported as failed, not
skipped -- that's an unrecoverable gene-identity problem for the cohort,
not a transient/parseable condition.

A resolved matrix whose unit is specifically "unknown" (geotool.download
verified it's a real gene-expression matrix by content, but neither the
filename nor the content told it what unit the values are in -- e.g.
GSE230065's "..._genes.tsv.gz") gets a best-effort default instead of being
skipped outright (see _guess_unit): integer values (the one unit that's
always a whole number) -> raw counts; otherwise transcript-level
identifiers -> CPM; otherwise (gene-level or already-symbol identifiers)
-> FPKM. This is a guess, not a verified fact -- every such cohort's report
row says so explicitly (and expression_final.tsv.gz's own transform_note),
so it's never presented as equivalent to a submitter-labeled unit.

Whether the matrix on disk is already log2(x+1)-scale or still linear is
detected per file with the same heuristic the rest of this codebase uses
(probe_mapping.needs_log2_transform: any value over 50 means still linear)
rather than assumed from which collection it came from -- different
collections can hold a mix of already-log2 and still-linear copies for
different cohorts. Output is re-log2(x+1)-transformed before writing, to
match this codebase's storage convention for processed expression matrices.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from geotool import config
from geotool import download as download_mod
from geotool import gene_symbol_mapping as gsm
from geotool import probe_mapping
from geotool import sample_id_matching

# fpkm/rpkm/tpm are already length-normalized -- rescaling them to sum to
# 1e6 per sample *is* the standard FPKM/RPKM->TPM conversion. count/cpm are
# not length-normalized at all (cpm only did the "per million" rescaling
# step), so both need compute_tpm's length-division for the result to be a
# real TPM rather than just a filtered-and-rescaled CPM.
_LENGTH_NORM_UNITS = {"count", "cpm"}

# gene_symbol_mapping.canonical_id unwraps Kallisto/Salmon-style *pipe*-
# delimited composite headers ("ENST...|ENSG...|..."), but not other
# submitter-specific compound formats -- live example, GSE79668:
# "RP1-67K17.4_ENSG00000237851.1" (a gene-symbol prefix, underscore, then
# the ENSG id+version). That row's identifier axis reads as "unknown" to
# gene_symbol_mapping.detect_identifier_type as-is. This extracts an
# embedded ENST/ENSG substring from anywhere in the index (not just a
# leading pipe-delimited field) as a preprocessing step, rather than
# broadening the shared module's own composite-ID handling for one cohort's
# one-off format.
_EMBEDDED_ENS_ID_RE = re.compile(r"(ENS[TG]\d+(?:\.\d+)?)")


def load_clean_symbols(clean_genes_path: Path) -> set[str]:
    clean = pd.read_csv(clean_genes_path, sep="\t")
    return set(clean["gene_symbol"].unique())


def unwrap_embedded_ensembl_ids(matrix: pd.DataFrame) -> pd.DataFrame:
    """If >50% of the index contains an embedded ENST/ENSG id, replace the
    index with just that extracted substring; otherwise return unchanged.
    """
    extracted = matrix.index.to_series().astype(str).str.extract(_EMBEDDED_ENS_ID_RE, expand=False)
    if extracted.notna().mean() > 0.5:
        m = matrix.copy()
        m.index = extracted.fillna(pd.Series(matrix.index.astype(str), index=extracted.index))
        return m
    return matrix


def to_linear_scale(matrix: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Return (linear_matrix, was_log2) -- inverse-transforms log2(x+1) back
    to linear scale only if the data actually looks log2-scale (same
    >50-means-still-linear heuristic as the rest of this codebase), since a
    collection root can hold a mix of already-log2 and still-linear files
    for different cohorts.
    """
    if probe_mapping.needs_log2_transform(matrix):
        return matrix.clip(lower=0), False
    return (2 ** matrix - 1).clip(lower=0), True


def looks_like_integer_data(matrix: pd.DataFrame) -> bool:
    """True if every finite value in matrix is (within float noise of) a
    whole number -- the one property raw read counts always have (an
    already length- or depth-normalized unit like TPM/FPKM/CPM essentially
    never produces an all-integer matrix by chance), and this module's
    primary signal for guessing an unlabeled quantification unit (see
    _guess_unit). Call on a *linear*-scale matrix (post to_linear_scale) --
    log2(x+1)-transformed counts are themselves fractional.
    """
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    return bool(np.allclose(finite, np.round(finite), atol=1e-6))


def find_original_raw_file(path: Path) -> Path | None:
    """The true, untransformed raw file geotool.download preserved
    alongside path when probe_mapping.normalize_expression_matrix (log2/
    orientation/format fixes, applied to every resolved primary file)
    changed it in place -- "<stem>.original.tsv.gz" next to path (see
    download._write_normalized_expression_matrix). None if no such backup
    exists: path was never modified from what the submitter published (so
    path's own values already are the raw ones), or it doesn't live in
    this same directory.
    """
    candidate = path.parent / f"{download_mod._strip_known_extensions(path.name)}.original.tsv.gz"
    return candidate if candidate.exists() else None


def guess_unit(
    linear_matrix: pd.DataFrame, ref: gsm.GencodeReference, integer_check_matrix: pd.DataFrame | None = None,
) -> tuple[str, str]:
    """Best-effort default quantification unit for a matrix whose
    expression_qc.json recorded primary_expression_unit as "unknown"
    (geotool.download verified it's a real gene-expression matrix by
    content, but couldn't determine its unit from filename or content
    alone). Returns (unit, explanation) -- integer values -> "count"
    (raw counts are always whole numbers); otherwise transcript-level
    identifiers -> "cpm"; otherwise (gene-level, or already gene symbols)
    -> "fpkm". A guess, not a verified fact -- callers should say so.

    integer_check_matrix, if given, is used instead of linear_matrix
    purely for the integer-values check. Pass this when linear_matrix was
    itself reconstructed by inverting an already-log2 file (2**x - 1):
    exponentiating a value that was rounded in log2 space reintroduces
    error large enough, at real gene-expression magnitudes, to erase
    whether the true underlying values were integers -- verified live on
    GSE230065's on-disk log2 file (rounded to 3 decimals): reconstructing
    its largest counts lands up to ~63 away from their true integer value,
    a dead giveaway (find_original_raw_file's ".original.tsv.gz", the
    exact pre-transform raw file geotool.download preserves whenever it
    changes a file in place) is the right thing to pass here when it
    exists -- no reconstruction involved, so no such error.
    """
    if looks_like_integer_data(integer_check_matrix if integer_check_matrix is not None else linear_matrix):
        return "count", "all values are (near-)whole numbers, so assumed raw counts"
    located = gsm.locate_identifier_axis(linear_matrix, ref)
    id_type = located[1] if located else "gene"
    if id_type == "transcript":
        return "cpm", "non-integer values with transcript-level identifiers, so assumed CPM"
    return "fpkm", "non-integer values with gene-level identifiers, so assumed FPKM"


def renormalize_to_1e6(gene_matrix: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    col_sums = gene_matrix.sum(axis=0)
    zero_cols = col_sums[col_sums <= 0].index.tolist()
    if zero_cols:
        return None, f"{len(zero_cols)} sample(s) sum to zero after filtering, cannot renormalize: {zero_cols[:5]}"
    return gene_matrix.div(col_sums, axis=1) * 1e6, None


def restrict_to_clean_genes(gene_matrix: pd.DataFrame, clean_symbols: set[str]) -> pd.DataFrame | None:
    kept = gene_matrix[gene_matrix.index.isin(clean_symbols)]
    if kept.empty:
        return None
    # Defensive: convert_to_gene_symbols/compute_tpm already group by
    # gene_symbol and sum, but restricting rows here can't itself introduce
    # duplicates -- a plain reindex would be enough. groupby-sum kept only
    # for robustness against any future upstream change in that guarantee.
    return kept.groupby(level=0).sum()


def find_local_expression_file(cohort_dir: Path, qc: dict) -> Path | None:
    fname = qc.get("primary_expression_file")
    if not fname:
        return None
    basename = Path(fname).name
    expr_dir = cohort_dir / "expression"
    candidate = expr_dir / basename
    if candidate.exists():
        return candidate
    matches = list(expr_dir.glob(basename)) if expr_dir.exists() else []
    return matches[0] if matches else None


SAMPLE_ID_MAP_ANNOTATION_COLUMNS = ["expression_id", "sample_id_match_method", "sample_id_match_confidence"]


def merge_sample_id_map_into_series_annotation(
    gse_id: str, id_map: pd.DataFrame, series_dir: Path | None = None,
) -> None:
    """Write sample_id_matching's per-sample match (expression_id,
    match_method, confidence -- renamed sample_id_match_method/
    sample_id_match_confidence here to read unambiguously once merged)
    onto this cohort's own *canonical* data/series/<gse_id>/annotation.tsv,
    joined on gsm_id. Not the project-specific collection-root copy
    write_sample_id_map read from -- geotool.harmonize's reuse tier always
    reads a cohort's annotation.tsv from data/series/<gse_id>/, so writing
    there is what actually makes these columns show up the next time
    cohorts get merged, with no changes needed in harmonize.py itself
    (outer-concat already NaN-fills a column a given cohort doesn't have).

    A no-op if data/series/<gse_id>/annotation.tsv doesn't exist or has no
    gsm_id column. Idempotent: re-running replaces any previous run's
    columns rather than duplicating them. An id_map row with no gsm_id
    (unmatched expression column) has nothing to join onto, so contributes
    nothing here -- not an error, just nothing added for that row.
    """
    series_dir = series_dir or config.SERIES_DIR
    annotation_path = series_dir / gse_id / "annotation.tsv"
    if not annotation_path.exists():
        return
    annotation = pd.read_csv(annotation_path, sep="\t", low_memory=False)
    if "gsm_id" not in annotation.columns:
        return

    to_merge = id_map.dropna(subset=["gsm_id"])[["gsm_id", "expression_id", "match_method", "confidence"]].rename(
        columns={"match_method": "sample_id_match_method", "confidence": "sample_id_match_confidence"}
    )
    annotation = annotation.drop(columns=[c for c in SAMPLE_ID_MAP_ANNOTATION_COLUMNS if c in annotation.columns])
    merged = annotation.merge(to_merge, on="gsm_id", how="left")
    merged.to_csv(annotation_path, sep="\t", index=False)


def write_sample_id_map(
    cohort_dir: Path, expression_columns: list[str], series_dir: Path | None = None,
) -> pd.DataFrame | None:
    """Match expression_columns (an expression matrix's sample columns, in
    their original submitter-chosen labels) back to this cohort's own
    annotation.tsv gsm_ids (see geotool.sample_id_matching), write the
    result to <cohort_dir>/sample_id_map.tsv, and merge it onto this
    cohort's canonical data/series/<gse_id>/annotation.tsv too (see
    merge_sample_id_map_into_series_annotation). None (nothing written) if
    there's no local annotation.tsv or it has no gsm_id column to match
    against.

    Run this before final cohort harmonization (geotool.harmonize) needs to
    join expression data back to sample annotation -- by the time a matrix
    is "finalized" here, this mapping should already exist rather than being
    reconstructed ad hoc downstream. A best-effort match: see match_method/
    confidence in the output for how much to trust each row: an "unmatched"
    gsm_id (None) means neither a name-based nor positional match could be
    made with any confidence, not that the sample doesn't exist.
    """
    annotation_path = cohort_dir / "annotation.tsv"
    if not annotation_path.exists():
        return None
    annotation = pd.read_csv(annotation_path, sep="\t", low_memory=False)
    if "gsm_id" not in annotation.columns:
        return None
    id_map = sample_id_matching.match_expression_columns(expression_columns, annotation)
    id_map.to_csv(cohort_dir / "sample_id_map.tsv", sep="\t", index=False)
    merge_sample_id_map_into_series_annotation(cohort_dir.name, id_map, series_dir=series_dir)
    return id_map


def finalize_cohort(
    cohort_dir: Path, ref: gsm.GencodeReference, clean_symbols: set[str], series_dir: Path | None = None,
) -> dict:
    """Finalize one cohort's expression matrix, writing
    <cohort_dir>/expression_final.tsv.gz on success. Returns a report row
    dict: status is "processed" (file written -- unit "unknown" gets a
    best-effort default rather than blocking this, see guess_unit),
    "skipped" (a transient/expected condition -- multi-file cohort, zero-sum
    sample after filtering, ...), or "failed" (an unrecoverable gene-identity
    problem for this cohort's data).

    Also writes <cohort_dir>/sample_id_map.tsv and merges it onto this
    cohort's canonical data/series/<gse_id>/annotation.tsv (see
    write_sample_id_map) as soon as the matrix's sample columns are known --
    independent of whether gene-symbol conversion below ultimately
    succeeds, since the column<->gsm_id correspondence is useful diagnostic
    information on its own even for a cohort that ends up skipped/failed
    here. series_dir overrides where that canonical annotation.tsv lives
    (default data/series/, see config.SERIES_DIR) -- mainly for tests.
    """
    gse_id = cohort_dir.name
    qc_path = cohort_dir / "expression_qc.json"
    if not qc_path.exists():
        return {"gse_id": gse_id, "status": "skipped", "reason": "no expression_qc.json (not RNA-seq, or no resolvable expression matrix)"}

    qc = json.loads(qc_path.read_text())
    unit = qc.get("primary_expression_unit")
    if unit is None:
        return {"gse_id": gse_id, "status": "skipped", "reason": "no single primary expression matrix for this cohort (multi-file case)"}

    path = find_local_expression_file(cohort_dir, qc)
    if path is None:
        return {"gse_id": gse_id, "status": "skipped", "reason": f"resolved file {Path(qc.get('primary_expression_file', '')).name!r} not found locally under {cohort_dir}/expression/"}

    try:
        matrix = pd.read_csv(path, sep="\t", index_col=0)
    except Exception as e:
        return {"gse_id": gse_id, "status": "skipped", "reason": f"could not parse {path.name}: {e}"}

    numeric = matrix.select_dtypes(include="number")
    if numeric.empty:
        return {"gse_id": gse_id, "status": "skipped", "reason": f"{path.name}: no numeric sample columns"}

    id_map = write_sample_id_map(cohort_dir, list(numeric.columns), series_dir=series_dir)

    linear, was_log2 = to_linear_scale(numeric)
    if float(linear.min().min()) < -1e-3:
        return {"gse_id": gse_id, "status": "skipped", "reason": f"{path.name}: negative values remain after scale detection -- not a clean expression matrix"}

    linear = unwrap_embedded_ensembl_ids(linear)

    unit_note = ""
    if unit == "unknown":
        integer_check_matrix = None
        original_path = find_original_raw_file(path)
        if original_path is not None:
            try:
                integer_check_matrix = pd.read_csv(original_path, sep="\t", index_col=0).select_dtypes(include="number")
            except Exception:
                integer_check_matrix = None
        unit, why = guess_unit(linear, ref, integer_check_matrix=integer_check_matrix)
        unit_note = f"quantification unit unknown in expression_qc.json -- {why}"

    converted, convert_note = gsm.convert_to_gene_symbols(linear, ref)
    if converted is None:
        # "no recognizable transcript/gene/symbol identifier found" (see
        # gene_symbol_mapping.convert_to_gene_symbols) means the matrix's row
        # index is neither ENSG/ENST nor a HUGO symbol and locate_identifier_axis
        # couldn't resolve it to one either -- not a transient/skippable
        # condition like an unknown quantification unit, but an unrecoverable
        # gene-identity failure for this series. Live example: GSE163305, whose
        # Cufflinks output is indexed by XLOC_ novel-locus ids with no stable
        # cross-reference to any gene.
        if "no recognizable" in convert_note:
            return {"gse_id": gse_id, "status": "failed", "reason": "gene names unrecoverable"}
        reason = f"{path.name}: {convert_note}"
        return {"gse_id": gse_id, "status": "skipped", "reason": f"{unit_note}; {reason}" if unit_note else reason}

    if unit in _LENGTH_NORM_UNITS:
        tpm_df, tpm_note = gsm.compute_tpm(converted, ref)
        if tpm_df is None:
            reason = f"{path.name}: {tpm_note}"
            return {"gse_id": gse_id, "status": "skipped", "reason": f"{unit_note}; {reason}" if unit_note else reason}
        gene_matrix = tpm_df.set_index("gene_symbol")
        detail = f"{convert_note}; {tpm_note}"
    else:
        gene_matrix = converted.set_index("gene_symbol")
        detail = convert_note

    clean_matrix = restrict_to_clean_genes(gene_matrix, clean_symbols)
    if clean_matrix is None:
        reason = f"{path.name}: none of the mapped genes are in the clean reference gene set"
        return {"gse_id": gse_id, "status": "skipped", "reason": f"{unit_note}; {reason}" if unit_note else reason}

    tpm, err = renormalize_to_1e6(clean_matrix)
    if tpm is None:
        reason = f"{path.name}: {err}"
        return {"gse_id": gse_id, "status": "skipped", "reason": f"{unit_note}; {reason}" if unit_note else reason}

    out = probe_mapping.maybe_log2_transform(tpm)
    out_path = cohort_dir / "expression_final.tsv.gz"
    out.round(3).to_csv(out_path, sep="\t")

    n_matched = int(id_map["gsm_id"].notna().sum()) if id_map is not None else None
    n_id_map = len(id_map) if id_map is not None else None
    if unit_note:
        detail = f"{unit_note}; {detail}"

    return {
        "gse_id": gse_id, "status": "processed", "reason": f"{detail}; {len(tpm)} clean genes kept",
        "unit": unit, "source_file": path.name, "was_log2_on_disk": was_log2,
        "n_genes": len(tpm), "n_samples": tpm.shape[1], "out_file": str(out_path),
        "n_samples_matched_to_gsm": n_matched, "n_samples_in_id_map": n_id_map,
    }


def build_final_matrices(
    cohort_roots: list[Path], gencode_version: str = "50", references_dir: Path | None = None,
    series_dir: Path | None = None,
) -> pd.DataFrame:
    """Finalize every cohort under each root in cohort_roots (a root's
    immediate GSE* subdirectories), writing expression_final.tsv.gz per
    cohort as a side effect. Returns a one-row-per-cohort report DataFrame
    (status/reason plus finalize_cohort's other fields). series_dir
    overrides where each cohort's canonical annotation.tsv lives for the
    sample-id-map merge (default data/series/, see config.SERIES_DIR).
    """
    ref = gsm.load_gencode_reference(gencode_version, references_dir=references_dir)
    clean_genes_path = (references_dir or config.REFERENCES_DIR) / f"gencode{gencode_version}" / f"clean_transcript_gene_symbol_v{gencode_version}.tsv.gz"
    clean_symbols = load_clean_symbols(clean_genes_path)

    report = []
    for root in cohort_roots:
        root = Path(root)
        for cohort_dir in sorted(p for p in root.glob("GSE*") if p.is_dir()):
            report.append(finalize_cohort(cohort_dir, ref, clean_symbols, series_dir=series_dir))

    return pd.DataFrame(report)
