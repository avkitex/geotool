"""Finalize microarray cohorts' gene-level expression matrices: restrict to
the clean GENCODE reference gene set (data/references/gencode<version>) --
the same clean-set policy geotool.rnaseq_finalize applies to RNA-seq -- for
every cohort under one or more collection roots that has a resolved,
ready (expression_status == "ok") single-sample gene-level matrix.

Unlike RNA-seq, microarray never needs gene-ID conversion or unit/TPM
handling here: geotool.download already maps each platform's own probes to
HUGO gene symbols and applies the log2 transform at download time
(geotool.probe_mapping.aggregate_probes_to_genes / maybe_log2_transform),
so by the time a cohort reaches this module its expression.tsv.gz (or, for
a two-channel cohort with a resolved signal channel, its
channel_signal_expression.tsv.gz) is already a single per-sample
gene-symbol-indexed matrix in a comparable log2 scale. There is no "TPM"
or analogous compositional-renormalization concept for hybridization-
intensity data -- this step is restrict-only, not restrict-and-renormalize.

A cohort whose own expression_status isn't "ok" is skipped outright, not
processed on a best-effort basis: "not_available"/"unparseable"/etc. means
no usable matrix at all, and "two_channel_signal_unresolved" specifically
means the only thing on disk is a Cy3/Cy5 ratio with no way to recover
which channel is the actual tumor/signal measurement (see
clinical_annotate.classify_expression_status) -- restricting that ratio's
genes wouldn't make it any more usable, so there's nothing worth writing.

Writes <collection_root>/<GSE>/expression_final.tsv.gz -- the same output
filename/location convention geotool.rnaseq_finalize uses, so
geotool.cohort_report's collection_root-based readiness check treats a
finalized microarray cohort exactly like a finalized RNA-seq one, with no
special-casing needed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from geotool import config
from geotool import rnaseq_finalize as rf


def _source_matrix_path(cohort_dir: Path) -> tuple[Path | None, str]:
    """Prefer channel_signal_expression.tsv.gz (resolved two-channel
    signal) over the cohort's own expression.tsv.gz -- same priority
    cohort_report._resolve_own_expression_file already uses, since a
    two-channel cohort's expression.tsv.gz is the Cy3/Cy5 ratio, not a
    per-sample measurement (see probe_mapping.detect_reference_channel).
    Returns (None, "") if neither exists -- e.g. a platform with no gene
    symbol/ID column at all, where only probe_matrix.tsv.gz was ever
    written (see probe_mapping.py's five mapping-strategy docstring).
    """
    signal_path = cohort_dir / "channel_signal_expression.tsv.gz"
    if signal_path.exists():
        return signal_path, "channel_signal_expression.tsv.gz"
    expr_path = cohort_dir / "expression.tsv.gz"
    if expr_path.exists():
        return expr_path, "expression.tsv.gz"
    return None, ""


def _read_expression_status(gse_id: str, series_dir: Path | None) -> str | None:
    annotation_path = (series_dir or config.SERIES_DIR) / gse_id / "annotation.tsv"
    if not annotation_path.exists():
        return None
    annotation = pd.read_csv(annotation_path, sep="\t", low_memory=False, nrows=1)
    if "expression_status" not in annotation.columns or not len(annotation):
        return None
    return annotation["expression_status"].iloc[0]


def finalize_cohort(cohort_dir: Path, clean_symbols: set[str], series_dir: Path | None = None) -> dict:
    """Finalize one microarray cohort, writing
    <cohort_dir>/expression_final.tsv.gz on success. Returns a report row
    dict shaped like rnaseq_finalize.finalize_cohort's own (status
    "processed"/"skipped" -- microarray has no unrecoverable gene-identity
    failure mode analogous to RNA-seq's "failed", since probe->gene
    mapping already succeeded or didn't at download time).
    """
    gse_id = cohort_dir.name
    expression_status = _read_expression_status(gse_id, series_dir)
    if expression_status != "ok":
        return {
            "gse_id": gse_id, "status": "skipped",
            "reason": f"expression_status is {expression_status!r}, not 'ok' -- no resolved single-sample matrix",
        }

    path, source_name = _source_matrix_path(cohort_dir)
    if path is None:
        return {
            "gse_id": gse_id, "status": "skipped",
            "reason": "no expression.tsv.gz or channel_signal_expression.tsv.gz found here -- "
                      "gene-level mapping unavailable for this platform (probe-level matrix only, if any)",
        }

    try:
        matrix = pd.read_csv(path, sep="\t", index_col=0)
    except Exception as e:
        return {"gse_id": gse_id, "status": "skipped", "reason": f"could not parse {path.name}: {e}"}

    numeric = matrix.select_dtypes(include="number")
    if numeric.empty:
        return {"gse_id": gse_id, "status": "skipped", "reason": f"{path.name}: no numeric sample columns"}

    clean_matrix = rf.restrict_to_clean_genes(numeric, clean_symbols)
    if clean_matrix is None:
        return {
            "gse_id": gse_id, "status": "skipped",
            "reason": f"{path.name}: none of this platform's mapped genes are in the clean reference gene set",
        }

    out_path = cohort_dir / "expression_final.tsv.gz"
    clean_matrix.round(3).to_csv(out_path, sep="\t")

    return {
        "gse_id": gse_id, "status": "processed",
        "reason": f"source={source_name}; {len(numeric)} -> {len(clean_matrix)} clean genes kept",
        "source_file": path.name, "n_genes": len(clean_matrix), "n_samples": clean_matrix.shape[1],
        "out_file": str(out_path),
    }


def build_final_matrices(
    cohort_roots: list[Path], gencode_version: str = "50", references_dir: Path | None = None,
    series_dir: Path | None = None,
) -> pd.DataFrame:
    """Finalize every microarray cohort under each root in cohort_roots (a
    root's immediate GSE* subdirectories), writing expression_final.tsv.gz
    per cohort as a side effect. An RNA-seq cohort under the same root is
    silently skipped here (no expression.tsv.gz/channel_signal_expression.
    tsv.gz at its own cohort_dir root -- RNA-seq's own raw files live under
    its expression/ subdirectory instead) -- run
    rnaseq_finalize.build_final_matrices for those. Returns a one-row-per-
    cohort report DataFrame (status/reason plus finalize_cohort's other
    fields).
    """
    clean_genes_path = (
        (references_dir or config.REFERENCES_DIR) / f"gencode{gencode_version}"
        / f"clean_transcript_gene_symbol_v{gencode_version}.tsv.gz"
    )
    clean_symbols = rf.load_clean_symbols(clean_genes_path)

    report = []
    for root in cohort_roots:
        root = Path(root)
        for cohort_dir in sorted(p for p in root.glob("GSE*") if p.is_dir()):
            report.append(finalize_cohort(cohort_dir, clean_symbols, series_dir=series_dir))

    return pd.DataFrame(report)
