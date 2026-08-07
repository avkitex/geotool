"""Filter RNA-seq cohorts' gene-level expression matrices down to the clean
GENCODE reference gene set (data/references/gencode50), then renormalize
each sample to a 1,000,000 composition (TPM-style), for every bulk_rnaseq
cohort in data/pdac_cohorts and data/mtap_prmt5_cohorts that has a resolved
primary expression matrix (geotool.download's own expression_qc.json,
written for every RNA-seq cohort it successfully resolved a file for).

Row-identifier detection, ENST/ENSG->gene-symbol mapping, and duplicate-row
aggregation are all delegated to geotool.gene_symbol_mapping (built on this
same reference), not reimplemented here -- it already handles the ENST/
ENSG/symbol cascade robustly (including submitter oddities like Kallisto/
Salmon pipe-delimited composite headers). This script adds only what that
shared module doesn't do: restricting the *result* to the clean,
protein-coding/CCDS-filtered gene set regardless of which ID scheme the
input used (gene_symbol_mapping's ENSG/symbol paths intentionally map
against its *full* gene_to_symbol/known_symbols -- broader than this
reference's clean set -- since it's meant as a general-purpose converter,
not this pipeline's specific inclusion policy), the raw-count-vs-already-
normalized branch (compute_tpm vs a plain sum-to-1e6 rescale -- rescaling
already-length-normalized TPM/FPKM/RPKM to sum 1e6 *is* the standard
FPKM/RPKM->TPM conversion), and the per-cohort file resolution / skip
reporting.

Quantification unit comes from expression_qc.json's primary_expression_unit
(tpm/fpkm/rpkm need no length step; count/cpm do, via compute_tpm's
per-gene median included-transcript length). Cohorts whose unit is
"unknown", missing entirely (no expression_qc.json), a multi-file cohort
(unit present but null), or whose file's actual value scale doesn't match
what this script can safely process are skipped and reported, never
guessed at.

Whether the matrix on disk is already log2(x+1)-scale or still linear is
detected per file with the same heuristic the rest of this codebase uses
(probe_mapping.needs_log2_transform: any value over 50 means still linear)
rather than assumed from which collection it came from -- data/pdac_cohorts
and data/mtap_prmt5_cohorts turn out to hold already-log2-transformed
copies distinct from data/series' own (still-linear) resolved files for the
same GSE, so trusting a fixed assumption per directory would silently
mishandle one or the other. Output is re-log2(x+1)-transformed before
writing, to match this codebase's storage convention for processed
expression matrices.

Usage: .venv/Scripts/python.exe data/reports/filter_renormalize_rnaseq_cohorts.py
"""
import json
import re
from pathlib import Path

import pandas as pd

from geotool import gene_symbol_mapping as gsm
from geotool import probe_mapping

COHORT_ROOTS = [Path("data/pdac_cohorts"), Path("data/mtap_prmt5_cohorts")]
REPORT_PATH = Path("data/reports/rnaseq_clean_tpm_report.tsv")
CLEAN_GENES_PATH = Path("data/references/gencode50/clean_transcript_gene_symbol_v50.tsv.gz")

# fpkm/rpkm/tpm are already length-normalized -- rescaling them to sum to
# 1e6 per sample *is* the standard FPKM/RPKM->TPM conversion. count/cpm are
# not length-normalized at all (cpm only did the "per million" rescaling
# step), so both need compute_tpm's length-division for the result to be a
# real TPM rather than just a filtered-and-rescaled CPM.
_LENGTH_NORM_UNITS = {"count", "cpm"}


def load_clean_symbols():
    clean = pd.read_csv(CLEAN_GENES_PATH, sep="\t")
    return set(clean["gene_symbol"].unique())


# gene_symbol_mapping.canonical_id unwraps Kallisto/Salmon-style *pipe*-
# delimited composite headers ("ENST...|ENSG...|..."), but not other
# submitter-specific compound formats -- live example, GSE79668:
# "RP1-67K17.4_ENSG00000237851.1" (a gene-symbol prefix, underscore, then
# the ENSG id+version). That row's identifier axis reads as "unknown" to
# gene_symbol_mapping.detect_identifier_type as-is. This extracts an
# embedded ENST/ENSG substring from anywhere in the index (not just a
# leading pipe-delimited field) as a local preprocessing step, rather than
# broadening the shared module's own composite-ID handling for one cohort's
# one-off format.
_EMBEDDED_ENS_ID_RE = re.compile(r"(ENS[TG]\d+(?:\.\d+)?)")


def unwrap_embedded_ensembl_ids(matrix):
    """If >50% of the index contains an embedded ENST/ENSG id, replace the
    index with just that extracted substring; otherwise return unchanged.
    """
    extracted = matrix.index.to_series().astype(str).str.extract(_EMBEDDED_ENS_ID_RE, expand=False)
    if extracted.notna().mean() > 0.5:
        m = matrix.copy()
        m.index = extracted.fillna(pd.Series(matrix.index.astype(str), index=extracted.index))
        return m
    return matrix


def to_linear_scale(matrix):
    """Return (linear_matrix, was_log2) -- inverse-transforms log2(x+1) back
    to linear scale only if the data actually looks log2-scale (same
    >50-means-still-linear heuristic as the rest of this codebase), since
    data/pdac_cohorts and data/mtap_prmt5_cohorts hold a mix of
    already-log2 and still-linear files for different cohorts.
    """
    if probe_mapping.needs_log2_transform(matrix):
        return matrix.clip(lower=0), False
    return (2 ** matrix - 1).clip(lower=0), True


def renormalize_to_1e6(gene_matrix):
    col_sums = gene_matrix.sum(axis=0)
    zero_cols = col_sums[col_sums <= 0].index.tolist()
    if zero_cols:
        return None, f"{len(zero_cols)} sample(s) sum to zero after filtering, cannot renormalize: {zero_cols[:5]}"
    return gene_matrix.div(col_sums, axis=1) * 1e6, None


def restrict_to_clean_genes(gene_matrix, clean_symbols):
    kept = gene_matrix[gene_matrix.index.isin(clean_symbols)]
    if kept.empty:
        return None
    # Defensive: convert_to_gene_symbols/compute_tpm already group by
    # gene_symbol and sum, but restricting rows here can't itself introduce
    # duplicates -- a plain reindex would be enough. groupby-sum kept only
    # for robustness against any future upstream change in that guarantee.
    return kept.groupby(level=0).sum()


def find_local_expression_file(cohort_dir, qc):
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


def process_cohort(cohort_dir, ref, clean_symbols):
    gse_id = cohort_dir.name
    qc_path = cohort_dir / "expression_qc.json"
    if not qc_path.exists():
        return {"gse_id": gse_id, "status": "skipped", "reason": "no expression_qc.json (not RNA-seq, or no resolvable expression matrix)"}

    qc = json.loads(qc_path.read_text())
    unit = qc.get("primary_expression_unit")
    if unit is None:
        return {"gse_id": gse_id, "status": "skipped", "reason": "no single primary expression matrix for this cohort (multi-file case)"}
    if unit == "unknown":
        return {"gse_id": gse_id, "status": "skipped", "reason": "quantification unit unknown -- can't safely length-normalize or trust as already-normalized"}

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

    linear, was_log2 = to_linear_scale(numeric)
    if float(linear.min().min()) < -1e-3:
        return {"gse_id": gse_id, "status": "skipped", "reason": f"{path.name}: negative values remain after scale detection -- not a clean expression matrix"}

    linear = unwrap_embedded_ensembl_ids(linear)
    converted, convert_note = gsm.convert_to_gene_symbols(linear, ref)
    if converted is None:
        return {"gse_id": gse_id, "status": "skipped", "reason": f"{path.name}: {convert_note}"}

    if unit in _LENGTH_NORM_UNITS:
        tpm_df, tpm_note = gsm.compute_tpm(converted, ref)
        if tpm_df is None:
            return {"gse_id": gse_id, "status": "skipped", "reason": f"{path.name}: {tpm_note}"}
        gene_matrix = tpm_df.set_index("gene_symbol")
        detail = f"{convert_note}; {tpm_note}"
    else:
        gene_matrix = converted.set_index("gene_symbol")
        detail = convert_note

    clean_matrix = restrict_to_clean_genes(gene_matrix, clean_symbols)
    if clean_matrix is None:
        return {"gse_id": gse_id, "status": "skipped", "reason": f"{path.name}: none of the mapped genes are in the clean reference gene set"}

    tpm, err = renormalize_to_1e6(clean_matrix)
    if tpm is None:
        return {"gse_id": gse_id, "status": "skipped", "reason": f"{path.name}: {err}"}

    out = probe_mapping.maybe_log2_transform(tpm)
    out_path = cohort_dir / "expression_clean.tsv.gz"
    out.round(3).to_csv(out_path, sep="\t")

    return {
        "gse_id": gse_id, "status": "processed", "reason": f"{detail}; {len(tpm)} clean genes kept",
        "unit": unit, "source_file": path.name, "was_log2_on_disk": was_log2,
        "n_genes": len(tpm), "n_samples": tpm.shape[1], "out_file": str(out_path),
    }


def main():
    ref = gsm.load_gencode_reference("50")
    clean_symbols = load_clean_symbols()

    report = []
    for root in COHORT_ROOTS:
        for cohort_dir in sorted(p for p in root.glob("GSE*") if p.is_dir()):
            report.append(process_cohort(cohort_dir, ref, clean_symbols))

    report_df = pd.DataFrame(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_PATH, sep="\t", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print(report_df.to_string(index=False))
    n_processed = (report_df["status"] == "processed").sum()
    n_skipped = (report_df["status"] == "skipped").sum()
    print(f"\nprocessed: {n_processed}, skipped: {n_skipped}")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
