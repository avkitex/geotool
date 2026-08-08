"""geotool command-line interface."""
from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd

from geotool import config, download as download_mod, harmonize as harmonize_mod, report, search as search_mod
from geotool import rnaseq_finalize as rnaseq_finalize_mod


def _ensure_utf8_streams() -> None:
    """Windows terminals often leave stdout/stderr on a legacy codepage (e.g.
    cp1252) that can't encode characters GEO metadata frequently contains
    (accented names, Greek letters as in "IFN-γ", em dashes, ...).
    Reconfiguring here -- once, at the entrypoint -- covers every later
    print()/click.echo() call in the app, rather than every call site having
    to guard against it: without this, a print() near the very end of an
    otherwise fully successful `download` run raises UnicodeEncodeError and
    the CLI reports the whole cohort as FAILED.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_ensure_utf8_streams()


@click.group()
def main():
    """Search, download, and harmonize GEO expression cohorts."""


@main.command()
@click.option("--title", default=None, help="Match against series title")
@click.option("--description", default=None, help="Match against series description/summary")
@click.option("--organism", default=None, help='e.g. "Homo sapiens"')
@click.option(
    "--sample-property",
    "sample_properties",
    multiple=True,
    help="key:value filter on sample characteristics, e.g. tissue:liver. Repeatable; "
    "a series is kept only if every filter matches at least one sample. "
    "Slower: fetches each candidate series' full record.",
)
@click.option(
    "--llm-annotate/--no-llm-annotate",
    "llm_annotate_flag",
    default=False,
    show_default=True,
    help="Use Claude to classify sample source/tissue/diagnosis/therapy for every candidate "
    "(fetches each candidate's full record; needs ANTHROPIC_API_KEY -- one call per candidate, "
    "up to --max-results of them). Adds llm_* columns to samples.tsv and a diagnosis breakdown "
    "to the report. Off by default -- no ANTHROPIC_API_KEY needed, no LLM calls made, results "
    "come back with those columns empty rather than classified.",
)
@click.option(
    "--llm-escalate",
    is_flag=True,
    help="With --llm-annotate: re-run fields the model marked unknown/ambiguous through the "
    "escalation model (GEOTOOL_LLM_ESCALATION_MODEL, default claude-sonnet-5).",
)
@click.option("--max-results", default=100, show_default=True, help="Max series from the Entrez search")
@click.option("--out", "out_name", default="report", show_default=True, help="Output basename under data/reports/")
def search(
    title, description, organism, sample_properties, llm_annotate_flag, llm_escalate, max_results, out_name
):
    """Search GEO by title/description/organism and optional sample properties.

    No LLM calls, no ANTHROPIC_API_KEY needed, by default. Pass --llm-annotate
    to additionally classify every candidate with Claude (tissue/diagnosis/
    sample source/therapy) -- needs ANTHROPIC_API_KEY, one call per candidate.
    """
    if not title and not description and not organism:
        raise click.UsageError("Provide at least one of --title / --description / --organism")

    rows = search_mod.search(
        title=title,
        description=description,
        organism=organism,
        sample_properties=list(sample_properties) or None,
        max_results=max_results,
        llm_annotate_flag=llm_annotate_flag,
        llm_escalate=llm_escalate,
    )
    df = report.build(rows)
    tsv_path, xlsx_path = report.write(df, out_name)
    report.print_table(df)
    click.echo(f"\n{len(df)} series written to {tsv_path} and {xlsx_path}")


@main.command()
@click.argument("text")
@click.option(
    "--llm-escalate",
    is_flag=True,
    help="Re-run a candidate through the escalation model when the cheap model's "
    "classification came back mostly unknown from the summary alone.",
)
@click.option("--max-results", default=100, show_default=True, help="Max series from the Entrez recall stage")
@click.option("--out", "out_name", default="query", show_default=True, help="Output basename under data/reports/")
@click.option(
    "--quiet",
    is_flag=True,
    help="Suppress the parsed-filters and per-candidate classification log lines; print only the final table.",
)
def query(text, llm_escalate, max_results, out_name, quiet):
    """Natural-language cohort search, e.g.

    geotool query "human biopsy pancreatic cancer cohorts with sample size more than 20"

    Parses TEXT into a diagnosis (plus synonyms) and filter categories (species,
    biopsy/cell line, tissue, assay type, material selection) with one Claude
    call, recalls candidate series from GEO by the diagnosis and its synonyms,
    then classifies each candidate's title/summary against the filters with one
    lightweight Claude call each. Every candidate is reported with its own
    columns -- nothing is silently dropped, so you can filter/sort the table
    yourself. Logs the parsed filters and each candidate's classification as it
    runs (use --quiet to suppress).
    """
    rows = search_mod.run_nl_query(
        text, max_results=max_results, escalate_ambiguous=llm_escalate, verbose=not quiet
    )
    df = report.build_query_report(rows)
    tsv_path, xlsx_path = report.write(df, out_name)
    report.print_table(df)
    click.echo(f"\n{len(df)} series written to {tsv_path} and {xlsx_path}")


@main.command()
@click.argument("gse_ids", nargs=-1)
@click.option(
    "--from-report",
    "from_report",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Read gse_id values from a saved search/query report (.tsv) instead of passing them as arguments.",
)
@click.option(
    "--rma",
    "rma_flag",
    is_flag=True,
    help="For Affymetrix microarray series, also download raw CEL files and RMA-renormalize "
    "them (in addition to the submitter-value expression matrix, which is always produced). "
    "Requires R installed with Rscript on PATH; Bioconductor and every R package RMA needs "
    "(affy/oligo, plus a chip-specific CDF or pd.* package) are installed automatically on "
    "first use. Skips gracefully per-platform when Rscript is missing, the platform is "
    "unknown, or an install/run fails.",
)
@click.option(
    "--force",
    "force_flag",
    is_flag=True,
    help="Redo a cohort even if it's already been downloaded (by default, an already-downloaded "
    "cohort -- annotation.tsv already on disk -- is reused as-is rather than re-fetched and "
    "re-processed, which would otherwise repeat any --clinical-annotate LLM call and RMA run "
    "for no reason). Adding --rma to a cohort previously downloaded without it still computes "
    "just the missing RMA output, without needing --force.",
)
@click.option(
    "--clinical-annotate/--no-clinical-annotate",
    "clinical_annotate_flag",
    default=False,
    show_default=True,
    help="Use Claude to identify redundant/treatment/response/survival columns in each cohort's "
    "own annotation table and unify them (geotool.clinical_annotate.plan_column_mapping) -- "
    "needs ANTHROPIC_API_KEY, one call per cohort. Off by default: annotation.tsv still gets "
    "the LLM-independent cleanup (constant-column drop, 'Label: ' prefix strip), just not "
    "treatment/response/RECIST/survival unification.",
)
def download(gse_ids, from_report, rma_flag, force_flag, clinical_annotate_flag):
    """Download expression data + a cleaned annotation table for one or more cohorts, e.g.

    geotool download GSE10846 GSE339488

    No LLM calls, no ANTHROPIC_API_KEY needed, by default. Pass
    --clinical-annotate to additionally use Claude for treatment/response/
    RECIST/survival column unification (see --clinical-annotate's own help).

    RNA-seq series get their supplementary expression file(s) downloaded as-is
    -- when there's more than one, the one that looks like the actual
    quantification matrix (by filename: TPM > FPKM > RPKM > CPM > raw counts) is
    additionally checked for two easy-to-miss problems and reported as
    "expression QC" notes: values that don't look log2-transformed (a linear-
    scale note, not necessarily wrong), and negative values (a real red flag
    -- often means log2 was applied without a +1 pseudocount, or the file
    isn't actually a raw expression matrix). The same QC runs on microarray
    series' expression.tsv.gz. Microarray series get reshaped from each
    sample's probe values into a
    probes x samples matrix, then mapped to a genes x samples matrix via each
    platform's own annotation table. Two-channel Agilent samples that publish
    per-channel intensity columns also get channel1_expression.tsv.gz /
    channel2_expression.tsv.gz alongside the ratio-based expression.tsv.gz
    (unchanged). When it's confident which channel is the actual biological
    sample vs. a fixed reference (metadata text and/or lower cross-sample
    variance -- see probe_mapping.detect_reference_channel), also writes
    channel_signal_expression.tsv.gz / channel_reference_expression.tsv.gz;
    otherwise only the neutral channel1/channel2 files are written. Add
    --rma to also RMA-renormalize Affymetrix series from raw CEL files.
    Every cohort also gets a cleaned annotation.tsv (constant columns
    dropped, "Label: " prefixes stripped -- plus, with --clinical-annotate,
    redundant columns dropped and treatment/response/RECIST/survival unified
    via Claude), plus an expression_status column (clinical_annotate.
    classify_expression_status) flagging, in every row, the same
    cohort-level QC verdict as a word-enum: "ok", "no_expression_matrix"
    (e.g. a series that only ever published differential-expression/
    splicing-analysis output, never a raw or normalized matrix -- see
    GSE108651), "unparseable", or "not_log2_transformed"/"negative_values"
    (joined with ";" if both apply). A cohort that's already been downloaded
    is reused rather than redone -- pass --force to redo it anyway. Writes
    into data/series/<GSE_ID>/.

    A SuperSeries is automatically expanded into its subseries (recursively),
    each downloaded independently into its own data/series/<subseries_id>/.
    Non-human cohorts and unsupported microarray content (miRNA/lncRNA-only or
    CNA arrays, or platforms below ~8000 probes/genes) are reported as a clean
    FAILED rather than attempted -- a platform that combines mRNA with
    miRNA/lncRNA content on one chip is unaffected.
    """
    ids = list(gse_ids)
    if from_report:
        ids.extend(pd.read_csv(from_report, sep="\t")["gse_id"].dropna().astype(str).tolist())
    ids = list(dict.fromkeys(ids))  # de-dupe, keep order

    if not ids:
        raise click.UsageError("Provide one or more GSE IDs, or --from-report <path>")

    for gse_id in ids:
        try:
            target_ids = download_mod.resolve_download_targets(gse_id, force=force_flag)
        except Exception as exc:
            click.echo(f"{gse_id}:")
            click.echo(f"  FAILED: {exc}")
            continue
        if target_ids != [gse_id]:
            click.echo(f"{gse_id}: SuperSeries with {len(target_ids)} sub-series -- downloading each: {', '.join(target_ids)}")

        for target_id in target_ids:
            click.echo(f"{target_id}:")
            try:
                result = download_mod.download_cohort(
                    target_id, rma=rma_flag, force=force_flag, clinical_annotate_flag=clinical_annotate_flag,
                )
            except Exception as exc:
                click.echo(f"  FAILED: {exc}")
                continue
            click.echo(f"  assay type(s): {', '.join(result['assay_types']) or 'unknown'}")
            if result.get("expression_path"):
                click.echo(f"  expression matrix: {result['expression_path']}")
            if result.get("channel_expression_paths"):
                for channel_num, path in sorted(result["channel_expression_paths"].items()):
                    click.echo(f"  channel {channel_num} expression matrix: {path}")
            if result.get("channel_roles"):
                roles = result["channel_roles"]
                click.echo(
                    f"  channel roles ({roles['method']}): "
                    f"channel {roles['signal_channel']} = signal, channel {roles['reference_channel']} = reference"
                )
            if result.get("channel_signal_expression_path"):
                click.echo(f"  channel signal expression matrix: {result['channel_signal_expression_path']}")
            if result.get("channel_reference_expression_path"):
                click.echo(f"  channel reference expression matrix: {result['channel_reference_expression_path']}")
            if result.get("expression_rma_path"):
                click.echo(f"  RMA expression matrix: {result['expression_rma_path']}")
            if result.get("expression_files"):
                click.echo(f"  expression files: {len(result['expression_files'])} downloaded")
            if result.get("primary_expression_file"):
                click.echo(f"  primary expression file ({result.get('primary_expression_unit')}): {result['primary_expression_file']}")
            if result.get("expression_qc_notes"):
                for note in result["expression_qc_notes"]:
                    click.echo(f"  expression QC: {note}")
            if result.get("expression_status"):
                click.echo(f"  expression status: {result['expression_status']}")
            click.echo(f"  annotation: {result['annotation_path']}")


@main.command()
@click.argument("gse_ids", nargs=-1)
@click.option(
    "--from-report",
    "from_report",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Read gse_id values from a saved search/query report (.tsv) instead of passing them as arguments.",
)
@click.option(
    "--llm-annotate",
    "llm_annotate_flag",
    is_flag=True,
    help="Backfill tissue/diagnosis classification for cohorts with no cached llm_annotations.json "
    "(from a prior `search --llm-annotate`) -- needs ANTHROPIC_API_KEY, and costs a real LLM call "
    "per such cohort. Off by default: cohorts without a cache just get 'unknown' for those fields.",
)
@click.option("--out", "out_name", default="harmonized", show_default=True, help="Output basename under data/harmonized/")
@click.option(
    "--master",
    "master_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Start from an existing harmonized annotation.tsv: its cohorts are skipped (not reprocessed), its "
    "already-canonical columns take priority when matching new cohorts' columns onto them, and its rows are "
    "kept in the output unchanged.",
)
@click.option(
    "--match-columns/--no-match-columns",
    "match_columns",
    default=True,
    show_default=True,
    help="Cross-cohort column-concept matching LLM call (e.g. COO/PAM50/IGHV status unification). "
    "--no-match-columns skips it and just applies the existing per-cohort unification and alias renames.",
)
@click.option(
    "--collection-root",
    "collection_root",
    type=click.Path(file_okay=False),
    default=None,
    help="A project-specific processed-matrix collection (e.g. data/mtap_prmt5_cohorts) to check for "
    "each cohort's final analysis-ready expression_final.tsv.gz when computing cohort_annotations.tsv's "
    "readiness column. Without this, readiness falls back to each cohort's own expression_status alone.",
)
def harmonize(gse_ids, from_report, llm_annotate_flag, out_name, master_path, match_columns, collection_root):
    """Unify already-downloaded cohorts' annotation into one harmonized set of tables, e.g.

    geotool harmonize GSE10846 GSE98588

    Two outputs, together "the harmonization process":

    \b
    - annotation.tsv: one row per *sample*, reusing each cohort's own
      annotation.tsv (from `geotool download`) plus, if present, its cached
      llm_annotations.json (from `geotool search --llm-annotate`) -- at zero
      additional cost. Add --llm-annotate to backfill tissue/diagnosis
      classification for cohorts that don't have that cache yet. By default,
      a further LLM pass finds raw characteristic columns from different
      cohorts that describe the same concept (e.g. a DLBCL cell-of-origin
      call spelled three different ways across cohorts) and merges them
      under one canonical name with unified values; pass --no-match-columns
      to skip this. --master lets you grow an existing harmonized table
      incrementally rather than starting over each time.
    - cohort_annotations.tsv: one row per *cohort* (geotool.cohort_report),
      including every subseries a requested SuperSeries id expanded to.

    Cohorts that haven't been downloaded yet (no annotation.tsv) are skipped
    with a warning rather than failing the run.

    Writes data/harmonized/<name>/annotation.tsv and
    data/harmonized/<name>/cohort_annotations.tsv.
    """
    ids = list(gse_ids)
    if from_report:
        ids.extend(pd.read_csv(from_report, sep="\t")["gse_id"].dropna().astype(str).tolist())
    ids = list(dict.fromkeys(ids))  # de-dupe, keep order

    if not ids:
        raise click.UsageError("Provide one or more GSE IDs, or --from-report <path>")

    out_dir = config.DATA_DIR / "harmonized" / out_name
    master, cohort_df = harmonize_mod.harmonize_and_report(
        ids, out_dir, llm_annotate_flag=llm_annotate_flag, master_path=master_path,
        match_columns=match_columns, collection_root=collection_root,
    )

    if master.empty:
        click.echo("No cohorts could be harmonized -- none of the given GSE IDs have been downloaded yet.")
    else:
        n_cohorts = master["gse_id"].nunique() if "gse_id" in master.columns else len(ids)
        click.echo(f"{len(master)} samples across {n_cohorts} cohort(s) written to {out_dir / 'annotation.tsv'}")

    n_ready = int((cohort_df["readiness"] == "ready").sum())
    click.echo(f"{len(cohort_df)} cohort(s) ({n_ready} ready) written to {out_dir / 'cohort_annotations.tsv'}")


@main.command("finalize-rnaseq")
@click.argument("cohort_roots", nargs=-1, required=True, type=click.Path(file_okay=False))
@click.option("--gencode-version", default="50", show_default=True, help="GENCODE reference release under data/references/gencode<version>.")
@click.option("--out", "out_name", default="rnaseq_finalize", show_default=True, help="Output basename under data/reports/")
def finalize_rnaseq(cohort_roots, gencode_version, out_name):
    """Finalize every RNA-seq cohort's expression matrix under one or more
    collection roots, e.g.

    geotool finalize-rnaseq data/pdac_cohorts data/mtap_prmt5_cohorts

    For each root's immediate GSE* subdirectories with a resolved primary
    expression matrix (geotool.download's own expression_qc.json), converts
    row identifiers to HUGO gene symbols (geotool.gene_symbol_mapping),
    restricts to the clean GENCODE reference gene set, and renormalizes each
    sample to a 1,000,000 composition (TPM-style) -- writing
    <root>/<GSE>/expression_final.tsv.gz, the actual analysis-ready matrix.

    Writes data/reports/<name>.tsv (one row per cohort: processed/skipped/failed).
    """
    report_df = rnaseq_finalize_mod.build_final_matrices(
        [Path(r) for r in cohort_roots], gencode_version=gencode_version,
    )
    config.ensure_dirs()
    out_path = config.REPORTS_DIR / f"{out_name}.tsv"
    report_df.to_csv(out_path, sep="\t", index=False)

    n_processed = int((report_df["status"] == "processed").sum())
    n_skipped = int((report_df["status"] == "skipped").sum())
    n_failed = int((report_df["status"] == "failed").sum())
    click.echo(f"{len(report_df)} cohort(s) written to {out_path}")
    click.echo(f"processed: {n_processed}, skipped: {n_skipped}, failed: {n_failed}")


if __name__ == "__main__":
    main()
