"""geotool command-line interface."""
from __future__ import annotations

import sys

import click
import pandas as pd

from geotool import download as download_mod, report, search as search_mod


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
    "--llm-annotate",
    "llm_annotate_flag",
    is_flag=True,
    help="Use Claude to classify sample source/tissue/diagnosis/therapy for every candidate "
    "(fetches each candidate's full record; needs ANTHROPIC_API_KEY). Adds llm_* columns "
    "to samples.tsv and a diagnosis breakdown to the report.",
)
@click.option(
    "--llm-escalate",
    is_flag=True,
    help="With --llm-annotate: re-run fields the model marked unknown/ambiguous through "
    "the escalation model (GEOTOOL_LLM_ESCALATION_MODEL, default claude-sonnet-5).",
)
@click.option("--max-results", default=100, show_default=True, help="Max series from the Entrez search")
@click.option("--out", "out_name", default="report", show_default=True, help="Output basename under data/reports/")
def search(
    title, description, organism, sample_properties, llm_annotate_flag, llm_escalate, max_results, out_name
):
    """Search GEO by title/description/organism and optional sample properties."""
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
def download(gse_ids, from_report, rma_flag):
    """Download expression data + a cleaned annotation table for one or more cohorts, e.g.

    geotool download GSE10846 GSE339488

    RNA-seq series get their supplementary expression file(s) downloaded as-is.
    Microarray series get reshaped from each sample's probe values into a
    probes x samples matrix, then mapped to a genes x samples matrix via each
    platform's own annotation table. Two-channel Agilent samples that publish
    per-channel intensity columns also get channel1_expression.tsv.gz /
    channel2_expression.tsv.gz alongside the ratio-based expression.tsv.gz
    (unchanged). Add --rma to also RMA-renormalize Affymetrix series from raw
    CEL files. Every cohort also gets a cleaned, semantically-unified
    annotation.tsv (redundant columns dropped; treatment/response/RECIST/
    survival unified where possible). Needs ANTHROPIC_API_KEY. Writes into
    data/series/<GSE_ID>/.
    """
    ids = list(gse_ids)
    if from_report:
        ids.extend(pd.read_csv(from_report, sep="\t")["gse_id"].dropna().astype(str).tolist())
    ids = list(dict.fromkeys(ids))  # de-dupe, keep order

    if not ids:
        raise click.UsageError("Provide one or more GSE IDs, or --from-report <path>")

    for gse_id in ids:
        click.echo(f"{gse_id}:")
        try:
            result = download_mod.download_cohort(gse_id, rma=rma_flag)
        except Exception as exc:
            click.echo(f"  FAILED: {exc}")
            continue
        click.echo(f"  assay type(s): {', '.join(result['assay_types']) or 'unknown'}")
        if result.get("expression_path"):
            click.echo(f"  expression matrix: {result['expression_path']}")
        if result.get("channel_expression_paths"):
            for channel_num, path in sorted(result["channel_expression_paths"].items()):
                click.echo(f"  channel {channel_num} expression matrix: {path}")
        if result.get("expression_rma_path"):
            click.echo(f"  RMA expression matrix: {result['expression_rma_path']}")
        if result.get("expression_files"):
            click.echo(f"  expression files: {len(result['expression_files'])} downloaded")
        click.echo(f"  annotation: {result['annotation_path']}")


if __name__ == "__main__":
    main()
