"""geotool command-line interface."""
from __future__ import annotations

import click

from geotool import report, search as search_mod


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


if __name__ == "__main__":
    main()
