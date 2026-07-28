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
@click.option("--max-results", default=100, show_default=True, help="Max series from the Entrez search")
@click.option("--out", "out_name", default="report", show_default=True, help="Output basename under data/reports/")
def search(title, description, organism, sample_properties, max_results, out_name):
    """Search GEO by title/description/organism and optional sample properties."""
    if not title and not description and not organism:
        raise click.UsageError("Provide at least one of --title / --description / --organism")

    rows = search_mod.search(
        title=title,
        description=description,
        organism=organism,
        sample_properties=list(sample_properties) or None,
        max_results=max_results,
    )
    df = report.build(rows)
    tsv_path, xlsx_path = report.write(df, out_name)
    report.print_table(df)
    click.echo(f"\n{len(df)} series written to {tsv_path} and {xlsx_path}")


if __name__ == "__main__":
    main()
