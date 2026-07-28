"""Build the search-result report and export it to TSV/Excel/console."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tabulate import tabulate

from geotool import config

COLUMNS = [
    "gse_id",
    "title",
    "organism",
    "n_samples",
    "platforms",
    "submission_date",
    "sample_property_matches",
    "pubmed_ids",
    "url",
]


def build(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        record = dict(row)
        if isinstance(record.get("platforms"), list):
            record["platforms"] = ";".join(record["platforms"])
        if isinstance(record.get("pubmed_ids"), list):
            record["pubmed_ids"] = ";".join(str(p) for p in record["pubmed_ids"])
        record.setdefault("sample_property_matches", "")
        gse_id = record.get("gse_id", "")
        record["url"] = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}" if gse_id else ""
        records.append(record)
    df = pd.DataFrame(records)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[COLUMNS]


def write(df: pd.DataFrame, out_name: str) -> tuple[Path, Path]:
    config.ensure_dirs()
    tsv_path = config.REPORTS_DIR / f"{out_name}.tsv"
    xlsx_path = config.REPORTS_DIR / f"{out_name}.xlsx"
    df.to_csv(tsv_path, sep="\t", index=False)
    df.to_excel(xlsx_path, index=False)
    return tsv_path, xlsx_path


def print_table(df: pd.DataFrame, max_rows: int = 20) -> None:
    display_df = df.head(max_rows).copy()
    if "summary" in display_df.columns:
        display_df["summary"] = display_df["summary"].str.slice(0, 60)
    display_df["title"] = display_df["title"].str.slice(0, 50)
    print(tabulate(display_df, headers="keys", tablefmt="simple", showindex=False))
    if len(df) > max_rows:
        print(f"... ({len(df) - max_rows} more rows in the saved report)")
