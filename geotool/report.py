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
    "array_content",
    "submission_date",
    "sample_property_matches",
    "pubmed_ids",
    "url",
]

# Only added to the report if a row actually has them (e.g. --llm-annotate).
OPTIONAL_COLUMNS = ["llm_diagnosis_breakdown", "llm_assay_type"]

QUERY_COLUMNS = [
    "gse_id",
    "title",
    "matches_diagnosis",
    "diagnosis_detail",
    "species",
    "sample_type",
    "tissue_class",
    "assay_type",
    "selection_method",
    "n_samples",
    "meets_min_samples",
    "organism",
    "platforms",
    "array_content",
    "submission_date",
    "pubmed_ids",
    "notes",
    "url",
]


def _normalize_record(row: dict) -> dict:
    record = dict(row)
    if isinstance(record.get("platforms"), list):
        record["platforms"] = ";".join(record["platforms"])
    if isinstance(record.get("pubmed_ids"), list):
        record["pubmed_ids"] = ";".join(str(p) for p in record["pubmed_ids"])
    if isinstance(record.get("diagnosis_breakdown"), dict):
        record["diagnosis_breakdown"] = "; ".join(f"{k}={v}" for k, v in record["diagnosis_breakdown"].items())
    gse_id = record.get("gse_id", "")
    record["url"] = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}" if gse_id else ""
    return record


def build(rows: list[dict]) -> pd.DataFrame:
    records = [_normalize_record(row) for row in rows]
    for record in records:
        record.setdefault("sample_property_matches", "")
    df = pd.DataFrame(records)
    columns = list(COLUMNS)
    for col in OPTIONAL_COLUMNS:
        if any(col in record for record in records):
            columns.append(col)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df[columns]


def build_query_report(rows: list[dict]) -> pd.DataFrame:
    """Report shape for `geotool query`: receipt columns instead of sample_property_matches."""
    records = [_normalize_record(row) for row in rows]
    df = pd.DataFrame(records)
    for col in QUERY_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[QUERY_COLUMNS]


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
