# geotool

Search NCBI GEO for series by title, description, and sample properties, and
produce a TSV/Excel report to help pick cohorts.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -e ".[dev]"
```

Set your NCBI contact email (required by NCBI's usage policy):

```bash
export GEOTOOL_NCBI_EMAIL="you@example.com"
```

## Usage

```bash
geotool search --title "breast cancer" --organism "Homo sapiens" \
  --sample-property "tissue:liver" --sample-property "treatment:tamoxifen" \
  --max-results 100 --out report
```

Writes `data/reports/report.tsv` and `data/reports/report.xlsx`. If any
`--sample-property` filters are given, per-series annotation is also saved to
`data/series/<GSE_ID>/series.tsv` and `data/series/<GSE_ID>/samples.tsv`.

## Status

Phase 1 (search + report) is implemented. Download/renormalization and
cross-cohort harmonization are planned but not yet implemented — see stub
modules `geotool/download.py`, `geotool/probe_mapping.py`, `geotool/harmonize.py`.
