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

For the LLM-assisted commands below, set an Anthropic API key (e.g. in a
local `.env` file, which is gitignored):

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
set -a && source .env && set +a
```

## Usage

### Keyword search

```bash
geotool search --title "breast cancer" --organism "Homo sapiens" \
  --sample-property "tissue:liver" --sample-property "treatment:tamoxifen" \
  --max-results 100 --out report
```

Writes `data/reports/report.tsv` and `data/reports/report.xlsx`. If any
`--sample-property` filters are given, per-series annotation is also saved to
`data/series/<GSE_ID>/series.tsv` and `data/series/<GSE_ID>/samples.tsv`.

Add `--llm-annotate` to have Claude classify each candidate's samples
(species, biopsy/cell line, tissue, diagnosis, prior therapy) from their full
GEO record; `--llm-escalate` re-runs low-confidence fields on a stronger
model. Needs `ANTHROPIC_API_KEY`.

### Natural-language cohort query

```bash
geotool query "human biopsy pancreatic cancer cohorts with sample size more than 20"
```

Parses the request into a diagnosis (plus synonyms) and filter categories
(species, biopsy/cell line, tissue, assay type, material selection) with one
Claude call, recalls candidate series from GEO by the diagnosis and its
synonyms, then classifies each candidate's title/summary against the filters
with one lightweight Claude call each. Every candidate is written to the
report with its own columns — nothing is silently dropped, so you can
filter/sort the table yourself. Logs the parsed filters and each candidate's
classification as it runs (`--quiet` to suppress). Needs `ANTHROPIC_API_KEY`.

## Status

Phase 1 (search + report) and LLM-assisted annotation/query are implemented.
Download/renormalization and cross-cohort harmonization are planned but not
yet implemented — see stub modules `geotool/download.py`,
`geotool/probe_mapping.py`, `geotool/harmonize.py`.
