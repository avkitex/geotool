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

### Download

```bash
geotool download GSE10846 GSE339488
```

RNA-seq series get their supplementary expression file(s) downloaded as-is.
Microarray series get reshaped from each sample's own probe values into a
probes x samples matrix, then mapped to a genes x samples matrix via each
platform's own annotation table (`probe_matrix.tsv.gz` / `expression.tsv.gz`
-- gzip-compressed, values rounded to 3 decimal places).
Every cohort also gets a cleaned, semantically-unified `annotation.tsv`.
Writes into `data/series/<GSE_ID>/`.

Two-channel Agilent samples (Cy3/Cy5 reference-design arrays) whose own data
table publishes per-channel intensity columns -- not all do; most only carry
the precomputed ratio -- also get each channel's own probe/gene matrix as an
*additional* output: `channel1_probe_matrix.tsv.gz` / `channel1_expression.tsv.gz`
and the `channel2_*` equivalents, alongside the unchanged, ratio-based
`expression.tsv.gz`. Neither channel is assumed to be "the real sample" or
"the reference" -- that's a per-study convention this tool has no reliable
way to infer, so both are simply named by channel number (channel 1 is
always the green/Cy3 scanner channel, channel 2 always red/Cy5).

Add `--rma` to also RMA-renormalize Affymetrix microarray series from their
raw CEL files (`probe_matrix_rma_<GPL>.tsv.gz` / `expression_rma.tsv.gz`,
written alongside the submitter-value files above, never replacing them).
CEL files are deleted once a platform's RMA run succeeds, to avoid keeping
both the raw data and its derived matrix on disk. This is opt-in and needs
R installed with `Rscript` on `PATH` -- Bioconductor itself and every R
package RMA needs (BiocManager, `affy`/`oligo`, and the chip-specific
CDF/pd.* package) are installed automatically on first use, into a
user-writable library, so no manual R package setup is required. The
tradeoff is that the very first `--rma` run for a given chip can take a
while (network install time on top of the RMA computation); every run after
that is fast since the packages persist.

Only platforms listed in `geotool/renormalize.py`'s `_CHIP_PACKAGES` table are
supported; an unlisted platform, or a failed install/run, just skips
RMA for that series (logged) rather than failing the download.

## Status

Phase 1 (search + report) and LLM-assisted annotation/query are implemented.
Phase 2 (download, plus opt-in CEL/RMA renormalization) is implemented — see
`geotool/download.py`, `geotool/probe_mapping.py`, `geotool/renormalize.py`.
Cross-cohort harmonization is planned but not yet implemented — see the stub
module `geotool/harmonize.py`.
