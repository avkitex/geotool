# Project notes

GEO cohort acquisition/harmonization pipeline for PRMT5/MTAP and PDAC RNA-seq
cohorts, built on the `geotool` package. This file documents the reference
data, gene-ID conversion, and normalization steps well enough to reproduce
the pipeline from a clean checkout.

## Pipeline overview

1. **`geotool download <GSE_ID>...`** (`geotool/download.py`) — fetches a
   cohort's raw supplementary files and metadata from GEO/NCBI Entrez,
   writes `data/series/<GSE_ID>/` (`series.tsv`, `samples.tsv`,
   `annotation.tsv`, `expression/`, `expression_qc.json`). RNA-seq matrices
   are taken as published (no gene-ID conversion at this stage); microarray
   probes are mapped to genes via each platform's own annotation. A
   SuperSeries is auto-expanded into its subseries recursively.
2. **`geotool finalize-rnaseq <collection_root>...`**
   (`geotool/rnaseq_finalize.py`) — per cohort: resolves the quantification
   unit (guessing when unlabeled, see below), converts row identifiers to
   HUGO gene symbols (`geotool/gene_symbol_mapping.py`), restricts to the
   clean GENCODE gene set, TPM-renormalizes, and writes
   `<collection_root>/<GSE_ID>/expression_final.tsv.gz` — the actual
   analysis-ready matrix. Also matches expression-matrix sample columns
   back to `gsm_id` (`geotool/sample_id_matching.py`) and writes that onto
   the cohort's own `data/series/<GSE_ID>/annotation.tsv`.
3. **`geotool harmonize <GSE_ID>...`** (`geotool/harmonize.py` +
   `geotool/cohort_report.py`) — merges every requested cohort's
   `annotation.tsv` into one sample-level table and one cohort-level
   readiness table under `data/harmonized/<name>/`.

For this project specifically, `data/reports/rebuild_prmt5_collection.py`
and `data/reports/prmt5_common.py` (local, untracked — deliberately not in
this repo, see git history) chain these three steps for the PRMT5/MTAP
cohort list in `data/reports/mtap_prmt5_selected.tsv`.

## Reference data: GENCODE

All gene-ID conversion is built on GENCODE release 50 (GRCh38.p14), built by
`data/references/build_gencode_reference.py`. The raw GTFs (~125–160MB) are
**not vendored into git** — `fetch_sources()` downloads and sha256-verifies
them from `data/references/sources_manifest.json` on first run:

- `gencode.v50.annotation.gtf.gz` — GENCODE 50 GRCh38, from
  `https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/`
- `gencode.v50lift37.annotation.gtf.gz` — the same gene models backmapped
  to GRCh37/hg19 (used for older-array-platform cohorts)
- IntOGen cancer-driver-gene compendium (`intogen.org`, CC0) — cross-
  referenced against genes the clean-gene filter drops entirely, so a
  known cancer driver being excluded is visible rather than silent

To rebuild from scratch: `.venv/Scripts/python.exe
data/references/build_gencode_reference.py` (needs network access; nothing
else to install first).

Output tables per release (`data/references/gencode50/`,
`gencode50_hg19/`, both tracked in git):
- `id2gene_gencode_v50.tsv.gz` / `ensg2hugo_gencode_v50.tsv.gz` — raw,
  unfiltered ID→symbol maps (every annotated transcript/gene)
- `clean_transcript_gene_symbol_v50.tsv.gz` — the **filtered** transcript→
  symbol map `geotool.gene_symbol_mapping`/`rnaseq_finalize` actually use
  for final matrices (see "Clean gene set" below)
- `transcript_annotation_v50.tsv.gz` — every transcript with an
  `included`/`exclusion_reason` column (why each one was kept or dropped)
- `fully_dropped_genes_v50.tsv.gz` / `dropped_genes_cancer_relevance_v50.tsv`
  — genes with zero included transcripts, and which of those are IntOGen
  cancer drivers

A much older `gencode32` reference also exists (`data/references/gencode32/`)
for legacy compatibility — no clean-set filtering, gene-level only, kept
only as an id2gene/ensg2hugo fallback.

## Clean gene set (what "clean" means)

A transcript is **included** in the clean set only if, all of:
- its gene's `gene_type` is `protein_coding`
- the transcript itself is `transcript_type == protein_coding`
- spliced transcript length ≥ 300bp
- CCDS-backed (in the Consensus CDS set — independent evidence of a
  well-characterized coding transcript)
- not mitochondrially-encoded and not a replication-dependent histone gene
  (both excluded because their apparent expression is highly library-prep-
  dependent, not comparable biology — see `mt_gene_ids`/`rd_histone_gene_ids`
  in `build_gencode_reference.py`)

A gene is in the clean set if it has ≥1 included transcript. This is a
deliberate, opinionated policy for *this* pipeline's TPM output — it is
**not** applied inside `gene_symbol_mapping`'s own ID→symbol conversion
(that maps against the full, unfiltered gene/transcript universe, since
it's meant as a general-purpose converter); the clean-set restriction is
applied on top, in `rnaseq_finalize.restrict_to_clean_genes`.

## Gene ID → HUGO symbol conversion (`geotool/gene_symbol_mapping.py`)

Given a matrix's row identifiers, `detect_identifier_type` classifies a
random sample as `transcript` (ENST), `gene` (ENSG), `symbol` (already a
HUGO symbol), or `unknown` — checked in that priority order. Handles real
submitter oddities: version suffixes (`ENSG...18` → stripped), Kallisto/
Salmon pipe-delimited composite FASTA headers, and RSEM's doubled-symbol
IDs (`ACCSL_ACCSL` → `ACCSL`). `convert_to_gene_symbols` then maps via the
matching reference table and **sums** duplicate rows that collapse onto the
same symbol (multiple transcripts of one gene, or two IDs that happen to
share a symbol) — refuses outright on any negative values (would be
mathematically wrong to sum across log-fold-change/non-additive data).

## Quantification unit (raw counts vs. TPM/FPKM/RPKM/CPM)

`geotool.download` first tries to infer the unit from the filename
(`tpm`/`fpkm`/`rpkm`/`cpm`/`count` keyword); when none is found, content
verification confirms it's a real gene matrix but can't determine the unit
— recorded as `"unknown"` in `expression_qc.json`, not guessed at that
stage. `rnaseq_finalize.guess_unit` then applies a deliberate default: on
the *linear-scale* matrix (inverting any on-disk log2 first — using the
preserved `<name>.original.tsv.gz` raw file when one exists, since
reconstructing linear values from a lossily-rounded log2 file is not
reliable for this check) —
- **all values are (near-)whole numbers → raw counts** (the one unit
  that's always integer)
- otherwise, **transcript-level identifiers → CPM**
- otherwise (gene-level, or already symbols) **→ FPKM**

Every guess is stamped into that cohort's own report row/`transform_note`
(e.g. *"quantification unit unknown ... non-integer values with gene-level
identifiers, so assumed FPKM"*) — never silently presented as a
submitter-labeled unit.

## TPM computation, and why clean-gene renormalization inflates some samples

`rnaseq_finalize` computes TPM once over the *full* mapped gene set
(`compute_tpm`, using each gene's median included-transcript length from
the GENCODE reference), then restricts rows to the clean gene set and
renormalizes to sum-1e6 **again**, this time only among that clean subset.

If a sample has substantial expression outside the clean set (rRNA,
non-coding RNA, contamination), a gene's relative share can jump sharply
after this second renormalization. Live example: GSE253260 (a PDAC
cohort) — hemoglobin (HBB/HBA1/HBA2) went from 68% of raw counts to 96% of
final clean-gene TPM for the worst-contaminated sample.

**This is correct, not a bug** (confirmed 2026-08-20): restricting the
denominator to protein-coding genes and renormalizing reveals that, once
non-coding/contamination reads are excluded, almost nothing else is left
in the coding transcriptome for such samples — the right signal that a
sample is contaminated beyond recovery, not a distortion introduced by the
math. Don't flag a large jump between a gene's raw-count fraction and its
final clean-gene-TPM fraction as a conversion/mapping bug on its own; to
judge a cohort/sample's actual contamination severity, check the *raw*
counts' fraction directly (sum the candidate genes' raw counts over total
raw counts per sample, before any TPM/clean-set processing).

## Sample-ID matching (`geotool/sample_id_matching.py`)

An expression matrix's column headers almost never use `gsm_id` — they use
whatever label the submitter chose (`DMSO_1`, `D5_EPZ.r1`, `dmso1`, ...).
`match_expression_columns` resolves each column to a `gsm_id` via, in
order: exact `gsm_id` match, exact text match against any of the cohort's
own annotation columns, normalized-text exact match, substring match,
reverse-substring match, and — only for whatever's left unresolved by name,
and only when the remaining count exactly matches the count of unused
`gsm_id`s — GEO submission order as a last resort. An ambiguous match is
left unresolved (`None`) rather than guessed. Result is written to
`sample_id_map.tsv` and merged onto the cohort's own
`data/series/<GSE_ID>/annotation.tsv` as `expression_id`/
`sample_id_match_method`/`sample_id_match_confidence` columns, which then
flow into the harmonized sample table for free.

## LLM (Claude) usage — opt-in by default everywhere

No `geotool` command requires `ANTHROPIC_API_KEY` by default:
- `geotool download` — `--clinical-annotate` (default **off**) opts into
  Claude-based treatment/response/RECIST/survival column unification.
  Without it, `annotation.tsv` still gets the deterministic cleanup
  (constant-column drop, `"Label: "` prefix strip).
- `geotool search` — `--llm-annotate` (default **off**) opts into
  per-candidate Claude classification of tissue/diagnosis/sample source
  (one call per candidate, up to `--max-results`).
- `geotool harmonize` — `--match-columns` (default **on**, one call per
  *run*, not per-cohort) does cross-cohort column-concept matching;
  `--no-match-columns` opts out. `--llm-annotate` (default off) separately
  opts into per-cohort tissue/diagnosis backfill.

## Additional session memory

Claude's own cross-session memory (outside this repo, local to this
machine — see `~/.claude/projects/.../memory/`) may hold additional
project-specific learnings not duplicated here.
