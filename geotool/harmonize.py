"""PHASE 3 (not implemented): unify sample annotation across cohorts.

Planned design: read every chosen cohort's data/series/<GSE>/series.tsv and
samples.tsv (written by geotool.annotate), map messy characteristic column
names (e.g. "tissue" vs "Tissue" vs "organ") onto a canonical schema (tissue,
disease_state, sex, age, treatment, cell_type, ...) via a small alias file
(YAML/JSON, not a database), and concatenate into one tidy master annotation
table keyed by gsm_id/gse_id. Exact alias-file format is still an open
design decision.
"""
