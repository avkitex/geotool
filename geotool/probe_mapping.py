"""PHASE 2 (not implemented): map microarray probes to genes.

Planned strategies, chosen once per platform (GPL) and reused for every
series on that platform so results stay consistent:

- "first_probe": pick a single canonical probe per gene (persisted to
  data/platforms/<GPL>/probe_gene_map.tsv so the choice isn't re-derived
  differently across runs/series).
- "mean" / "median": aggregate all probes mapping to a gene at
  expression-matrix-load time, no persisted per-gene choice needed.
"""
