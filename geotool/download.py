"""PHASE 2 (not implemented): download + renormalize expression data.

Planned design (see plan doc for full rationale):

- Classify each platform in data/series/<GSE>/series.tsv by its GEO
  `technology`/title string into one of: "rnaseq", "affy_microarray",
  "illumina_microarray", "other" (skip "other" for now).
- RNA-seq series: fetch supplementary counts + the samples.tsv annotation
  already produced by geotool.annotate, no renormalization.
- Affy/Illumina series: download raw CEL/IDAT supplementary files and
  renormalize (RMA via Bioconductor `affy`/`oligo`, likely through an
  `rpy2` bridge since Python has no mature RMA implementation). This needs
  a decision on the R dependency before it's built.
"""
