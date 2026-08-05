# Vendored InMoose (limma subset)

Source: https://github.com/epigenelabs/inmoose, version **0.9.1** (PyPI
sdist `inmoose-0.9.1.tar.gz`, retrieved 2026-08). InMoose is a Python port
of Bioconductor's `limma`/`edgeR`/`DESeq2`, published and validated against
the real R packages (Pearson r = 1.0, log-fold-change differences ~1e-14) --
see [Colange et al., BMC Bioinformatics 2025](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-025-06180-7).

## Why vendored instead of a normal pip dependency

InMoose has never published a prebuilt wheel for any release -- every
version on PyPI is source-only. Building it from source requires a C++
compiler, needed only for its `edgepy` submodule (a Cython port of edgeR)
and its `fastcluster` dependency, neither of which `limma` uses. On a
machine without a C++ toolchain already installed (e.g. plain Windows),
that's a multi-GB compiler install just to reach code this project never
calls. Only `limma`, `utils`, and `diffexp` (limma's `topTable` imports
`diffexp.DEResults` for its return type) are included here, all pure
Python, so no compiler is needed.

`geotool/diffexp.py` is the only place in this repo that imports from this
directory.

## What's excluded from upstream

- `edgepy/`, `pycombat/`, `deseq2/`, `cohort_qc/`, `consensus_clustering/`,
  `data/`, `sim/` -- not needed for limma; several transitively require the
  compiled `edgepy_cpp` extension anyway (see `pycombat/pycombat_seq.py`).
- `utils/stats.py` and `utils/_stats.pyx` -- needs the compiled `stats_cpp`
  extension; its symbols (`dnbinom_mu`/`dnorm`/`pnorm`/`pt`/`rnbinom`, all
  negative-binomial helpers) aren't used by `limma`.

## Patches on top of upstream

Import-graph trims:

- `inmoose/__init__.py`: only imports `utils` and `limma` (upstream also
  imports `edgepy`, `pycombat`, `deseq2`, `cohort_qc`,
  `consensus_clustering` unconditionally, which would require the compiled
  extension even though this subset never calls them).
- `inmoose/utils/__init__.py`: drops the `from .stats import ...` line for
  the reason above.

One upstream bugfix (real logic bug, not specific to trimming):

- `inmoose/limma/ebayes.py`: `Infdf = out["df_prior"] > 1e6` wrapped in
  `np.asarray(...)`. When `df_prior` is scalar and equals `np.inf` (the
  no-covariate, non-robust case with no evidence for a finite prior --
  common with small/all-zero-variance gene sets), the comparison produces
  a native Python `bool` rather than a numpy `bool_`. `~` on a native
  Python bool inverts it as a two's-complement int (`~True == -2`, not
  `False`), which then gets used as a DataFrame column position a few
  lines later and raises `KeyError: -2`. Wrapping in `np.asarray` keeps it
  a proper 0-d numpy bool array, where `~` means logical negation as the
  surrounding code assumes. Worth reporting upstream.

Everything else -- `limma/*.py` (bar the one line above), `diffexp/*.py`,
the rest of `utils/*.py` -- is byte-for-byte upstream source, including
original copyright headers.

## License

GPL-3.0-or-later (see `LICENSE` in this directory), same as upstream
InMoose. This is a separate license from the rest of this repository;
anything that imports from this directory should be treated as a
GPL-licensed combination.
