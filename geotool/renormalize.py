"""Phase 2b: RMA renormalization of Affymetrix microarray series from raw CEL
files, via R/Bioconductor.

Opt-in (see --rma on `geotool download`) because it needs R installed and
`Rscript` on PATH, on top of everything Phase 2 (download.py) already does
from the submitter's own, possibly inconsistent, already-summarized values.
Bioconductor itself and every R package RMA needs (BiocManager, affy/oligo,
and the chip-specific CDF/pd.* package) are installed lazily, on first use,
into a user-writable library (R_LIBS_USER) -- no manual R setup is required
beyond having R itself. Any missing prerequisite (no Rscript, no package
mapping for the platform, or an install/run failure) raises RmaUnavailable,
which callers catch and treat as "skip RMA for this series", never as a
reason to fail the whole cohort download.

RMA itself (background correction, quantile normalization, median-polish
summarization) is shelled out to R rather than reimplemented in Python: the
Bioconductor implementation is what GEO2R and the published literature use,
so results stay reproducible and comparable across studies.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from string import Template

import pandas as pd

# GPL id -> (chip family, package). "3prime" arrays (older, whole-probe
# designs like HG-U133) go through affy::ReadAffy/rma with a CDF package --
# the value here is the *cdfname* (e.g. "hgu133plus2"), matching ReadAffy's
# own argument; the actual Bioconductor package to install is that name plus
# a literal "cdf" suffix (e.g. "hgu133plus2cdf" -- NOT the same as the
# "hgu133plus2" gene-annotation package). "gene_st" arrays (Gene/Exon ST, HTA,
# Clariom) go through oligo::read.celfiles/rma with a pd.* platform-design
# package, where the value here *is* the installable package name as-is. An
# unlisted GPL just skips RMA for that series -- extend this table as new
# platforms come up rather than guessing a package name.
#
# GPL17586/GPL5175/GPL5188/GPL13667 added and each live-verified (a real
# oligo::read.celfiles + rma() run against that platform's own CEL files,
# not just "the package installs") after `--rma` first shipped only knowing
# 9 platforms. GPL5175 and GPL5188 are the *same* physical HuEx-1.0-st array
# and CEL data -- GEO just has two separate platform records for it (gene-
# level vs exon-level probe-set annotation), so both map to the one pd.*
# package that matches the actual chip, not the annotation choice. GPL17586
# is likewise the same HTA-2.0 hardware as GPL16686 (a different GEO
# platform record, same array), hence the same pd.hta.2.0. GPL13667 (HG-U219)
# looks like a classic whole-probe "3prime" design but was released on
# Affymetrix's newer GeneTitan/Command-Console CEL format -- verified it
# needs the gene_st-style oligo::read.celfiles path (pd.hg.u219), not
# affy::ReadAffy, despite not being a Gene/Exon ST array biologically.
#
# GPL15048 ("Rosetta/Merck Human RSTA Custom Affymetrix 2.0 microarray
# [HuRSTA_2a520709.CDF]") was checked too and deliberately left out: it's a
# one-off custom re-annotation of a chip, not a standard commercial array,
# and has no CDF/pd.* package anywhere in Bioconductor's repository (checked
# via available.packages(repos = BiocManager::repositories()) -- zero
# matches for any hurst*/rsta*-shaped name). RMA support for it would need a
# custom CDF from outside Bioconductor's normal install path, which this
# lazy ensure_pkg()-based installer has no way to fetch.
_CHIP_PACKAGES: dict[str, tuple[str, str]] = {
    "GPL96": ("3prime", "hgu133a"),
    "GPL97": ("3prime", "hgu133b"),
    "GPL570": ("3prime", "hgu133plus2"),
    "GPL571": ("3prime", "hgu133a2"),
    "GPL1261": ("3prime", "mouse4302"),
    "GPL6244": ("gene_st", "pd.hugene.1.0.st.v1"),
    "GPL6246": ("gene_st", "pd.mogene.1.0.st.v1"),
    "GPL11532": ("gene_st", "pd.hugene.2.0.st"),
    "GPL16686": ("gene_st", "pd.hta.2.0"),
    "GPL17586": ("gene_st", "pd.hta.2.0"),
    "GPL5175": ("gene_st", "pd.huex.1.0.st.v2"),
    "GPL5188": ("gene_st", "pd.huex.1.0.st.v2"),
    "GPL13667": ("gene_st", "pd.hg.u219"),
}

# Shared by both templates below. Uses R_LIBS_USER (a per-user library R
# already knows about) rather than the system library, since the latter is
# frequently not writable without admin rights -- discovered the hard way
# when a first real install attempt failed with exactly that error. ensure_pkg
# only calls BiocManager::install when the package isn't already loadable, so
# repeat runs on an already-provisioned machine pay no install cost at all.
_R_PREAMBLE = """\
lib <- Sys.getenv("R_LIBS_USER")
if (nzchar(lib)) {
  dir.create(lib, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(lib, .libPaths()))
}
ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    if (!requireNamespace("BiocManager", quietly = TRUE)) {
      install.packages("BiocManager", repos = "https://cloud.r-project.org")
    }
    BiocManager::install(pkg, update = FALSE, ask = FALSE)
  }
}
"""

# string.Template ($identifier) rather than str.format(), so the many literal
# R braces below don't need doubling-up escaping.
_RMA_3PRIME_TEMPLATE = Template(_R_PREAMBLE + """\
ensure_pkg("affy")
ensure_pkg("${package}cdf")
suppressMessages({
  library(affy)
  library(${package}cdf)
})
cel_files <- c($cel_files)
raw <- ReadAffy(filenames = cel_files, cdfname = "$package")
eset <- rma(raw)
write.csv(exprs(eset), file = $out_csv)
""")

_RMA_GENE_ST_TEMPLATE = Template(_R_PREAMBLE + """\
ensure_pkg("oligo")
ensure_pkg("$package")
suppressMessages({
  library(oligo)
  library($package)
})
cel_files <- c($cel_files)
raw <- read.celfiles(cel_files, pkgname = "$package")
eset <- rma(raw)
write.csv(exprs(eset), file = $out_csv)
""")


class RmaUnavailable(Exception):
    """RMA can't be run for this series -- caller should skip, not fail."""


def chip_package_for(gpl_id: str) -> tuple[str, str]:
    package = _CHIP_PACKAGES.get(gpl_id)
    if package is None:
        raise RmaUnavailable(f"no known Bioconductor CDF/pd package for {gpl_id}")
    return package


def _r_quote(path: Path) -> str:
    # Forward slashes read fine on Windows R and sidestep backslash-escaping
    # inside the R string literal.
    return '"' + str(path).replace("\\", "/") + '"'


def run_rma(cel_files: dict[str, Path], gpl_id: str, timeout: int = 3600) -> pd.DataFrame:
    """Run RMA on one platform's CEL files via Rscript.

    cel_files maps gsm_id -> CEL path; the returned probes x samples
    DataFrame's columns are exactly those gsm_ids (assigned positionally from
    the order passed to R, not parsed back from R's own, often-mangled,
    column names).

    The generated R script installs any missing prerequisite (BiocManager,
    affy/oligo, the chip's CDF/pd package) before running RMA, so the default
    timeout is generous enough to cover a first-time install (mostly network
    time) on top of the RMA computation itself; repeat calls for an
    already-provisioned platform are fast.

    Raises RmaUnavailable if Rscript isn't on PATH, the platform has no known
    chip package, or the R run (install or RMA) fails -- callers should catch
    this and fall back to the submitter-value expression matrix.
    """
    if not cel_files:
        raise RmaUnavailable("no CEL files to normalize")
    if shutil.which("Rscript") is None:
        raise RmaUnavailable("Rscript not found on PATH -- install R to use --rma")

    family, package = chip_package_for(gpl_id)
    template = _RMA_3PRIME_TEMPLATE if family == "3prime" else _RMA_GENE_ST_TEMPLATE

    gsm_ids = list(cel_files.keys())
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        out_csv = tmp_dir / "rma_matrix.csv"
        script_path = tmp_dir / "rma.R"
        cel_list = ", ".join(_r_quote(cel_files[gsm_id]) for gsm_id in gsm_ids)
        script_path.write_text(
            template.substitute(package=package, cel_files=cel_list, out_csv=_r_quote(out_csv)),
            encoding="utf-8",
        )
        proc = subprocess.run(
            ["Rscript", str(script_path)], capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode != 0:
            raise RmaUnavailable(f"Rscript failed for {gpl_id} ({package}): {proc.stderr.strip()[-500:]}")
        if not out_csv.exists():
            raise RmaUnavailable(f"Rscript produced no output for {gpl_id}")
        matrix = pd.read_csv(out_csv, index_col=0)

    if matrix.shape[1] != len(gsm_ids):
        raise RmaUnavailable(
            f"RMA output for {gpl_id} has {matrix.shape[1]} columns, expected {len(gsm_ids)}"
        )
    matrix.columns = gsm_ids
    return matrix
