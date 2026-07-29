"""Phase 2b: RMA renormalization of Affymetrix microarray series from raw CEL
files, via R/Bioconductor.

Opt-in (see --rma on `geotool download`) because it needs R plus Bioconductor's
`affy`/`oligo` packages and a chip-specific CDF or pd.* platform-design
package installed on the machine running geotool -- on top of everything
Phase 2 (download.py) already does from the submitter's own, possibly
inconsistent, already-summarized values. Any missing prerequisite raises
RmaUnavailable, which callers catch and treat as "skip RMA for this series",
never as a reason to fail the whole cohort download.

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
}

_RMA_3PRIME_TEMPLATE = """\
suppressMessages({{
  library(affy)
  library({package}cdf)
}})
cel_files <- c({cel_files})
raw <- ReadAffy(filenames = cel_files, cdfname = "{package}")
eset <- rma(raw)
write.csv(exprs(eset), file = {out_csv})
"""

_RMA_GENE_ST_TEMPLATE = """\
suppressMessages({{
  library(oligo)
  library({package})
}})
cel_files <- c({cel_files})
raw <- read.celfiles(cel_files, pkgname = "{package}")
eset <- rma(raw)
write.csv(exprs(eset), file = {out_csv})
"""


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


def run_rma(cel_files: dict[str, Path], gpl_id: str) -> pd.DataFrame:
    """Run RMA on one platform's CEL files via Rscript.

    cel_files maps gsm_id -> CEL path; the returned probes x samples
    DataFrame's columns are exactly those gsm_ids (assigned positionally from
    the order passed to R, not parsed back from R's own, often-mangled,
    column names).

    Raises RmaUnavailable if Rscript isn't on PATH, the platform has no known
    chip package, or the R run fails -- callers should catch this and fall
    back to the submitter-value expression matrix.
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
            template.format(package=package, cel_files=cel_list, out_csv=_r_quote(out_csv)),
            encoding="utf-8",
        )
        proc = subprocess.run(
            ["Rscript", str(script_path)], capture_output=True, text=True, timeout=1800
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
