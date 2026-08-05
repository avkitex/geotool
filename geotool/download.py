"""Phase 2: download expression data for a chosen cohort.

Routes by platform assay_type (from platform_classify, already computed in
series_row()'s platform_details): RNA-seq/scRNA-seq -> download the
supplementary expression file(s) verbatim, no parsing or reshaping.
Microarray -> reshape each sample's own data table into a probe matrix, map
probes to genes via probe_mapping.py -- including, for two-channel Agilent
samples that publish per-channel columns, each channel's own matrix as an
*additional* output alongside the always-produced ratio-based one (see
build_and_map_channel_expression_matrices / probe_mapping.py's
detect_channel_columns). Always also produce a cleaned, semantically-unified
per-sample annotation table via clinical_annotate.py.

Phase 2b (opt-in via download_cohort(..., rma=True)): for Affymetrix
microarray series, also download raw CEL files and RMA-renormalize them via
renormalize.py, producing expression_rma.tsv.gz alongside the
submitter-value expression.tsv.gz above. See renormalize.py for why this
needs R/Bioconductor and is not the default. CEL files are deleted once a
platform's RMA run succeeds -- they're fully captured by the resulting
probe matrix and otherwise just sit on disk as raw, mostly-redundant data.

All expression/probe matrices are written gzip-compressed with values
rounded to 3 decimal places (see _write_matrix) to keep them from ballooning
into hundreds of MB for large series.

A cohort that's already been downloaded (annotation.tsv + series.tsv already
on disk) is reused rather than re-fetched/re-processed -- see
download_cohort(..., force=True) to force a full redo regardless, and
_cached_result for what "already downloaded" checks. The one exception:
if --rma is newly requested on a cohort previously downloaded without it,
only the missing RMA output is computed (the series still needs
re-fetching for its CEL files, but the already-done clinical_annotate LLM
call and probe/gene matrix build are not repeated).

Before doing any of the above, download_cohort() checks eligibility and fails
fast (UnsupportedCohortError) rather than attempting a cohort it was never
designed for: non-human organism, or a microarray platform whose content
isn't mRNA expression (miRNA/lncRNA-only or CNA arrays -- a platform that
combines mRNA with miRNA/lncRNA content on one chip is fine, see
platform_classify.classify_array_content) or is too old/low-density
(platform_classify.platform_supported). A series with a mix of supported and
unsupported platforms proceeds using only the supported ones. A SuperSeries
(see resolve_download_targets / geo_fetch.resolve_leaf_series_ids) is never
downloaded as itself -- its own fetched record merges every subseries'
samples together with no reliable way to separate them back out, so the CLI
always expands it into its subseries first and calls download_cohort once
per leaf, each getting its own independent eligibility check.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import re
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from geotool import (
    annotate,
    clinical_annotate,
    companion_platforms,
    config,
    geo_fetch,
    platform_classify,
    probe_mapping,
    renormalize,
)

_SKIP_EXTENSIONS = (
    ".bam", ".bai", ".fastq", ".fastq.gz", ".fq", ".fq.gz",
    ".bw", ".bigwig", ".cel", ".cel.gz",
)


class UnsupportedCohortError(Exception):
    """A (sub)series failed an eligibility check (organism / array content / platform
    coverage) and shouldn't be attempted at all -- raised instead of letting it fail
    deep inside matrix-building with a confusing error, or worse, silently blending
    incompatible platforms (e.g. a CNA array) into an "expression" matrix.
    """


def _series_dir(gse_id: str, series_dir: Path | None = None) -> Path:
    out = (series_dir or config.SERIES_DIR) / gse_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _persist_series_annotation(out_dir: Path, series_row: dict, samples: pd.DataFrame) -> None:
    pd.DataFrame([series_row]).to_csv(out_dir / "series.tsv", sep="\t", index=False)
    samples.to_csv(out_dir / "samples.tsv", sep="\t", index=False)


def _write_channel_roles(out_dir: Path, channel_roles: dict) -> None:
    """Sidecar for probe_mapping.detect_reference_channel's result -- a
    separate small file rather than a series.tsv column since it's only ever
    produced partway through a fresh download, after series.tsv is already
    written. Only written when a confident call was actually made (method !=
    "ambiguous"), so its mere existence on disk means "there's a call here"
    for both the fresh-download and cache-reuse (_cached_result) paths.
    """
    if not channel_roles or channel_roles.get("method") == "ambiguous":
        return
    with open(out_dir / "channel_roles.json", "w", encoding="utf-8") as f:
        json.dump(channel_roles, f, indent=2)


def _load_channel_roles(out_dir: Path) -> dict | None:
    path = out_dir / "channel_roles.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_expression_qc(out_dir: Path, primary_path: Path | None, primary_unit: str | None, qc_notes: list[str]) -> None:
    """Sidecar for check_rnaseq_expression_qc/check_expression_qc's findings --
    same reasoning as _write_channel_roles: only written when there's
    actually something to say (a primary file was picked, and/or QC notes
    exist), so both the fresh-download and cache-reuse paths can tell "was
    this checked at all" from the file's mere existence.
    """
    if primary_path is None and not qc_notes:
        return
    payload = {
        "primary_expression_file": str(primary_path) if primary_path else None,
        "primary_expression_unit": primary_unit,
        "notes": qc_notes,
    }
    with open(out_dir / "expression_qc.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_expression_qc(out_dir: Path) -> dict | None:
    path = out_dir / "expression_qc.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_matrix(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    """Write a probe/gene x sample matrix as gzip-compressed TSV.

    Numeric columns are rounded to 3 decimal places first -- expression
    values don't carry meaningful precision beyond that, and it also lets
    gzip compress the repeated short decimals much better, so both the
    rounding and the compression are there to control on-disk size for
    matrices that can otherwise run into the hundreds of MB.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols):
        df = df.copy()
        df[numeric_cols] = df[numeric_cols].round(3)
    df.to_csv(path, sep="\t", index=index, compression="gzip")


def _should_skip_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in _SKIP_EXTENSIONS)


def _is_cel_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".cel") or path.endswith(".cel.gz")


def _as_https(url: str) -> str:
    """requests has no ftp:// adapter, but NCBI's FTP host serves the same
    paths over HTTPS (verified live for ftp.ncbi.nlm.nih.gov) -- rewrite
    rather than failing outright, since GEO supplementary_file URLs are
    commonly given as ftp://.
    """
    if url.startswith("ftp://"):
        return "https://" + url[len("ftp://") :]
    return url


def _download_file(url: str, out_dir: Path, retries: int = 3) -> Path | None:
    """Download url to out_dir, retrying transient failures.

    Non-streaming on purpose: chunked stream=True reads of these files
    reliably hit mid-transfer connection drops (IncompleteRead) in some
    network environments even though a plain buffered GET of the exact same
    URL succeeds every time. Supplementary expression files are already
    filtered to processed matrices (raw sequencing/CEL excluded above), so
    they're small enough that buffering the whole response is fine.
    """
    filename = os.path.basename(urlparse(url).path)
    if not filename:
        return None
    dest = out_dir / filename
    if dest.exists():
        return dest

    https_url = _as_https(url)
    last_exc: Exception | None = None
    for _attempt in range(retries):
        try:
            resp = requests.get(https_url, timeout=120)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return dest
        except requests.RequestException as exc:
            last_exc = exc
            dest.unlink(missing_ok=True)  # don't leave a partial file behind
    print(f"    could not download {url}: {last_exc}")
    return None


def download_rnaseq_files(gse, out_dir: Path) -> list[Path]:
    """Download the series'/samples' supplementary expression files verbatim --
    no parsing or reshaping. Raw-data extensions (FASTQ/BAM/CEL/...) are skipped.
    """
    urls = set(gse.metadata.get("supplementary_file", []))
    for gsm in gse.gsms.values():
        for key, values in gsm.metadata.items():
            if key.startswith("supplementary_file"):
                urls.update(values)

    expr_dir = out_dir / "expression"
    downloaded = []
    for url in sorted(urls):
        if not url or url.strip().upper() == "NONE" or _should_skip_url(url):
            continue
        expr_dir.mkdir(parents=True, exist_ok=True)
        path = _download_file(url, expr_dir)
        if path is not None:
            downloaded.append(path)
    return downloaded


# Quantification-unit keywords a supplementary filename might carry, ranked
# best-first: TPM > FPKM > RPKM (paired-end-normalized equivalent of FPKM,
# ranked alongside it) > CPM > raw counts ("count", not "counts", so
# "count_matrix"/"gene_count"-style singular names still match -- live
# example, GSE273376's "..._count_matrix.csv.gz"). An RNA-seq series
# routinely also publishes non-matrix supplementary files alongside the real
# expression matrix (differential-expression result tables, splicing-
# analysis output, ...) -- live example, GSE163305: 4 supplementary files,
# only one ("..._FPKM_....csv.gz") is an actual gene x sample matrix; the
# other 3 are rMATS splicing output and a Cuffdiff-style DE table whose
# log2_fold_change column is legitimately negative, which would be a false
# positive for check_expression_qc below if it were treated as an
# expression matrix.
_QUANT_UNIT_PRIORITY = ["tpm", "fpkm", "rpkm", "cpm", "count"]

# A filename carrying its own GSM accession is inherently a per-sample
# fragment, never a whole-cohort combined matrix, regardless of what
# quantification-unit keyword also appears in it -- live example,
# GSE236498/GSE236499: 12 "GSM*_gene_counts.txt.gz" files, one per sample,
# no combined matrix at all. Without this exclusion, select_primary_
# expression_file would pick one arbitrary sample's own file and silently
# misrepresent the whole cohort's data as if it were that one sample's.
_GSM_NAME_RE = re.compile(r"GSM\d+", re.IGNORECASE)

# Only these extensions (after stripping a trailing .gz) are ever plausible
# for a delimited/Excel expression matrix -- live example, GSE161706's
# "..._dexseq_count.py.gz": a compressed Python *script*, not data, that
# would otherwise match the "count" unit keyword purely because of its name.
_DATA_FILE_EXTENSIONS = (".txt", ".tsv", ".csv", ".xlsx", ".xls")


def _is_data_file(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".gz"):
        name = name[: -len(".gz")]
    return name.endswith(_DATA_FILE_EXTENSIONS)


# A filename carrying one of these is a comparison/analysis output derived
# from an expression matrix, not the matrix itself -- even when it also
# happens to carry a quantification-unit keyword. Live example, GSE194360/
# GSE194362: "..._snp_counts_significance.csv.gz" matches "count" but is a
# differential-significance table, not a per-gene-per-sample matrix (same
# false-positive risk already handled for GSE163305's "_GSK_vs_DMSO_D6.csv.gz"
# -- that one just didn't happen to also contain a unit keyword). ".original"
# is a generic backup-file marker (e.g. "<name>.original.tsv.gz", written
# when replacing a primary file in place elsewhere -- see
# gene_symbol_mapping) -- without excluding it, a backup sitting next to the
# real file it was copied from is an equally-ranked, filesystem-order-
# dependent tie for the same unit, live-verified to nondeterministically win.
_NON_MATRIX_KEYWORDS = ("diff", "deg", "significance", "clinical", "rmats", "dexseq", "novel_filtered", ".original")


def _passes_basic_matrix_filters(name: str) -> bool:
    """True if `name` isn't disqualified outright as a per-sample fragment, a
    non-data file, or a derived comparison/analysis output -- the same
    exclusions _classify_candidate applies, minus the unit-keyword check.
    Factored out so a candidate with no recognizable unit keyword can still
    be considered by content (see select_primary_expression_file_by_content)
    using the exact same disqualification rules, without also matching
    obviously-wrong files like a splicing-analysis script or a per-sample
    fragment.
    """
    candidate = Path(name)
    lname = candidate.name.lower()
    return not (
        _GSM_NAME_RE.search(candidate.name) or not _is_data_file(candidate) or any(k in lname for k in _NON_MATRIX_KEYWORDS)
    )


def _classify_candidate(name: str) -> tuple[int, str] | None:
    """(rank, unit) if `name` (a filename, or an archive member name -- it
    doesn't need to exist on disk) looks like a plausible primary expression
    matrix, honoring the same exclusions select_primary_expression_file
    documents; None otherwise. Factored out so archive members can be
    ranked by the exact same rules as plain downloaded files (see
    resolve_primary_expression_matrix).
    """
    if not _passes_basic_matrix_filters(name):
        return None
    lname = Path(name).name.lower()
    for rank, unit in enumerate(_QUANT_UNIT_PRIORITY):
        if unit in lname:
            return rank, unit
    return None


def select_primary_expression_file(paths: list[Path]) -> tuple[Path, str] | None:
    """Among downloaded RNA-seq supplementary files, pick the one that looks
    like the actual gene-expression quantification matrix, by
    _QUANT_UNIT_PRIORITY -- (path, unit), or None if no filename carries a
    recognizable unit keyword at all (nothing is guessed at that point,
    rather than risk QC-checking an unrelated file as if it were expression
    data). Files named after an individual GSM accession (_GSM_NAME_RE),
    that aren't a plausible data file at all (_is_data_file), or that look
    like a derived comparison/analysis output rather than the matrix itself
    (_NON_MATRIX_KEYWORDS) are never candidates. Doesn't look inside .zip/
    .tar archives -- see resolve_primary_expression_matrix for that.
    """
    best: tuple[int, Path, str] | None = None
    for path in paths:
        classified = _classify_candidate(path.name)
        if classified is None:
            continue
        rank, unit = classified
        if best is None or rank < best[0]:
            best = (rank, path, unit)
    return (best[1], best[2]) if best else None


def _load_expression_file_for_qc(path: Path) -> pd.DataFrame | None:
    """Best-effort read of an arbitrary submitter supplementary file for QC
    purposes only -- format/delimiter varies a lot across submitters, so
    this sniffs rather than assumes, and returns None (not raised) on
    anything it can't parse. Never used for anything but check_expression_qc:
    a QC nice-to-have must never be able to fail a real download.
    """
    try:
        if path.name.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(path)
        return pd.read_csv(path, sep=None, engine="python", compression="infer")
    except Exception:
        return None


# A real gene-level (or even transcript-level) expression matrix always has
# at least this many rows -- live examples run into the tens of thousands
# (GSE253260: 60671, GSE172356: 45140). A supplementary table/figure excerpt
# that isn't the real matrix (live example, GSE161706's sole remaining file
# after excluding script files, "..._Processed_data_for_Table_S3_Figure_6.xlsx")
# parses to a handful of rows or fewer, so this alone rules those out before
# the column-count check below even runs.
_MIN_MATRIX_ROWS = 1000

# How far a candidate's data-column count (assumed one non-numeric gene/probe
# ID column, so total columns - 1) may be from the cohort's actual sample
# count and still be treated as its combined matrix. Real submitter matrices
# routinely drop a handful of samples that failed QC, or add a few
# replicate/multi-run columns, so exact equality is too strict -- but this
# must still firmly reject an unrelated table that merely survived the
# filename exclusions. Live-calibrated against 5 real single-file cases
# (GSE253260 396 vs 397, GSE310252 118 vs 118, GSE224564 174 vs 175,
# GSE248014 45 vs 45, GSE172356 62 vs 62) and 2 real multi-file sum cases
# (GSE293744 12+36=48 vs 48, GSE131050 66+125=191 vs 191).
def _matches_sample_count(n_cols: int, n_samples: int) -> bool:
    if n_samples <= 0:
        return False
    return abs(n_cols - n_samples) <= max(5, round(0.15 * n_samples))


def _content_verified_column_count(path: Path) -> int | None:
    """Best-effort data-column count (total columns minus one assumed
    gene/probe-ID column) for content-based verification, or None if the
    file can't be parsed at all or is implausibly small to be a real
    expression matrix (_MIN_MATRIX_ROWS) -- e.g. a supplementary table
    excerpt rather than the whole-cohort matrix.
    """
    df = _load_expression_file_for_qc(path)
    if df is None or df.shape[0] < _MIN_MATRIX_ROWS:
        return None
    return max(df.shape[1] - 1, 0)


def select_primary_expression_file_by_content(paths: list[Path], n_samples: int) -> tuple[list[Path], str] | None:
    """Fallback for when select_primary_expression_file finds no filename
    carrying a recognizable quantification-unit keyword: verify by content
    instead of guessing from the name. Only ever considers candidates that
    already pass _passes_basic_matrix_filters (same GSM-name/non-data/
    derived-comparison exclusions as the filename path), and only accepts
    one when its (or, for several small files together, their combined)
    data-column count is close to the cohort's actual sample count
    (_matches_sample_count) -- there is no real ambiguity left once content
    corroborates the column count against ground truth, unlike a bare
    "only one file remains" guess (which live-broke on GSE161706: a sole
    remaining ..._Table_S3_Figure_6.xlsx that looked like the only option by
    filename alone, but parses to 0 rows -- not the matrix).

    Returns (paths, "unknown") -- "unknown" because content verification
    doesn't tell us the quantification unit, only that the shape matches --
    or None if nothing verifies. Capped at 5 remaining candidates: beyond
    that, summing an arbitrary subset to hit the sample count by chance
    becomes a real risk rather than a confident signal.
    """
    candidates = [p for p in paths if _passes_basic_matrix_filters(p.name)]
    if not candidates or len(candidates) > 5:
        return None

    for path in candidates:
        n_cols = _content_verified_column_count(path)
        if n_cols is not None and _matches_sample_count(n_cols, n_samples):
            return [path], "unknown"

    counts = [(path, _content_verified_column_count(path)) for path in candidates]
    if len(counts) > 1 and all(n is not None for _, n in counts):
        total = sum(n for _, n in counts)
        if _matches_sample_count(total, n_samples):
            return [path for path, _ in counts], "unknown"

    return None


# Submitters routinely bundle a series' supplementary files (or even the one
# real combined matrix, alongside unrelated per-sample fragments) into an
# archive rather than publishing it standalone -- e.g. a "..._RAW.tar".
_ARCHIVE_EXTENSIONS = (".zip", ".tar", ".tar.gz", ".tgz")


def _is_archive(path: Path) -> bool:
    return path.name.lower().endswith(_ARCHIVE_EXTENSIONS)


def _archive_member_names(path: Path) -> list[str]:
    """Best-effort listing of every regular-file member inside a .zip/.tar/
    .tar.gz/.tgz archive, without extracting anything yet -- cheap enough to
    do for every archive, so only a member that's actually worth extracting
    (see resolve_primary_expression_matrix) ever gets pulled out.
    """
    try:
        if path.name.lower().endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                return [n for n in zf.namelist() if not n.endswith("/")]
        with tarfile.open(path) as tf:
            return [m.name for m in tf.getmembers() if m.isfile()]
    except Exception:
        return []


def _extract_archive_member_bytes(path: Path, member_name: str) -> bytes | None:
    try:
        if path.name.lower().endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                return zf.read(member_name)
        with tarfile.open(path) as tf:
            extracted = tf.extractfile(member_name)
            return extracted.read() if extracted else None
    except Exception:
        return None


def _strip_known_extensions(name: str) -> str:
    if name.lower().endswith(".gz"):
        name = name[: -len(".gz")]
    for ext in (".xlsx", ".xls", ".csv", ".tsv", ".txt"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def _needs_tsv_conversion(name: str) -> bool:
    """False for a file already in the guaranteed final format (plain,
    gzip-or-not, tab-separated -- GEO's usual ".tsv"/".txt" convention);
    True for anything that needs reformatting (Excel, comma-separated) or
    that isn't even a standalone file yet (an archive member).
    """
    return not name.lower().endswith((".tsv.gz", ".txt.gz", ".tsv", ".txt"))


def _read_dataframe_bytes(data: bytes, name: str) -> pd.DataFrame | None:
    """Same best-effort sniffing as _load_expression_file_for_qc, but reading
    from in-memory bytes (an already-extracted archive member) instead of a
    file on disk.
    """
    try:
        lname = name.lower()
        if lname.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(data))
        if lname.endswith(".gz"):
            data = gzip.decompress(data)
        return pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    except Exception:
        return None


def resolve_primary_expression_matrix(
    files: list[Path], out_dir: Path, n_samples: int | None = None
) -> tuple[Path, str] | None:
    """Find this RNA-seq cohort's primary expression matrix among its
    downloaded supplementary files -- including inside .zip/.tar/.tar.gz/
    .tgz archives -- and guarantee the result is a plain, gzip-compressed,
    tab-separated .tsv.gz file, regardless of how the submitter originally
    published it (Excel, comma-separated, or packed inside an archive
    alongside unrelated files). Pure format conversion only: rows/columns/
    values are carried through exactly as submitted -- e.g. a transcript-
    level file stays transcript-level, not aggregated to genes here (a
    separate concern).

    Returns (path, unit), or None if nothing recognizable was found (mirrors
    select_primary_expression_file). A plain downloaded file that's already
    an acceptable .tsv/.txt(.gz) is returned as-is, no new file written; an
    Excel/CSV file, or a member pulled out of an archive, is written as a
    new "<name>.tsv.gz" in out_dir (which should be the cohort's own
    expression/ directory) alongside the original download -- nothing
    already on disk is ever modified or deleted. If a candidate is found but
    can't actually be parsed/converted, its (extracted, if needed) original
    file is returned unconverted rather than None -- callers that go on to
    parse it themselves (check_rnaseq_expression_qc) can still report
    exactly which file and why, instead of that looking identical to "no
    candidate found at all".

    When no filename carries a recognizable unit keyword and `n_samples` is
    given, falls back to select_primary_expression_file_by_content -- but
    only its single-candidate result; a multi-file sum match is reported by
    check_rnaseq_expression_qc as a QC note instead of picked here, since
    "the primary file" is a one-file concept and no single one of those
    files is the whole matrix on its own. Content verification isn't
    attempted for archive members (out of scope for now -- plain downloaded
    files cover every real case seen so far).
    """
    candidates = list(files)
    archive_sources: dict[str, tuple[Path, str]] = {}  # virtual name -> (archive path, member name)
    for path in files:
        if not _is_archive(path):
            continue
        for member_name in _archive_member_names(path):
            if _classify_candidate(member_name) is None:
                continue
            virtual_name = Path(member_name).name
            candidates.append(Path(virtual_name))
            archive_sources[virtual_name] = (path, member_name)

    picked = select_primary_expression_file(candidates)
    if picked is None and n_samples is not None:
        by_content = select_primary_expression_file_by_content(candidates, n_samples)
        if by_content is not None and len(by_content[0]) == 1:
            picked = by_content[0][0], by_content[1]
    if picked is None:
        return None
    primary, unit = picked

    if primary.name in archive_sources:
        archive_path, member_name = archive_sources[primary.name]
        data = _extract_archive_member_bytes(archive_path, member_name)
        if data is None:
            return None  # genuinely couldn't even extract -- no file to point to at all
        source_name = Path(member_name).name
        if not _needs_tsv_conversion(source_name):
            dest = out_dir / source_name
            if not dest.exists():
                dest.write_bytes(data)
            return dest, unit
        df = _read_dataframe_bytes(data, source_name)
        if df is None:
            # Extracted fine but couldn't parse -- write the raw extracted
            # bytes out under their own name so the caller still has a real
            # file to point to and report "could not parse" against, same as
            # the plain-file case below (rather than looking identical to
            # "no candidate found at all").
            dest = out_dir / source_name
            if not dest.exists():
                dest.write_bytes(data)
            return dest, unit
        dest = out_dir / f"{_strip_known_extensions(source_name)}.tsv.gz"
        _write_matrix(df, dest, index=False)
        return dest, unit

    if not _needs_tsv_conversion(primary.name):
        return primary, unit
    df = _load_expression_file_for_qc(primary)
    if df is None:
        return primary, unit  # couldn't parse/convert -- return the original so the caller can still report it
    dest = out_dir / f"{_strip_known_extensions(primary.name)}.tsv.gz"
    _write_matrix(df, dest, index=False)
    return dest, unit


def check_rnaseq_expression_qc(
    paths: list[Path], out_dir: Path, n_samples: int | None = None
) -> tuple[Path | None, str | None, list[str]]:
    """resolve_primary_expression_matrix + probe_mapping.check_expression_qc
    on the resulting canonical .tsv.gz, for the common (one series- or
    sample-level matrix per series) case. Returns (primary_path, unit,
    qc_notes) -- primary_path/unit are None if nothing recognizable was
    found; qc_notes is empty if the file couldn't be parsed either (noted as
    its own entry) or nothing stood out.

    When resolve_primary_expression_matrix can't name a single primary file
    but `n_samples` is given, checks whether several remaining candidates'
    combined column count matches it (select_primary_expression_file_by_
    content) -- live example, GSE293744's two files (12 + 36 = 48, matching
    its 48 samples exactly) and GSE131050's two (66 + 125 = 191, matching
    its 191). Reported as a QC note pointing at the files rather than picked
    as "the" primary, since no single one of them is the whole matrix.
    """
    resolved = resolve_primary_expression_matrix(paths, out_dir, n_samples=n_samples)
    if resolved is None:
        notes = []
        if n_samples is not None:
            by_content = select_primary_expression_file_by_content(paths, n_samples)
            if by_content is not None and len(by_content[0]) > 1:
                names = ", ".join(p.name for p in by_content[0])
                notes.append(
                    f"no single combined matrix, but {len(by_content[0])} files together "
                    f"({names}) sum to a column count matching this cohort's {n_samples} samples "
                    "-- likely a legitimate multi-part matrix, see each file directly"
                )
        return None, None, notes
    primary_path, unit = resolved

    matrix = _load_expression_file_for_qc(primary_path)
    if matrix is None:
        return primary_path, unit, [f"{primary_path.name}: could not parse for QC"]

    notes = probe_mapping.check_expression_qc(matrix)
    return primary_path, unit, [f"{primary_path.name}: {note}" for note in notes]


def _combined_probe_gene_map(gpl_a: str, gpl_b: str) -> pd.DataFrame:
    """Union of two platforms' probe->gene maps, for a companion-chip pair
    whose combined sample column carries probes from both. The tiny handful
    of probe IDs both chips happen to share (see companion_platforms) keep
    gpl_a's mapping, matching combine_paired_probe_columns's own tie-break.
    """
    return pd.concat(
        [probe_mapping.get_or_build_probe_gene_map(gpl_a), probe_mapping.get_or_build_probe_gene_map(gpl_b)],
        ignore_index=True,
    ).drop_duplicates(subset="probe_id", keep="first")


def build_and_map_expression_matrix(gse, out_dir: Path) -> tuple[Path | None, list[str]]:
    """Probe matrix (raw) + gene-level matrix (mapped via probe_mapping.py).

    Returns (expression_path, qc_notes) -- qc_notes is
    probe_mapping.check_expression_qc's report on the final gene-level
    matrix (empty if nothing stood out).
    """
    probe_matrix = probe_mapping.build_probe_matrix(gse)
    if probe_matrix.empty:
        print("    no per-sample data tables found; skipping expression matrix")
        return None, []
    _write_matrix(probe_matrix, out_dir / "probe_matrix.tsv.gz")

    # A companion-chip pair (see companion_platforms.py -- e.g. GPL96+GPL97,
    # the two halves of the Affymetrix HG-U133 Set) means two GSM records
    # are actually one biological sample split across two arrays; combine
    # each matched pair's columns before any gene mapping happens, so the
    # rest of this function (and every downstream consumer) sees one sample,
    # not two.
    pairings = companion_platforms.detect_pairings(gse)
    flat_pairing = {gsm_a: gsm_b for pairing in pairings.values() for gsm_a, gsm_b in pairing.items()}
    if flat_pairing:
        pair_desc = ", ".join(f"{gpl_a}+{gpl_b}" for gpl_a, gpl_b in pairings)
        print(f"    combining {len(flat_pairing)} companion-chip sample pair(s) ({pair_desc})")
        probe_matrix = companion_platforms.combine_paired_probe_columns(probe_matrix, flat_pairing)

    # Usually one platform per series; handle the rare multi-platform case by
    # mapping+aggregating each platform's samples separately, then combining.
    # Each companion-chip pair is its own group spanning both platforms'
    # probe->gene maps; every other platform is its own singleton group.
    paired_gpl_ids = {gpl for pair in pairings for gpl in pair}
    singleton_gpl_ids = sorted(
        {p for gsm in gse.gsms.values() for p in gsm.metadata.get("platform_id", [])} - paired_gpl_ids
    )
    gene_frames = []
    for (gpl_a, gpl_b), pairing in pairings.items():
        combined_cols = [c for c in (f"{a}+{b}" for a, b in pairing.items()) if c in probe_matrix.columns]
        if not combined_cols:
            continue
        gene_frames.append(
            probe_mapping.aggregate_probes_to_genes(
                probe_matrix[combined_cols], _combined_probe_gene_map(gpl_a, gpl_b)
            )
        )
    for gpl_id in singleton_gpl_ids:
        sample_cols = [
            gsm_id for gsm_id, gsm in gse.gsms.items() if gpl_id in gsm.metadata.get("platform_id", [])
        ]
        cols = [c for c in sample_cols if c in probe_matrix.columns]
        if not cols:
            continue
        probe_gene_map = probe_mapping.get_or_build_probe_gene_map(gpl_id)
        gene_frames.append(probe_mapping.aggregate_probes_to_genes(probe_matrix[cols], probe_gene_map))

    if not gene_frames:
        print("    no probe->gene mapping available for this platform; wrote probe_matrix.tsv.gz only")
        return None, []

    expression = gene_frames[0]
    for other in gene_frames[1:]:
        expression = expression.merge(other, on=["gene_symbol", "entrez_id"], how="outer")

    qc_notes = probe_mapping.check_expression_qc(expression)
    expression_path = out_dir / "expression.tsv.gz"
    _write_matrix(expression, expression_path, index=False)
    return expression_path, qc_notes


def build_and_map_channel_expression_matrices(
    gse, out_dir: Path, platform_details: list[dict]
) -> tuple[dict[int, Path], dict]:
    """For two-channel Agilent samples whose own data table exposes
    recognizable per-channel columns (probe_mapping.detect_channel_columns),
    write each channel's own probe/gene matrix as an *additional* output --
    channel1_expression.tsv.gz / channel2_expression.tsv.gz -- alongside the
    ratio-based expression.tsv.gz that build_and_map_expression_matrix always
    produces unchanged. Neither channel1/channel2 is assumed to be "the real
    sample" or "the reference" -- both are always written, just named by
    channel number.

    Returns (channel_expression_paths, channel_roles) -- channel_roles is
    probe_mapping.detect_reference_channel's best-effort, confidence-gated
    guess at which channel is which. When it's confident (method != that
    dict's default "ambiguous"), this *additionally* writes
    channel_signal_expression.tsv.gz / channel_reference_expression.tsv.gz --
    plain copies of whichever channelN_expression.tsv.gz was determined to
    hold the actual biological sample / the fixed reference, respectively.
    Those neutral channelN_expression.tsv.gz files are unaffected either way.

    Most two-channel series don't publish per-channel columns at all (only
    the precomputed ratio), so this silently writes nothing when there's
    nothing to split -- never an error, and never touches expression.tsv.gz.
    """
    agilent_gpl_ids = {p["gpl_id"] for p in platform_details if p.get("vendor") == "agilent"}
    if not agilent_gpl_ids:
        return {}, {}

    channel1_matrix, channel2_matrix = probe_mapping.build_channel_probe_matrices(gse)
    if channel1_matrix.empty:
        return {}, {}

    # Only samples on an Agilent platform qualify, even though the channel
    # matrices themselves are built vendor-agnostically -- restrict both
    # matrices to those columns up front so probe- and gene-level outputs
    # stay consistent with each other.
    eligible_cols = [
        gsm_id
        for gsm_id in channel1_matrix.columns
        if agilent_gpl_ids & set(gse.gsms[gsm_id].metadata.get("platform_id", []))
    ]
    if not eligible_cols:
        return {}, {}
    channel1_matrix = channel1_matrix[eligible_cols]
    channel2_matrix = channel2_matrix[eligible_cols]

    channel_roles = probe_mapping.detect_reference_channel(gse, channel1_matrix, channel2_matrix)

    result: dict[int, Path] = {}
    channel_expressions: dict[int, pd.DataFrame] = {}
    for channel_num, probe_matrix in ((1, channel1_matrix), (2, channel2_matrix)):
        platform_ids = sorted({
            gpl_id
            for gsm_id in probe_matrix.columns
            for gpl_id in gse.gsms[gsm_id].metadata.get("platform_id", [])
            if gpl_id in agilent_gpl_ids
        })
        gene_frames = []
        for gpl_id in platform_ids:
            cols = [
                gsm_id for gsm_id in probe_matrix.columns
                if gpl_id in gse.gsms[gsm_id].metadata.get("platform_id", [])
            ]
            probe_gene_map = probe_mapping.get_or_build_probe_gene_map(gpl_id)
            gene_frames.append(probe_mapping.aggregate_probes_to_genes(probe_matrix[cols], probe_gene_map))

        _write_matrix(probe_matrix, out_dir / f"channel{channel_num}_probe_matrix.tsv.gz")
        if not gene_frames:
            continue

        expression = gene_frames[0]
        for other in gene_frames[1:]:
            expression = expression.merge(other, on=["gene_symbol", "entrez_id"], how="outer")

        expression_path = out_dir / f"channel{channel_num}_expression.tsv.gz"
        _write_matrix(expression, expression_path, index=False)
        result[channel_num] = expression_path
        channel_expressions[channel_num] = expression

    signal_channel = channel_roles.get("signal_channel")
    if signal_channel in channel_expressions:
        _write_matrix(channel_expressions[signal_channel], out_dir / "channel_signal_expression.tsv.gz", index=False)
    reference_channel = channel_roles.get("reference_channel")
    if reference_channel in channel_expressions:
        _write_matrix(channel_expressions[reference_channel], out_dir / "channel_reference_expression.tsv.gz", index=False)

    return result, channel_roles


def download_cel_files(gse, out_dir: Path) -> dict[str, Path]:
    """Download each sample's raw CEL supplementary file, keyed by gsm_id --
    used by renormalize.py to run RMA. Samples with no CEL supplementary file
    (e.g. a series that only ever published already-summarized values) are
    silently skipped, not an error: --rma just runs on whichever subset of
    samples actually has raw data.
    """
    cel_dir = out_dir / "cel"
    downloaded: dict[str, Path] = {}
    for gsm_id, gsm in gse.gsms.items():
        cel_url = None
        for key, values in gsm.metadata.items():
            if not key.startswith("supplementary_file"):
                continue
            cel_url = next(
                (u for u in values if u and u.strip().upper() != "NONE" and _is_cel_url(u)), None
            )
            if cel_url:
                break
        if cel_url is None:
            continue
        cel_dir.mkdir(parents=True, exist_ok=True)
        path = _download_file(cel_url, cel_dir)
        if path is not None:
            downloaded[gsm_id] = path
    return downloaded


def build_and_renormalize_expression_matrix(
    gse, out_dir: Path, platform_details: list[dict]
) -> Path | None:
    """RMA-renormalize each Affymetrix platform's raw CEL files (--rma only),
    then map probes -> genes the same way as the submitter-value path in
    build_and_map_expression_matrix. Writes nothing (and the submitter-value
    expression.tsv.gz stands alone) when there are no Affymetrix platforms, no
    CEL files to download, or RMA can't run for every platform present -- see
    renormalize.RmaUnavailable for why a given platform might be skipped.
    """
    affy_gpl_ids = {p["gpl_id"] for p in platform_details if p.get("vendor") == "affymetrix"}
    if not affy_gpl_ids:
        return None

    cel_files = download_cel_files(gse, out_dir)
    if not cel_files:
        print("    --rma requested but no CEL supplementary files found")
        return None

    probe_matrices: dict[str, pd.DataFrame] = {}
    for gpl_id in sorted(affy_gpl_ids):
        platform_cel_files = {
            gsm_id: path
            for gsm_id, path in cel_files.items()
            if gpl_id in gse.gsms[gsm_id].metadata.get("platform_id", [])
        }
        if not platform_cel_files:
            continue
        try:
            probe_matrix = renormalize.run_rma(platform_cel_files, gpl_id)
        except renormalize.RmaUnavailable as exc:
            print(f"    RMA skipped for {gpl_id}: {exc}")
            continue
        _write_matrix(probe_matrix, out_dir / f"probe_matrix_rma_{gpl_id}.tsv.gz")
        probe_matrices[gpl_id] = probe_matrix

        # CEL files are only useful up to a successful RMA run -- they're raw
        # data (hundreds of MB to GBs across a series) that's fully captured
        # by the probe matrix above, so keeping them around afterward is pure
        # disk waste.
        for cel_path in platform_cel_files.values():
            cel_path.unlink(missing_ok=True)

    cel_dir = out_dir / "cel"
    if cel_dir.is_dir() and not any(cel_dir.iterdir()):
        cel_dir.rmdir()

    if not probe_matrices:
        return None

    # Same companion-chip combination as build_and_map_expression_matrix,
    # restricted to pairs where RMA actually succeeded for both halves --
    # each platform here got its own separately-normalized probe matrix
    # (RMA runs per chip type), so the pair's two matrices are concatenated
    # column-wise first (aligning their almost-disjoint probe rows) before
    # reusing the same combine/dedup helper.
    pairings = {
        pair: pairing
        for pair, pairing in companion_platforms.detect_pairings(gse).items()
        if pair[0] in probe_matrices and pair[1] in probe_matrices
    }
    paired_gpl_ids = {gpl for pair in pairings for gpl in pair}

    gene_frames = []
    for (gpl_a, gpl_b), pairing in pairings.items():
        print(f"    combining {len(pairing)} companion-chip sample pair(s) ({gpl_a}+{gpl_b})")
        merged_probes = pd.concat([probe_matrices[gpl_a], probe_matrices[gpl_b]], axis=1)
        combined_probes = companion_platforms.combine_paired_probe_columns(merged_probes, pairing)
        gene_frames.append(
            probe_mapping.aggregate_probes_to_genes(combined_probes, _combined_probe_gene_map(gpl_a, gpl_b))
        )
    for gpl_id, probe_matrix in probe_matrices.items():
        if gpl_id in paired_gpl_ids:
            continue
        probe_gene_map = probe_mapping.get_or_build_probe_gene_map(gpl_id)
        gene_frames.append(probe_mapping.aggregate_probes_to_genes(probe_matrix, probe_gene_map))

    if not gene_frames:
        return None

    expression = gene_frames[0]
    for other in gene_frames[1:]:
        expression = expression.merge(other, on=["gene_symbol", "entrez_id"], how="outer")

    expression_path = out_dir / "expression_rma.tsv.gz"
    _write_matrix(expression, expression_path, index=False)
    return expression_path


def _cached_result(gse_id: str, out_dir: Path) -> tuple[dict, list[dict]] | None:
    """Build a download_cohort()-shaped result purely from files already on
    disk from a previous run, without re-fetching or re-parsing the GEO
    record. Returns (result, platform_details), or None if this cohort has
    never been downloaded (no annotation.tsv/series.tsv yet).
    """
    annotation_path = out_dir / "annotation.tsv"
    series_path = out_dir / "series.tsv"
    if not annotation_path.exists() or not series_path.exists():
        return None

    srow = pd.read_csv(series_path, sep="\t").iloc[0].to_dict()
    platform_details = json.loads(srow.get("platform_details") or "[]")
    assay_types = sorted({p["assay_type"] for p in platform_details})

    result: dict = {
        "gse_id": gse_id, "assay_types": assay_types,
        "expression_path": None, "annotation_path": str(annotation_path),
    }

    annotation_df = pd.read_csv(annotation_path, sep="\t", nrows=1)
    if "expression_status" in annotation_df.columns and len(annotation_df):
        result["expression_status"] = annotation_df.iloc[0]["expression_status"]

    expr_path = out_dir / "expression.tsv.gz"
    if expr_path.exists():
        result["expression_path"] = str(expr_path)

    channel_paths = {
        str(n): str(out_dir / f"channel{n}_expression.tsv.gz")
        for n in (1, 2) if (out_dir / f"channel{n}_expression.tsv.gz").exists()
    }
    if channel_paths:
        result["channel_expression_paths"] = channel_paths

    channel_roles = _load_channel_roles(out_dir)
    if channel_roles:
        result["channel_roles"] = channel_roles
        signal_path = out_dir / "channel_signal_expression.tsv.gz"
        if signal_path.exists():
            result["channel_signal_expression_path"] = str(signal_path)
        reference_path = out_dir / "channel_reference_expression.tsv.gz"
        if reference_path.exists():
            result["channel_reference_expression_path"] = str(reference_path)

    rma_path = out_dir / "expression_rma.tsv.gz"
    if rma_path.exists():
        result["expression_rma_path"] = str(rma_path)

    expr_dir = out_dir / "expression"
    if expr_dir.is_dir():
        files = sorted(expr_dir.iterdir())
        if files:
            result["expression_files"] = [str(p) for p in files]

    expression_qc = _load_expression_qc(out_dir)
    if expression_qc:
        if expression_qc.get("primary_expression_file"):
            result["primary_expression_file"] = expression_qc["primary_expression_file"]
            result["primary_expression_unit"] = expression_qc.get("primary_expression_unit")
        if expression_qc.get("notes"):
            result["expression_qc_notes"] = expression_qc["notes"]

    return result, platform_details


def _filter_supported_platforms(gse, platform_details: list[dict]) -> tuple[list[dict], list[str]]:
    """Split platform_details into (supported, human-readable rejection reasons) via
    platform_classify.platform_supported, and drop any gse.gsms/gse.gpls entries on a
    rejected platform *in place*. Every function in this module already just reads
    gse.gsms/gse.gpls, so this is the only place that needs to know about eligibility --
    everything downstream automatically only ever sees the supported subset.
    """
    supported = []
    rejected_reasons = []
    rejected_gpl_ids = set()
    for detail in platform_details:
        ok, reason = platform_classify.platform_supported(detail)
        if ok:
            supported.append(detail)
        else:
            rejected_gpl_ids.add(detail["gpl_id"])
            rejected_reasons.append(f"{detail['gpl_id']} ({reason})")

    if rejected_gpl_ids:
        gse.gsms = {
            gsm_id: gsm for gsm_id, gsm in gse.gsms.items()
            if not (rejected_gpl_ids & set(gsm.metadata.get("platform_id", [])))
        }
        gse.gpls = {gpl_id: gpl for gpl_id, gpl in gse.gpls.items() if gpl_id not in rejected_gpl_ids}

    return supported, rejected_reasons


def resolve_download_targets(gse_id: str, series_dir: Path | None = None, force: bool = False) -> list[str]:
    """Expand gse_id into the leaf series id(s) download_cohort should actually be
    called on, for the CLI's download loop.

    A series already fully downloaded under gse_id itself is returned as-is with zero
    network calls -- preserves download_cohort's own "already downloaded" cache reuse
    for the common (non-SuperSeries) case. Otherwise defers to
    geo_fetch.resolve_leaf_series_ids to detect and expand a SuperSeries into its
    subseries, so each gets its own independent download and eligibility check rather
    than being blended into one incoherent series.
    """
    if not force and _cached_result(gse_id, _series_dir(gse_id, series_dir)) is not None:
        return [gse_id]
    return geo_fetch.resolve_leaf_series_ids(gse_id)


def download_cohort(
    gse_id: str, series_dir: Path | None = None, escalate_ambiguous: bool = False, rma: bool = False,
    force: bool = False,
) -> dict:
    out_dir = _series_dir(gse_id, series_dir)

    if not force:
        cached_pair = _cached_result(gse_id, out_dir)
        if cached_pair is not None:
            cached, platform_details = cached_pair
            rma_missing = rma and "microarray" in cached["assay_types"] and "expression_rma_path" not in cached
            if not rma_missing:
                print(f"  {gse_id}: already downloaded -- reusing existing files (pass force=True / --force to redo)")
                return cached

            # Everything else about this cohort is already on disk and
            # reusable -- only the newly-requested --rma output is actually
            # missing, so fetch the series just for that (skipping the
            # clinical_annotate LLM call and probe/gene matrix rebuild
            # entirely, since annotation.tsv/expression.tsv.gz already exist).
            print(f"  {gse_id}: already downloaded -- reusing existing files, computing the missing --rma output")
            gse = geo_fetch.fetch_series(gse_id)
            rma_path = build_and_renormalize_expression_matrix(gse, out_dir, platform_details)
            cached["expression_rma_path"] = str(rma_path) if rma_path else None
            return cached

    gse = geo_fetch.fetch_series(gse_id)

    organism = annotate._series_organism(gse)
    if organism and not annotate.is_human_organism(organism):
        raise UnsupportedCohortError(f"non-human organism ({organism}) -- only Homo sapiens is supported")

    platform_details, rejected_reasons = _filter_supported_platforms(gse, annotate.platform_details(gse))
    for reason in rejected_reasons:
        print(f"  {gse_id}: skipping platform {reason}")
    if not gse.gsms:
        raise UnsupportedCohortError("; ".join(rejected_reasons) or "no supported platforms")

    srow = annotate.series_row(gse)
    samples = annotate.samples_table(gse)
    _persist_series_annotation(out_dir, srow, samples)

    # samples.tsv above stays a faithful one-row-per-GSM record of what GEO
    # actually published; annotation.tsv (built from `samples` below) is the
    # "ready to use" view, so companion-chip pairs (see companion_platforms.py)
    # collapse to one row here -- otherwise every downstream sample-level
    # analysis double-counts them.
    pairings = companion_platforms.detect_pairings(gse)
    if pairings:
        samples = companion_platforms.collapse_paired_samples(samples, pairings)

    assay_types = {p["assay_type"] for p in platform_details}

    result = {"gse_id": gse_id, "assay_types": sorted(assay_types), "expression_path": None}

    qc_notes: list[str] = []
    primary_path: Path | None = None
    primary_unit: str | None = None
    matrix_found = False

    if assay_types & {"bulk_rnaseq", "scrnaseq"}:
        files = download_rnaseq_files(gse, out_dir)
        result["expression_files"] = [str(p) for p in files]
        if not files:
            print(f"  {gse_id}: no supplementary expression files found")
        else:
            primary_path, primary_unit, rnaseq_qc_notes = check_rnaseq_expression_qc(
                files, out_dir / "expression", n_samples=len(gse.gsms)
            )
            if primary_path is not None:
                result["primary_expression_file"] = str(primary_path)
                result["primary_expression_unit"] = primary_unit
                matrix_found = True
            qc_notes.extend(rnaseq_qc_notes)
    elif "microarray" in assay_types:
        expr_path, expr_qc_notes = build_and_map_expression_matrix(gse, out_dir)
        result["expression_path"] = str(expr_path) if expr_path else None
        if expr_path is not None:
            matrix_found = True
            qc_notes.extend(f"expression.tsv.gz: {note}" for note in expr_qc_notes)
        channel_paths, channel_roles = build_and_map_channel_expression_matrices(gse, out_dir, platform_details)
        if channel_paths:
            result["channel_expression_paths"] = {str(k): str(v) for k, v in channel_paths.items()}
        if channel_roles and channel_roles.get("method") != "ambiguous":
            _write_channel_roles(out_dir, channel_roles)
            result["channel_roles"] = channel_roles
            if channel_roles.get("signal_channel") in channel_paths:
                result["channel_signal_expression_path"] = str(out_dir / "channel_signal_expression.tsv.gz")
            if channel_roles.get("reference_channel") in channel_paths:
                result["channel_reference_expression_path"] = str(out_dir / "channel_reference_expression.tsv.gz")
        if rma:
            rma_path = build_and_renormalize_expression_matrix(gse, out_dir, platform_details)
            result["expression_rma_path"] = str(rma_path) if rma_path else None
    else:
        print(f"  {gse_id}: platform assay type(s) {sorted(assay_types)} not handled yet, skipping expression download")

    if qc_notes:
        result["expression_qc_notes"] = qc_notes
        for note in qc_notes:
            print(f"  {gse_id}: expression QC: {note}")
    _write_expression_qc(out_dir, primary_path, primary_unit, qc_notes)

    expression_status = clinical_annotate.classify_expression_status(qc_notes, matrix_found)
    result["expression_status"] = expression_status
    print(f"  {gse_id}: expression status: {expression_status}")

    plan = clinical_annotate.plan_column_mapping(samples)
    annotation = clinical_annotate.apply_column_mapping(samples, plan)
    annotation["expression_status"] = expression_status
    annotation_path = out_dir / "annotation.tsv"
    annotation.to_csv(annotation_path, sep="\t", index=False)
    result["annotation_path"] = str(annotation_path)

    notes = annotation.attrs.get("clinical_annotate_notes")
    if notes:
        print(f"  {gse_id}: annotation notes: {notes}")

    return result
