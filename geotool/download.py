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

import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from geotool import annotate, clinical_annotate, config, geo_fetch, platform_classify, probe_mapping, renormalize

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


def select_primary_expression_file(paths: list[Path]) -> tuple[Path, str] | None:
    """Among downloaded RNA-seq supplementary files, pick the one that looks
    like the actual gene-expression quantification matrix, by
    _QUANT_UNIT_PRIORITY -- (path, unit), or None if no filename carries a
    recognizable unit keyword at all (nothing is guessed at that point,
    rather than risk QC-checking an unrelated file as if it were expression
    data). Files named after an individual GSM accession (_GSM_NAME_RE) or
    that aren't a plausible data file at all (_is_data_file) are never
    candidates.
    """
    best: tuple[int, Path, str] | None = None
    for path in paths:
        if _GSM_NAME_RE.search(path.name) or not _is_data_file(path):
            continue
        name = path.name.lower()
        for rank, unit in enumerate(_QUANT_UNIT_PRIORITY):
            if unit in name:
                if best is None or rank < best[0]:
                    best = (rank, path, unit)
                break
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


def check_rnaseq_expression_qc(paths: list[Path]) -> tuple[Path | None, str | None, list[str]]:
    """select_primary_expression_file + probe_mapping.check_expression_qc on
    whichever supplementary file that picks, for the common (one series- or
    sample-level matrix per series) case. Returns (primary_path, unit,
    qc_notes) -- primary_path/unit are None if nothing recognizable was
    found; qc_notes is empty if the file couldn't be parsed either (noted as
    its own entry) or nothing stood out.
    """
    picked = select_primary_expression_file(paths)
    if picked is None:
        return None, None, []
    primary_path, unit = picked

    matrix = _load_expression_file_for_qc(primary_path)
    if matrix is None:
        return primary_path, unit, [f"{primary_path.name}: could not parse for QC"]

    notes = probe_mapping.check_expression_qc(matrix)
    return primary_path, unit, [f"{primary_path.name}: {note}" for note in notes]


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

    # Usually one platform per series; handle the rare multi-platform case by
    # mapping+aggregating each platform's samples separately, then combining.
    platform_ids = sorted({p for gsm in gse.gsms.values() for p in gsm.metadata.get("platform_id", [])})
    gene_frames = []
    for gpl_id in platform_ids:
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

    gene_frames = []
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
        probe_gene_map = probe_mapping.get_or_build_probe_gene_map(gpl_id)
        gene_frames.append(probe_mapping.aggregate_probes_to_genes(probe_matrix, probe_gene_map))

        # CEL files are only useful up to a successful RMA run -- they're raw
        # data (hundreds of MB to GBs across a series) that's fully captured
        # by the probe matrix above, so keeping them around afterward is pure
        # disk waste.
        for cel_path in platform_cel_files.values():
            cel_path.unlink(missing_ok=True)

    cel_dir = out_dir / "cel"
    if cel_dir.is_dir() and not any(cel_dir.iterdir()):
        cel_dir.rmdir()

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
            primary_path, primary_unit, rnaseq_qc_notes = check_rnaseq_expression_qc(files)
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
