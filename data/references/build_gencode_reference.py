"""Build GENCODE reference tables and a TPM-ready clean transcript/gene
mapping, starting from a raw GENCODE GTF. Runs for GRCh38 (v50) and GRCh37/
hg19 (v50lift37 -- the current v50 gene models lifted over to hg19
coordinates, not the outdated native v19).

The raw GTFs (~125-160MB) and the IntOGen driver-gene zip are third-party
downloads, not vendored into git -- see sources_manifest.json. On each run,
fetch_sources() downloads and sha256-verifies whatever's missing from
`data/references/` before the pipeline touches it, so a fresh checkout just
needs `python build_gencode_reference.py` and network access.

Pipeline per build:
  0. fetch_sources(): ensure the manifest's raw inputs are present and
     checksum-valid, downloading any that are missing.
  1. Parse the GTF once into per-gene and per-transcript tables (gene_type,
     transcript_type, GENCODE tags, CCDS membership, GENCODE support level,
     and spliced transcript length -- the sum of a transcript's exon
     widths, i.e. the mRNA length TPM's per-transcript length-normalization
     step actually divides by).
  2. Reformat the latest annotation into the same two-column (ID, Gene)
     format as the existing gencode32 files, gzip-compressed (and, for the
     GRCh38 build only, gzip-compress the existing v32 files in place).
  3. Report the transcript length distribution.
  4. Build a full transcript annotation table (nothing removed) with
     `included` / `exclusion_reason` columns, plus a filtered clean
     ENST -> ENSG -> symbol matrix for genes with >=1 included transcript.
     A transcript is included only if its gene is protein_coding, the
     transcript itself is protein_coding type, it's >=300bp, it's CCDS-
     backed, and it's neither mitochondrially-encoded nor a replication-
     dependent histone gene (both excluded for the same reason: their
     apparent TPM is highly library-prep-dependent rather than reflecting
     comparable biology -- see the mt_gene_ids / rd_histone_gene_ids
     comments in step4_clean_matrix for why).
  5. Cross-reference genes with zero included transcripts (i.e. dropped
     entirely) against the IntOGen compendium of cancer driver genes.

Usage: .venv/Scripts/python.exe data/references/build_gencode_reference.py
"""
import gzip
import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

REF_DIR = Path("data/references")
V32_DIR = REF_DIR / "gencode32"
CANCER_DIR = REF_DIR / "cancer_genes"
MANIFEST_PATH = REF_DIR / "sources_manifest.json"

MIN_TRANSCRIPT_LENGTH = 300
# GENCODE gene-level tags marking a locus as low-confidence / not a clean
# standalone protein-coding gene, independent of its transcripts' biotypes.
# NOTE: "reference_genome_error" is deliberately NOT included here -- it
# flags a GRCh38 assembly problem *at that locus* (e.g. a collapsed
# segmental duplication), not a poorly-characterized or artifactual gene
# model. It's carried by real, well-studied genes (PTEN, POLR2A, SHANK3,
# ABO, ...) whose own transcript/CDS annotation is otherwise solid, so
# treating it as an exclusion reason would have dropped PTEN -- one of the
# most important tumor-suppressor genes in cancer biology -- for the wrong
# reason. "artifactual_duplication" genes are already gene_type=artifact and
# so already excluded by the gene_type check; kept here for clarity only.
BAD_GENE_TAGS = ("readthrough_gene", "artifactual_duplication")

# Replication-dependent histone genes (Seal et al 2022, Epigenetics &
# Chromatin -- the 2021 HGNC histone renaming): their mRNAs end in a
# stem-loop instead of a poly-A tail, processed by U7 snRNP rather than the
# canonical cleavage/polyadenylation machinery, and are only made in
# S-phase. Real, essential, CCDS-backed genes -- excluded for the same
# practical reason as the mitochondrial genes (see mt_gene_ids below), not
# a quality one. Everything else in the histone family (H1-0/7/8/10,
# H2AZ1/2, H2AX, H2AJ, H2AB1-3, H2AP, H2AL3, H2BW1/2, H2BK1/N1, H3-3A/B,
# H3-4/5/7, CENPA, MACROH2A1/2, ...) is a replication-independent "variant"
# histone and is polyadenylated normally, so it's left untouched.
_RD_HISTONE_RE = re.compile(r"^(H1-[1-6]|H2AC\d+|H2BC\d+|H3C\d+|H4C\d+)$")

# Immunoglobulin (IG) and T-cell receptor (TR) gene segments -- IGH/IGK/IGL
# V/D/J/C and TRA/TRB/TRG/TRD V/D/J/C. GENCODE already types nearly all of
# these IG_*_gene/TR_*_gene (not protein_coding), so the gene_type check
# below excludes them anyway; this symbol pattern is a documented, explicit
# reason for that exclusion (rather than the generic non-coding/pseudogene
# one) and a backstop for the rare case where a segment is otherwise typed
# (e.g. TRBV11-2, protein_coding but non-CCDS). These are germline
# VDJ-recombination loci: standard bulk RNA-seq read alignment can't
# resolve which rearranged allele a read came from, so they're not usable
# gene-expression signal regardless of biotype label. Requires a trailing
# digit/hyphen or end-of-string after the V/D/J/C letter so real unrelated
# genes that merely start with "TR"+letter (TRADD, TRAF1, TRAP1, TRAK1,
# ...) don't match.
_IG_TR_RECEPTOR_RE = re.compile(r"^(IG[HKL][VDJC]|TR[ABGD][VDJC])($|[0-9-])")

_ATTR_RE = re.compile(r'(\w+) (?:"([^"]*)"|(\S+));')


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_sources():
    """Download+verify (sha256) any manifest source missing from disk. Raw
    third-party inputs aren't vendored into git -- see sources_manifest.json
    -- so a fresh checkout needs this before the GTFs/cancer-gene list exist
    locally. Already-present, checksum-valid files are left untouched.
    """
    manifest = json.loads(MANIFEST_PATH.read_text())
    for src in manifest["sources"]:
        dest = REF_DIR / src["dest"]
        if dest.exists() and _sha256(dest) == src["sha256"]:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetching {src['name']} ({src['size_bytes'] / 1e6:.0f}MB) from {src['url']}")
        urllib.request.urlretrieve(src["url"], dest)
        actual = _sha256(dest)
        if actual != src["sha256"]:
            dest.unlink()
            raise ValueError(
                f"{src['name']}: sha256 mismatch (expected {src['sha256']}, got {actual}) -- "
                "download corrupted or upstream file changed; deleted the bad copy."
            )
        if "unzip_to" in src:
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(REF_DIR / src["unzip_to"])
            print(f"extracted {src['name']} -> {src['unzip_to']}")


def parse_attrs(field):
    attrs = {}
    for key, quoted_val, bare_val in _ATTR_RE.findall(field):
        val = quoted_val if quoted_val != "" else bare_val
        attrs[key] = attrs[key] + "," + val if key in attrs else val
    return attrs


def parse_gtf(gtf_path):
    genes, transcripts, exon_len = {}, {}, {}
    with gzip.open(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            feature = fields[2]
            if feature not in ("gene", "transcript", "exon"):
                continue
            start, end = int(fields[3]), int(fields[4])
            attrs = parse_attrs(fields[8])

            if feature == "gene":
                genes[attrs["gene_id"]] = {
                    "gene_id": attrs["gene_id"],
                    "gene_symbol": attrs.get("gene_name", ""),
                    "gene_type": attrs.get("gene_type", ""),
                    "gene_level": attrs.get("level", ""),
                    "hgnc_id": attrs.get("hgnc_id", ""),
                    "gene_tags": attrs.get("tag", ""),
                    "chrom": fields[0],
                    "start": start,
                    "end": end,
                }
            elif feature == "transcript":
                transcripts[attrs["transcript_id"]] = {
                    "transcript_id": attrs["transcript_id"],
                    "gene_id": attrs["gene_id"],
                    "gene_symbol": attrs.get("gene_name", ""),
                    "gene_type": attrs.get("gene_type", ""),
                    "transcript_type": attrs.get("transcript_type", ""),
                    "transcript_level": attrs.get("level", ""),
                    "tags": attrs.get("tag", ""),
                    "ccdsid": attrs.get("ccdsid", ""),
                    "transcript_support_level": attrs.get("transcript_support_level", ""),
                    # Only present on lift37 (GRCh38->GRCh37 backmapped) GTFs.
                    "remap_status": attrs.get("remap_status", ""),
                    "remap_num_mappings": attrs.get("remap_num_mappings", ""),
                }
            elif feature == "exon":
                tid = attrs["transcript_id"]
                exon_len[tid] = exon_len.get(tid, 0) + (end - start + 1)

    genes_df = pd.DataFrame(genes.values())
    tx_df = pd.DataFrame(transcripts.values())
    tx_df["transcript_length"] = tx_df["transcript_id"].map(exon_len)

    # Drop PAR (pseudoautosomal region) duplicate loci: pre-v50 GENCODE
    # suffixed the chrY copy's gene_id with "_PAR_Y"; v50 instead just
    # mints it a wholly separate ENSG id on chrY (e.g. CRLF2/P2RY8, both
    # real leukemia genes -- their chrY duplicate has no CCDS-backed
    # transcript and would otherwise show up as a spuriously "fully
    # dropped" gene even though the chrX copy is included fine). Identify
    # duplicates by gene_symbol appearing on both chrX and chrY -- verified
    # this is exactly the ~40-gene PAR1/PAR2 set (SHOX, CSF2RA, IL3RA,
    # CRLF2, P2RY8, ...) with no coincidental unrelated X/Y symbol
    # collisions, on both GRCh38 and its GRCh37 liftover. Matching on
    # (symbol, start, end) instead -- the original approach -- silently
    # fails on the lift37 build: GENCODE's backmap tool applies a small but
    # different coordinate shift to the chrX vs chrY PAR copies, so their
    # coordinates no longer agree post-liftover even though it's still the
    # same duplicated locus.
    par_y_ids = set(genes_df[genes_df["gene_id"].str.endswith("_PAR_Y")]["gene_id"])
    chrx_symbols = set(genes_df[genes_df["chrom"] == "chrX"]["gene_symbol"])
    chry = genes_df[genes_df["chrom"] == "chrY"]
    par_y_ids |= set(chry[chry["gene_symbol"].isin(chrx_symbols)]["gene_id"])

    genes_df = genes_df[~genes_df["gene_id"].isin(par_y_ids)].copy()
    tx_df = tx_df[~tx_df["gene_id"].isin(par_y_ids)].copy()
    print(f"PAR chrY-duplicate genes dropped: {len(par_y_ids)}")
    return genes_df, tx_df


def step1_id_maps(genes_df, tx_df, out_dir, tag, compress_v32=False):
    """Reformat the latest GENCODE release into the existing gencode32
    (ID, Gene) two-column format, and (once) gzip-compress the v32 files
    in place.
    """
    ensg2hugo = genes_df[["gene_id", "gene_symbol"]].rename(
        columns={"gene_id": "ID", "gene_symbol": "Gene"}
    ).sort_values("ID")
    id2gene = tx_df[["transcript_id", "gene_symbol"]].rename(
        columns={"transcript_id": "ID", "gene_symbol": "Gene"}
    ).sort_values("ID")

    ensg2hugo.to_csv(out_dir / f"ensg2hugo_gencode_{tag}.tsv.gz", sep="\t", index=False)
    id2gene.to_csv(out_dir / f"id2gene_gencode_{tag}.tsv.gz", sep="\t", index=False)
    print(f"wrote ensg2hugo_gencode_{tag}.tsv.gz ({len(ensg2hugo)} genes)")
    print(f"wrote id2gene_gencode_{tag}.tsv.gz ({len(id2gene)} transcripts)")

    if compress_v32:
        for name in ("ensg2hugo_gencode_v32.tsv", "id2gene_gencode_v32.tsv"):
            raw = V32_DIR / name
            gz = V32_DIR / (name + ".gz")
            if raw.exists():
                with open(raw, "rb") as f_in, gzip.open(gz, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                raw.unlink()
                print(f"compressed {name} -> {gz.name}")


def step3_length_distribution(tx_df, out_dir, tag):
    lengths = tx_df["transcript_length"]
    print("\n=== transcript length distribution (all transcripts, n={}) ===".format(len(tx_df)))
    print(lengths.describe(percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99]))

    pc_tx = tx_df[(tx_df["gene_type"] == "protein_coding") & (tx_df["transcript_type"] == "protein_coding")]
    print(f"\n=== length distribution, protein_coding transcripts of protein_coding genes (n={len(pc_tx)}) ===")
    print(pc_tx["transcript_length"].describe(percentiles=[.01, .02, .05, .1, .25, .5, .75, .9, .95, .99]))
    for cutoff in (50, 100, 150, 200):
        below = (pc_tx["transcript_length"] < cutoff).sum()
        print(f"  protein_coding tx < {cutoff}bp: {below} ({below / len(pc_tx) * 100:.3f}%)")

    dist = lengths.value_counts(bins=40).sort_index()
    dist_df = pd.DataFrame({
        "bin_left": [i.left for i in dist.index],
        "bin_right": [i.right for i in dist.index],
        "count": dist.values,
    })
    dist_df.to_csv(out_dir / f"transcript_length_distribution_{tag}.tsv", sep="\t", index=False)
    print(f"wrote transcript_length_distribution_{tag}.tsv")


def step4_clean_matrix(genes_df, tx_df, out_dir, tag):
    tx = tx_df.copy()
    bad_genes = set(genes_df[genes_df["gene_tags"].fillna("").apply(
        lambda t: any(tag in t.split(",") for tag in BAD_GENE_TAGS)
    )]["gene_id"])
    # Mitochondrially-encoded genes (chrM) are excluded outright, not on a
    # quality basis (all 13 are real, essential, well-characterized genes)
    # but a practical TPM one: their transcripts are polyadenylated as a
    # degradation mark rather than for translation/export the way nuclear
    # mRNAs are, so their apparent abundance swings heavily with library
    # prep (polyA-selection vs rRNA-depletion) rather than reflecting
    # comparable biology -- not worth carrying in a cross-cohort TPM matrix.
    mt_gene_ids = set(genes_df[genes_df["chrom"] == "chrM"]["gene_id"])
    rd_histone_gene_ids = set(genes_df[genes_df["gene_symbol"].str.match(_RD_HISTONE_RE)]["gene_id"])
    ig_tr_gene_ids = set(genes_df[
        genes_df["gene_type"].str.match(r"^(IG|TR)_")
        | genes_df["gene_symbol"].str.match(_IG_TR_RECEPTOR_RE)
    ]["gene_id"])

    def classify(row):
        if row["gene_id"] in mt_gene_ids:
            return False, "mitochondrially-encoded gene: excluded (polyA marks these for degradation, not translation, so apparent expression is highly library-prep-dependent and not comparable across cohorts)"
        if row["gene_id"] in rd_histone_gene_ids:
            return False, "replication-dependent histone gene: excluded (mRNA ends in a stem-loop, not a poly-A tail, so apparent expression is highly library-prep-dependent -- captured poorly or not at all by polyA-selected RNA-seq -- and not comparable across cohorts)"
        if row["gene_id"] in ig_tr_gene_ids:
            return False, "immunoglobulin/T-cell-receptor gene segment: excluded (germline VDJ-recombination locus -- reads can't be resolved to a specific rearranged allele, not usable bulk RNA-seq expression signal)"
        if row["gene_type"] != "protein_coding":
            return False, f"gene biotype is non-coding/pseudogene: gene_type={row['gene_type']}"
        if row["gene_id"] in bad_genes:
            return False, "gene flagged in GENCODE as readthrough/artifactual-duplication"
        if row["transcript_type"] != "protein_coding":
            return False, f"transcript biotype is non-coding within a coding gene: transcript_type={row['transcript_type']}"
        if row["transcript_length"] < MIN_TRANSCRIPT_LENGTH:
            return False, f"transcript too short: {row['transcript_length']}bp < {MIN_TRANSCRIPT_LENGTH}bp minimum"
        if row["ccdsid"] == "":
            return False, "not in the consensus CDS (CCDS) set -- insufficient independent evidence of a well-characterized coding transcript"
        # lift37-only fields (blank on the native GRCh38 GTF, so this is a
        # no-op there): a GRCh37 liftover that didn't map as one clean
        # contiguous block, or mapped to more than one place in GRCh37, has
        # unreliable coordinates/uniqueness on that build specifically.
        if row["remap_status"] not in ("", "full_contig"):
            return False, f"unreliable GRCh37 liftover: remap_status={row['remap_status']}"
        if row["remap_num_mappings"] not in ("", "1"):
            return False, f"unreliable GRCh37 liftover: mapped to {row['remap_num_mappings']} disjoint locations"
        return True, ""

    results = tx.apply(classify, axis=1, result_type="expand")
    tx["included"] = results[0]
    tx["exclusion_reason"] = results[1]

    full_cols = [
        "transcript_id", "gene_id", "gene_symbol", "gene_type", "transcript_type",
        "transcript_length", "ccdsid", "transcript_level", "remap_status",
        "remap_num_mappings", "included", "exclusion_reason",
    ]
    tx[full_cols].sort_values("transcript_id").to_csv(
        out_dir / f"transcript_annotation_{tag}.tsv.gz", sep="\t", index=False
    )
    print(f"\nwrote transcript_annotation_{tag}.tsv.gz ({len(tx)} transcripts, {tx['included'].sum()} included)")

    clean = tx[tx["included"]][["transcript_id", "gene_id", "gene_symbol"]].sort_values("transcript_id")
    clean.to_csv(out_dir / f"clean_transcript_gene_symbol_{tag}.tsv.gz", sep="\t", index=False)
    n_genes = clean["gene_id"].nunique()
    print(f"wrote clean_transcript_gene_symbol_{tag}.tsv.gz ({len(clean)} transcripts, {n_genes} genes)")

    included_gene_ids = set(clean["gene_id"])
    all_gene_ids = set(genes_df["gene_id"])
    dropped_gene_ids = all_gene_ids - included_gene_ids
    dropped_genes = genes_df[genes_df["gene_id"].isin(dropped_gene_ids)][
        ["gene_id", "gene_symbol", "gene_type"]
    ].sort_values("gene_symbol")
    dropped_genes.to_csv(out_dir / f"fully_dropped_genes_{tag}.tsv.gz", sep="\t", index=False)
    print(f"wrote fully_dropped_genes_{tag}.tsv.gz ({len(dropped_genes)} genes with zero included transcripts)")

    dropped_pc = dropped_genes[dropped_genes["gene_type"] == "protein_coding"]
    print(f"  of which gene_type=protein_coding (i.e. failed transcript-type/length/CCDS bar entirely): {len(dropped_pc)}")

    return dropped_genes


def step5_cancer_crossref(dropped_genes, out_dir, tag):
    compendium_path = CANCER_DIR / "2024-06-18_IntOGen-Drivers" / "Compendium_Cancer_Genes.tsv"
    drivers = pd.read_csv(compendium_path, sep="\t")
    driver_symbols = set(drivers["SYMBOL"].unique())
    print(f"\nIntOGen compendium driver gene symbols: {len(driver_symbols)}")

    dropped_pc = dropped_genes[dropped_genes["gene_type"] == "protein_coding"].copy()
    dropped_pc["is_cancer_driver"] = dropped_pc["gene_symbol"].isin(driver_symbols)
    hits = dropped_pc[dropped_pc["is_cancer_driver"]]
    print(f"fully-dropped protein_coding genes that ARE IntOGen cancer drivers: {len(hits)}")
    if len(hits):
        print(hits[["gene_id", "gene_symbol"]].to_string(index=False))

    hits_detail = drivers[drivers["SYMBOL"].isin(hits["gene_symbol"])]
    hits_detail.to_csv(out_dir / f"dropped_genes_cancer_relevance_{tag}.tsv", sep="\t", index=False)
    print(f"wrote dropped_genes_cancer_relevance_{tag}.tsv")
    return hits


def build(gtf_path, out_dir, tag, compress_v32=False):
    print(f"\n{'=' * 20} building {tag} {'=' * 20}")
    out_dir.mkdir(parents=True, exist_ok=True)
    genes_df, tx_df = parse_gtf(gtf_path)
    print(f"genes: {len(genes_df)}, transcripts: {len(tx_df)}")

    step1_id_maps(genes_df, tx_df, out_dir, tag, compress_v32=compress_v32)
    step3_length_distribution(tx_df, out_dir, tag)
    dropped_genes = step4_clean_matrix(genes_df, tx_df, out_dir, tag)
    step5_cancer_crossref(dropped_genes, out_dir, tag)


def main():
    fetch_sources()

    v50_dir = REF_DIR / "gencode50"
    build(v50_dir / "gencode.v50.annotation.gtf.gz", v50_dir, "v50", compress_v32=True)

    hg19_dir = REF_DIR / "gencode50_hg19"
    hg19_gtf = hg19_dir / "gencode.v50lift37.annotation.gtf.gz"
    build(hg19_gtf, hg19_dir, "v50lift37")


if __name__ == "__main__":
    main()
