"""Refresh data/mtap_prmt5_cohorts/<GSE> for every selected cohort from the
freshly re-downloaded data/series/<GSE> -- full replace (stale copies removed
first) so the collection root never mixes an old and a new download.

data/series/<GSE> never carries the derived, gene-symbol-mapped
expression_clean.tsv.gz (filter_renormalize_rnaseq_cohorts.py writes that
straight into data/mtap_prmt5_cohorts/<GSE>, from geotool.gene_symbol_mapping
-- see that script's own docstring) -- so a plain rmtree+copytree here used
to silently delete it on every rebuild, regressing every affected cohort back
to raw ENSG identifiers with nothing pointing that out (live incident: 2026-08-08,
GSE241402 and 7 other cohorts). Regenerating it here, right after the copy,
keeps it always present and always fresh against whatever is now on disk in
data/series, rather than either missing it or resurrecting a stale copy from
before the re-download.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import filter_renormalize_rnaseq_cohorts  # noqa: E402
import prmt5_common  # noqa: E402

SERIES_ROOT = Path("data/series")
COLLECTION_ROOT = Path("data/mtap_prmt5_cohorts")


def main():
    gse_ids, _selected_ids, _parent_of = prmt5_common.effective_cohort_ids()
    copied, missing = [], []
    for gse_id in gse_ids:
        src = SERIES_ROOT / gse_id
        dst = COLLECTION_ROOT / gse_id
        if not src.exists():
            missing.append(gse_id)
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        copied.append(gse_id)
    print(f"copied {len(copied)} cohort(s) into {COLLECTION_ROOT}")
    if missing:
        print(f"missing (no data/series/<GSE> dir, download must have failed): {', '.join(missing)}")

    print("\nregenerating gene-symbol-mapped expression_clean.tsv.gz for the refreshed collection...")
    filter_renormalize_rnaseq_cohorts.main()


if __name__ == "__main__":
    main()
