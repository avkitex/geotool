"""Phase 3: unify already-downloaded cohorts' sample annotation into one
master table, keyed by gsm_id/gse_id.

Three layers, cheapest first:

1. Reuse tier (free): each cohort's own data/series/<GSE>/annotation.tsv
   (clinical_annotate.py -- treatment/response/recist/survival already
   canonicalized *within* that cohort) plus, if present,
   data/series/<GSE>/llm_annotations.json (llm_annotate.py's cache --
   tissue_class/diagnosis/sample_source/prior_therapy, from a canonical
   vocab, written whenever `geotool search --llm-annotate` looked at that
   series). Reading the cache never triggers an LLM call.
2. Gap-filling tier (LLM cost, opt-in via harmonize_cohorts(...,
   llm_annotate_flag=True)): cohorts with no cached llm_annotations.json get
   classified now, via the exact same llm_annotate.annotate_and_cache the
   search command uses -- it writes to the same cache path, so a later
   `search --llm-annotate` on the same series pays nothing extra either.
3. Alias mapping (deterministic, no LLM): whatever raw characteristic
   columns neither tier above already touched (sex, age, cell_type, ...) get
   renamed onto a canonical name via a small JSON alias file
   (geotool/vocab_data/annotation_aliases.json). Columns with no alias match
   are left as-is -- never dropped, so nothing is silently lost.

The master table is one row per gsm_id across every requested cohort;
survival columns are unioned across cohorts (e.g. cohort A's OS_time/
OS_event and cohort B's PFS_time/PFS_event both appear, NaN where a given
cohort doesn't report that type) via a plain outer concat.

Cross-cohort *expression matrix* merging/batch correction (different
platforms, different gene coverage, batch effects) is explicitly out of
scope here -- a substantially harder, separate problem.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from geotool import config, llm_annotate

_DEFAULT_ALIASES_PATH = config.PROJECT_ROOT / "geotool" / "vocab_data" / "annotation_aliases.json"


def load_annotation_aliases(path: Path | None = None) -> dict[str, str]:
    with open(path or _DEFAULT_ALIASES_PATH, encoding="utf-8") as f:
        return json.load(f)


def apply_column_aliases(df: pd.DataFrame, aliases: dict[str, str] | None = None) -> pd.DataFrame:
    """Rename raw characteristic columns not already unified by
    clinical_annotate/llm_annotate onto a canonical name (e.g. "Sex"/"sex"/
    "gender" -> "sex"). Case-insensitive match; a column with no alias, or
    whose canonical name is already taken, is left untouched -- kept, not
    dropped, so nothing is silently lost.
    """
    aliases = aliases if aliases is not None else load_annotation_aliases()
    lower_aliases = {k.lower(): v for k, v in aliases.items()}
    rename_map = {}
    for col in df.columns:
        canonical = lower_aliases.get(col.lower())
        if canonical and canonical != col and canonical not in df.columns:
            rename_map[col] = canonical
    return df.rename(columns=rename_map)


def _read_cached_llm_result(gse_id: str, series_dir: Path | None = None):
    cache_path = (series_dir or config.SERIES_DIR) / gse_id / "llm_annotations.json"
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        return llm_annotate.SeriesLLMResult.model_validate(cached["result"])
    except (json.JSONDecodeError, OSError, KeyError):
        return None


def get_llm_annotation(
    gse_id: str,
    series_row: dict,
    samples: pd.DataFrame,
    series_dir: Path | None = None,
    backfill: bool = False,
    model: str | None = None,
    escalate_ambiguous: bool = False,
) -> pd.DataFrame | None:
    """Per-sample llm_* columns for one cohort, keyed by gsm_id.

    With backfill=False (the default), this only ever reads whatever's
    already cached at data/series/<GSE>/llm_annotations.json -- never an LLM
    call -- returning None if that cache doesn't exist. With backfill=True,
    a missing/stale cache gets computed now via llm_annotate.annotate_and_cache
    (gse=None: the only thing that needs a live GEOparse object is an
    optional protocol-text prompt hint, not worth a full re-fetch here).
    """
    if backfill:
        merged, _ = llm_annotate.annotate_and_cache(
            None, series_row, samples, series_dir=series_dir, model=model, escalate_ambiguous=escalate_ambiguous
        )
    else:
        result = _read_cached_llm_result(gse_id, series_dir)
        if result is None:
            return None
        fp_ids, _groups = llm_annotate.group_fingerprints(samples)
        merged = llm_annotate.merge_annotations(samples, fp_ids, result)

    llm_cols = [c for c in merged.columns if c.startswith("llm_")]
    return merged[["gsm_id", *llm_cols]]


def harmonize_cohort(
    gse_id: str,
    series_dir: Path | None = None,
    llm_annotate_flag: bool = False,
    model: str | None = None,
    escalate_ambiguous: bool = False,
) -> pd.DataFrame | None:
    """One cohort's annotation.tsv, with llm_* columns joined on where
    available (or backfilled now if llm_annotate_flag). None if this cohort
    hasn't been downloaded yet (no annotation.tsv) -- callers should skip it
    with a warning rather than fail the whole harmonize run.
    """
    out_dir = (series_dir or config.SERIES_DIR) / gse_id
    annotation_path = out_dir / "annotation.tsv"
    if not annotation_path.exists():
        return None
    annotation = pd.read_csv(annotation_path, sep="\t")
    # Enforce the known-correct gse_id rather than trust whatever's (or isn't)
    # in the file -- found live on an older cohort whose annotation.tsv had
    # no gse_id column at all, which would otherwise show up as NaN in the
    # master table despite gse_id being exactly what we're keyed on here.
    annotation["gse_id"] = gse_id

    series_row_path, samples_path = out_dir / "series.tsv", out_dir / "samples.tsv"
    if series_row_path.exists() and samples_path.exists():
        series_row = pd.read_csv(series_row_path, sep="\t").iloc[0].to_dict()
        samples = pd.read_csv(samples_path, sep="\t")
        llm_cols = get_llm_annotation(
            gse_id, series_row, samples, series_dir=series_dir,
            backfill=llm_annotate_flag, model=model, escalate_ambiguous=escalate_ambiguous,
        )
        if llm_cols is not None and "gsm_id" in annotation.columns:
            annotation = annotation.merge(llm_cols, on="gsm_id", how="left")

    return apply_column_aliases(annotation)


def harmonize_cohorts(
    gse_ids: list[str],
    series_dir: Path | None = None,
    llm_annotate_flag: bool = False,
    model: str | None = None,
    escalate_ambiguous: bool = False,
) -> pd.DataFrame:
    """Master annotation table across every requested cohort -- one row per
    gsm_id, columns unioned across cohorts (an outer concat, so e.g. one
    cohort's OS_time/OS_event and another's PFS_time/PFS_event both survive,
    NaN where a given cohort doesn't report that column). Cohorts that
    haven't been downloaded yet are skipped with a printed warning rather
    than failing the whole run.
    """
    frames = []
    for gse_id in gse_ids:
        df = harmonize_cohort(
            gse_id, series_dir=series_dir, llm_annotate_flag=llm_annotate_flag,
            model=model, escalate_ambiguous=escalate_ambiguous,
        )
        if df is None:
            print(f"  {gse_id}: not downloaded yet (no annotation.tsv) -- skipped, run `geotool download {gse_id}` first")
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)
