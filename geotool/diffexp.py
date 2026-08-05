"""limma-style two-group differential expression with covariate control.

The actual model fit (Smyth 2004's lmFit + empirical-Bayes moderation of
the per-gene residual variances, i.e. squeezeVar/eBayes) is delegated to
the vendored InMoose port of limma (geotool/_vendor/inmoose -- see the
README there for why it's vendored rather than a normal pip dependency).
InMoose is validated against real R limma (Pearson r = 1.0, differences at
floating-point noise level); this module previously reimplemented the
algorithm by hand, which was also checked against real R limma directly
and matched after fixing two bugs (see git history) -- InMoose is used now
because it's the actual upstream implementation, not a second
reimplementation to keep in sync with it.

Intended use: one cohort (one expression matrix, on an additive scale like
log2(TPM + 1)) at a time, comparing exactly two levels of some group column
(e.g. treatment vs. control) while optionally controlling for covariates
that vary within that cohort -- replicate, tissue/cell type, cancer type,
therapy identity, etc. Both group levels are assumed present in the cohort;
call two_group_diffexp separately per cohort rather than pooling matrices
across platforms.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._vendor.inmoose.limma import eBayes, lmFit, topTable


def build_design_matrix(
    annotation: pd.DataFrame,
    group_col: str,
    covariate_cols: list[str] | None = None,
    reference_level: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Build a numeric design matrix: an intercept, one 0/1 column for
    `group_col` (1 for the non-reference level), and one-hot dummy columns
    (first level dropped) for each of `covariate_cols` that actually varies
    within `annotation` and isn't collinear with the columns already in the
    design.

    Per-cohort annotation.tsv files only carry whichever columns were
    actually populated for that GSE, not a fixed schema, so `covariate_cols`
    is meant to be passed as a generic candidate list (e.g. "replicate",
    "tumor type", "cell_type", "batch") across very different cohorts: a
    covariate absent from `annotation` entirely, or constant within it (e.g.
    every sample is the same tissue), has nothing to control for and is
    silently dropped rather than erroring or adding a useless all-zero
    column. A covariate collinear with the design so far (most commonly:
    perfectly confounded with the group, like every replicate ID mapping 1:1
    to a group) can't be estimated alongside it and is dropped with a
    warning instead of silently producing a singular matrix.

    Returns (design, group_coef) where group_coef is the design column name
    to pass to moderated_ttest.
    """
    if group_col not in annotation.columns:
        raise ValueError(f"{group_col!r} not found in annotation columns")
    group = annotation[group_col]
    if group.isna().any():
        raise ValueError(f"{group_col!r} has missing values -- every sample must be assigned to a group")
    levels = sorted(group.unique())
    if len(levels) != 2:
        raise ValueError(f"{group_col!r} must have exactly two distinct levels, found {levels}")

    ref = reference_level if reference_level is not None else levels[0]
    if ref not in levels:
        raise ValueError(f"reference_level {ref!r} is not one of {levels}")
    other = levels[0] if levels[1] == ref else levels[1]
    group_coef = f"{group_col}[{other}]"

    design = pd.DataFrame(
        {"Intercept": 1.0, group_coef: (group == other).astype(float)},
        index=annotation.index,
    )

    for col in covariate_cols or []:
        if col not in annotation.columns:
            continue  # not populated for this cohort -- nothing to control for
        values = annotation[col]
        non_null = values.dropna()
        if non_null.nunique() <= 1:
            continue  # constant (or entirely missing) here -- nothing to control for
        if values.isna().any():
            warnings.warn(f"covariate {col!r} has missing values -- filling with its most common level")
            values = values.fillna(non_null.mode().iat[0])
        dummies = pd.get_dummies(values, prefix=col, drop_first=True, dtype=float)
        candidate = pd.concat([design, dummies], axis=1)
        if np.linalg.matrix_rank(candidate.to_numpy()) < candidate.shape[1]:
            warnings.warn(
                f"covariate {col!r} is collinear with the design so far (e.g. confounded "
                "with the group) -- can't be controlled for here, dropped"
            )
            continue
        design = candidate

    return design, group_coef


def moderated_ttest(expression: pd.DataFrame, design: pd.DataFrame, coef: str) -> pd.DataFrame:
    """Fit `expression = design @ beta + eps` independently per gene (row)
    via InMoose's lmFit, then apply its eBayes empirical-Bayes moderation to
    shrink the per-gene residual variances toward a common prior before
    testing `coef`. Returns a limma topTable-style frame (logFC, AveExpr, t,
    P.Value, adj.P.Val), sorted by P.Value.

    `expression` must already be on an additive scale (e.g. log2(TPM + 1)),
    not raw counts -- summing/differencing in linear space assumes fold
    changes are additive in this scale, the same assumption limma makes for
    microarray/log-CPM data. `design`'s index must be a subset of
    `expression`'s columns (see two_group_diffexp for the usual entry
    point).
    """
    if coef not in design.columns:
        raise ValueError(f"{coef!r} not in design matrix columns {list(design.columns)}")
    missing = set(design.index) - set(expression.columns)
    if missing:
        raise ValueError(f"{len(missing)} sample(s) in design have no matching expression column: {sorted(missing)[:5]}")
    df_residual = len(design.index) - np.linalg.matrix_rank(design.to_numpy(dtype=float))
    if df_residual < 1:
        raise ValueError("design matrix has no residual degrees of freedom -- too many covariates for the number of samples")

    expr = expression.loc[:, design.index]
    fit = eBayes(lmFit(expr, design))

    # lmFit wraps a bare DataFrame in patsy.DesignMatrix() without attaching
    # column-name metadata, so fit.coefficients ends up labeled "column0",
    # "column1", ... in design-column order rather than keeping `design`'s
    # own names -- look the real coefficient up by position instead.
    coef_label = fit.coefficients.columns[list(design.columns).index(coef)]
    top = topTable(fit, coef=coef_label, number=expr.shape[0], sort_by="P", adjust_method="fdr_bh")

    return pd.DataFrame(
        {
            "logFC": top["log2FoldChange"],
            "AveExpr": top["AveExpr"],
            "t": top["stat"],
            "P.Value": top["pvalue"],
            "adj.P.Val": top["adj_pvalue"],
        },
        index=top.index,
    )


def two_group_diffexp(
    expression: pd.DataFrame,
    annotation: pd.DataFrame,
    sample_id_col: str,
    group_col: str,
    covariate_cols: list[str] | None = None,
    reference_level: str | None = None,
) -> pd.DataFrame:
    """End-to-end limma-style two-group differential expression for a single
    cohort: build the design matrix from `annotation` (controlling for
    whichever of `covariate_cols` -- e.g. replicate, tissue/cell type,
    cancer type, therapy identity -- actually vary here and aren't
    collinear with the group) and fit the moderated-t model against
    `expression`.

    `annotation` must have one row per sample with `sample_id_col` matching
    `expression`'s column labels; rows for other cohorts/platforms are
    ignored. Both levels of `group_col` are assumed present among these
    samples -- run this once per cohort, not on a matrix pooled across
    cohorts/platforms.
    """
    ann = annotation.set_index(sample_id_col)
    ann = ann.loc[ann.index.isin(expression.columns)]
    missing = set(expression.columns) - set(ann.index)
    if missing:
        raise ValueError(f"{len(missing)} expression column(s) have no matching annotation row: {sorted(missing)[:5]}")

    design, coef = build_design_matrix(ann, group_col, covariate_cols, reference_level)
    return moderated_ttest(expression, design, coef)
