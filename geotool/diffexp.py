"""limma-style two-group differential expression with covariate control.

Bioconductor's limma isn't available in this environment (no R, no rpy2),
so this reimplements its core algorithm directly in Python: an ordinary
linear model fit per gene against a design matrix (Smyth 2004's lmFit),
followed by empirical-Bayes shrinkage of the per-gene residual variances
toward a common prior before testing (squeezeVar / eBayes). Design-matrix
construction and the moment-based prior-variance estimator follow the same
formulas as limma's own fitFDist/squeezeVar.

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
from scipy import special, stats
from scipy.optimize import brentq


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


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n)
    out[order] = adjusted
    return out


def _trigamma_inverse(x: float) -> float:
    """Solve trigamma(y) = x for y > 0. trigamma (scipy's polygamma(1, ·))
    is strictly decreasing on (0, inf), so a bracketing root-finder is both
    simple and robust here -- this is only ever called once per analysis
    (fitting the prior variance), so its cost is irrelevant.
    """
    if x <= 0:
        raise ValueError("trigamma inverse is undefined for x <= 0")
    lo, hi = 1e-8, 1e6
    while special.polygamma(1, hi) > x:
        hi *= 10
    return brentq(lambda y: special.polygamma(1, y) - x, lo, hi)


def _fit_prior_variance(sigma2: np.ndarray, df_residual: float) -> tuple[float, float]:
    """Method-of-moments estimate of the (d0, s0^2) scaled-inverse-chi-square
    prior on the per-gene residual variances -- limma's squeezeVar/fitFDist,
    specialized to the common case here where every gene shares the same
    residual df (one design matrix fit to every gene).

    Returns d0 = inf when the observed spread of log(sigma2) across genes is
    no larger than sampling noise alone predicts: there's no evidence of a
    finite prior to shrink toward, so every gene's posterior variance is
    just the common prior mean s0^2.
    """
    sigma2 = sigma2[sigma2 > 0]
    z = np.log(sigma2)
    e = special.digamma(df_residual / 2) - np.log(df_residual / 2)
    s0_sq = np.exp(np.mean(z) - e)

    sampling_var = special.polygamma(1, df_residual / 2)  # trigamma(df/2)
    excess_var = np.var(z, ddof=0) - sampling_var
    if excess_var <= 0:
        return np.inf, s0_sq
    d0 = 2 * _trigamma_inverse(excess_var)
    return d0, s0_sq


def moderated_ttest(expression: pd.DataFrame, design: pd.DataFrame, coef: str) -> pd.DataFrame:
    """Fit `expression = design @ beta + eps` independently per gene (row),
    then apply empirical-Bayes moderation (Smyth 2004) to shrink the
    per-gene residual variances toward a common prior before testing
    `coef`. Returns a limma topTable-style frame (logFC, AveExpr, t,
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

    samples = design.index
    expr = expression.loc[:, samples]

    X = design.to_numpy(dtype=float)
    Y = expr.to_numpy(dtype=float)  # genes x samples
    n, p = X.shape
    df_residual = n - np.linalg.matrix_rank(X)
    if df_residual < 1:
        raise ValueError("design matrix has no residual degrees of freedom -- too many covariates for the number of samples")

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = (XtX_inv @ X.T @ Y.T).T  # genes x p
    resid = Y - beta @ X.T
    sigma2 = np.sum(resid ** 2, axis=1) / df_residual

    coef_idx = list(design.columns).index(coef)
    se_unscaled = np.sqrt(XtX_inv[coef_idx, coef_idx])

    d0, s0_sq = _fit_prior_variance(sigma2, df_residual)
    if np.isinf(d0):
        sigma2_post = np.full_like(sigma2, s0_sq)
        df_post = np.inf
    else:
        sigma2_post = (d0 * s0_sq + df_residual * sigma2) / (d0 + df_residual)
        df_post = d0 + df_residual

    se = se_unscaled * np.sqrt(sigma2_post)
    t_stat = beta[:, coef_idx] / se
    if np.isinf(df_post):
        p_value = 2 * stats.norm.sf(np.abs(t_stat))
    else:
        p_value = 2 * stats.t.sf(np.abs(t_stat), df_post)

    result = pd.DataFrame(
        {
            "logFC": beta[:, coef_idx],
            "AveExpr": Y.mean(axis=1),
            "t": t_stat,
            "P.Value": p_value,
        },
        index=expression.index,
    )
    result["adj.P.Val"] = _bh_adjust(result["P.Value"].to_numpy())
    return result.sort_values("P.Value")


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
