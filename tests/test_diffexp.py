import numpy as np
import pandas as pd
import pytest

from geotool import diffexp


def _annotation(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


class TestBuildDesignMatrix:
    def test_two_level_group_coded_against_reference(self):
        ann = _annotation(group=["control", "control", "treated", "treated"])
        design, coef = diffexp.build_design_matrix(ann, "group")
        assert coef == "group[treated]"
        assert list(design.columns) == ["Intercept", "group[treated]"]
        assert design[coef].tolist() == [0.0, 0.0, 1.0, 1.0]
        assert (design["Intercept"] == 1.0).all()

    def test_explicit_reference_level_flips_coding(self):
        ann = _annotation(group=["control", "treated"])
        design, coef = diffexp.build_design_matrix(ann, "group", reference_level="treated")
        assert coef == "group[control]"
        assert design[coef].tolist() == [1.0, 0.0]

    def test_more_than_two_levels_raises(self):
        ann = _annotation(group=["a", "b", "c"])
        with pytest.raises(ValueError, match="exactly two"):
            diffexp.build_design_matrix(ann, "group")

    def test_missing_group_value_raises(self):
        ann = _annotation(group=["a", None, "b"])
        with pytest.raises(ValueError, match="missing values"):
            diffexp.build_design_matrix(ann, "group")

    def test_constant_covariate_is_dropped(self):
        ann = _annotation(
            group=["a", "a", "b", "b"],
            tissue=["lung", "lung", "lung", "lung"],
        )
        design, _coef = diffexp.build_design_matrix(ann, "group", covariate_cols=["tissue"])
        assert not any(c.startswith("tissue") for c in design.columns)

    def test_covariate_absent_from_this_cohort_is_silently_skipped(self):
        # per-cohort annotation.tsv files only carry whatever columns were
        # populated for that GSE -- passing a generic candidate list that
        # includes one this cohort never had should behave like "constant",
        # not raise.
        ann = _annotation(group=["a", "a", "b", "b"])
        design, coef = diffexp.build_design_matrix(ann, "group", covariate_cols=["tissue"])
        assert list(design.columns) == ["Intercept", coef]

    def test_varying_covariate_is_added(self):
        ann = _annotation(
            group=["a", "a", "b", "b"],
            replicate=["1", "2", "1", "2"],
        )
        design, _coef = diffexp.build_design_matrix(ann, "group", covariate_cols=["replicate"])
        assert any(c.startswith("replicate") for c in design.columns)

    def test_covariate_collinear_with_group_is_dropped_with_warning(self):
        # batch is 1:1 with group -- can't be told apart from it
        ann = _annotation(
            group=["a", "a", "b", "b"],
            batch=["batch1", "batch1", "batch2", "batch2"],
        )
        with pytest.warns(UserWarning, match="collinear"):
            design, _coef = diffexp.build_design_matrix(ann, "group", covariate_cols=["batch"])
        assert not any(c.startswith("batch") for c in design.columns)

    def test_covariate_missing_values_filled_with_mode(self):
        ann = _annotation(
            group=["a", "a", "b", "b"],
            tissue=["lung", None, "lung", "breast"],
        )
        with pytest.warns(UserWarning, match="missing values"):
            design, _coef = diffexp.build_design_matrix(ann, "group", covariate_cols=["tissue"])
        assert design.isna().sum().sum() == 0


class TestModeratedTtest:
    def test_recovers_known_differentially_expressed_genes(self):
        rng = np.random.default_rng(0)
        n_genes, n_de = 300, 20
        n_per_group = 6
        samples = [f"s{i}" for i in range(2 * n_per_group)]
        group = ["control"] * n_per_group + ["treated"] * n_per_group

        baseline = rng.normal(6, 1, size=n_genes)
        expr = np.tile(baseline, (2 * n_per_group, 1)).T
        expr = expr + rng.normal(0, 0.3, size=expr.shape)

        true_logfc = np.zeros(n_genes)
        true_logfc[:n_de] = rng.choice([-1, 1], size=n_de) * rng.uniform(1.5, 3.0, size=n_de)
        for j, g in enumerate(group):
            if g == "treated":
                expr[:, j] += true_logfc

        expression = pd.DataFrame(expr, index=[f"GENE{i}" for i in range(n_genes)], columns=samples)
        annotation = _annotation(sample=samples, group=group)

        result = diffexp.two_group_diffexp(expression, annotation, "sample", "group")

        true_de_genes = set(f"GENE{i}" for i in range(n_de))
        top_genes = set(result.head(n_de).index)
        assert len(top_genes & true_de_genes) >= n_de - 2  # allow a couple of misses

        null_genes = result.loc[~result.index.isin(true_de_genes)]
        assert (null_genes["adj.P.Val"] < 0.05).mean() < 0.2  # FDR roughly controlled

    def test_handles_zero_variance_genes_without_nan_or_inf(self):
        # an unexpressed gene (identical value -- e.g. log2(0+1)=0 -- in
        # every sample of both groups) has sigma2 == 0 exactly; this must
        # not blow up the fit or poison other genes' results.
        samples = [f"s{i}" for i in range(6)]
        group = ["control"] * 3 + ["treated"] * 3
        expression = pd.DataFrame(
            {
                "ZERO_VAR": [0.0] * 6,
                "NORMAL": [5.0, 5.1, 4.9, 7.0, 7.1, 6.9],
            },
            index=samples,
        ).T
        annotation = _annotation(sample=samples, group=group)
        result = diffexp.two_group_diffexp(expression, annotation, "sample", "group")
        assert np.isfinite(result["t"]).all()
        assert np.isfinite(result["P.Value"]).all()

    def test_group_coef_must_be_in_design(self):
        expression = pd.DataFrame({"s1": [1.0], "s2": [2.0]}, index=["g1"])
        design = pd.DataFrame({"Intercept": [1.0, 1.0]}, index=["s1", "s2"])
        with pytest.raises(ValueError, match="not in design"):
            diffexp.moderated_ttest(expression, design, "missing_coef")

    def test_controlling_for_confounder_reduces_residual_variance(self):
        # 2x2 balanced design: group x batch, batch has a large uniform
        # offset independent of group -- ignoring it should inflate the
        # residual variance relative to modeling it explicitly.
        rng = np.random.default_rng(1)
        n_genes = 100
        samples = [f"s{i}" for i in range(8)]
        group = ["control", "control", "treated", "treated"] * 2
        batch = ["b1", "b1", "b1", "b1", "b2", "b2", "b2", "b2"]

        baseline = rng.normal(5, 1, size=n_genes)
        expr = np.tile(baseline, (8, 1)).T + rng.normal(0, 0.2, size=(n_genes, 8))
        batch_effect = 4.0
        for j, b in enumerate(batch):
            if b == "b2":
                expr[:, j] += batch_effect

        expression = pd.DataFrame(expr, index=[f"GENE{i}" for i in range(n_genes)], columns=samples)
        annotation = _annotation(sample=samples, group=group, batch=batch)

        naive = diffexp.two_group_diffexp(expression, annotation, "sample", "group")
        controlled = diffexp.two_group_diffexp(
            expression, annotation, "sample", "group", covariate_cols=["batch"]
        )

        # same gene order isn't guaranteed (sorted by P.Value) -- compare aligned
        assert controlled.loc[naive.index, "t"].abs().median() >= naive["t"].abs().median()


class TestTwoGroupDiffexp:
    def test_missing_annotation_row_raises(self):
        expression = pd.DataFrame({"s1": [1.0], "s2": [2.0], "s3": [3.0]}, index=["g1"])
        annotation = _annotation(sample=["s1", "s2"], group=["a", "b"])
        with pytest.raises(ValueError, match="no matching annotation row"):
            diffexp.two_group_diffexp(expression, annotation, "sample", "group")

    def test_extra_annotation_rows_for_other_cohorts_are_ignored(self):
        expression = pd.DataFrame(
            {"s1": [1.0, 2.0], "s2": [1.2, 2.2], "s3": [5.0, 6.0], "s4": [5.2, 6.2]},
            index=["g1", "g2"],
        )
        annotation = _annotation(
            sample=["s1", "s2", "s3", "s4", "other_cohort_sample"],
            group=["a", "a", "b", "b", "a"],
        )
        result = diffexp.two_group_diffexp(expression, annotation, "sample", "group")
        assert set(result.index) == {"g1", "g2"}


class TestPriorVarianceHelpers:
    def test_trigamma_inverse_round_trips(self):
        from scipy import special

        for target in (0.01, 0.1, 1.0, 5.0):
            y = diffexp._trigamma_inverse(target)
            assert special.polygamma(1, y) == pytest.approx(target, rel=1e-4)

    def test_fit_prior_variance_floors_zero_variances_instead_of_dropping_them(self):
        # real RNA-seq matrices routinely have genes with sigma2 == 0 exactly
        # (e.g. a gene reading 0 TPM in every replicate of both groups).
        # limma floors these to a small fraction of the median rather than
        # excluding them from the prior fit -- dropping them instead (an
        # earlier version of this function did) biases s0^2 upward by
        # excluding precisely the lowest-variance half of genes.
        rng = np.random.default_rng(3)
        df_residual = 4
        nonzero = rng.chisquare(df_residual, size=200) / df_residual * 0.01
        sigma2 = np.concatenate([nonzero, np.zeros(200)])
        with pytest.warns(UserWarning, match="zero residual variances"):
            d0, s0_sq = diffexp._fit_prior_variance(sigma2, df_residual)
        assert np.isfinite(d0)
        assert s0_sq < np.median(nonzero) / 100  # pulled down hard by the floored zeros

    def test_fit_prior_variance_warns_when_more_than_half_are_zero(self):
        sigma2 = np.concatenate([np.array([1.0, 2.0, 3.0]), np.zeros(10)])
        with pytest.warns(UserWarning, match="more than half"):
            d0, s0_sq = diffexp._fit_prior_variance(sigma2, df_residual=4)
        assert np.isfinite(s0_sq)

    def test_fit_prior_variance_shrinks_toward_common_value_when_genes_agree(self):
        rng = np.random.default_rng(2)
        # all genes share the same true variance -- observed spread should
        # be explainable by sampling noise alone, so d0 = inf.
        df_residual = 6
        sigma2 = rng.chisquare(df_residual, size=200) / df_residual * 2.0
        d0, s0_sq = diffexp._fit_prior_variance(sigma2, df_residual)
        assert s0_sq == pytest.approx(2.0, rel=0.3)
