"""Regression tests for combining strata with singleton PSUs (SurveyGLM).

Expected coefficients come from an independent weighted fit with statsmodels
and the expected sandwich meat is aggregated by hand over the *real* survey
units: a PSU is identified by the (original stratum, psu) pair, so two PSUs
sharing a label in different strata remain distinct units after combining.
No expectation below is produced by calling samplics itself.
"""

import numpy as np
import pytest
import statsmodels.api as sm

from samplics import ModelType
from samplics.regression import SurveyGLM
from samplics.utils.types import SinglePSUEst

# Strata A and B hold one PSU each and both PSUs happen to be labelled 1;
# stratum C holds two PSUs. A and B are combined into the new stratum D.
Y = np.array([2.0, 4.0, 1.0, 3.0, 5.0, 7.0, 8.0, 6.0])
X = np.column_stack([np.ones(8), np.array([1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])])
WGT = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0])
STRATUM = np.array(["A", "A", "B", "B", "C", "C", "C", "C"])
PSU_COLLIDING = np.array([1, 1, 1, 1, 1, 1, 2, 2])  # (A,1), (B,1), (C,1), (C,2)
PSU_DISTINCT = np.array([1, 1, 2, 2, 3, 3, 4, 4])  # collision-free control
COMB = {"A": "D", "B": "D"}

# real units after combining: stratum D -> rows [0,1] and [2,3] ; stratum C ->
# rows [4,5] and [6,7]
FINAL_GROUPS = {"D": [np.array([0, 1]), np.array([2, 3])], "C": [np.array([4, 5]), np.array([6, 7])]}


def independent_sandwich(y, x, w, family):
    """Independent weighted GLM fit + design-aggregated score sandwich."""
    fit = sm.GLM(y, x, freq_weights=w, family=family).fit()
    e = (w * fit.resid_response)[:, None] * x / fit.scale
    meat = np.zeros((x.shape[1], x.shape[1]))
    for psus in FINAL_GROUPS.values():
        sums = np.vstack([e[idx].sum(axis=0) for idx in psus])
        n_h = sums.shape[0]
        centred = sums - sums.mean(axis=0)
        meat += (n_h / (n_h - 1)) * centred.T @ centred
    bread = fit.cov_params()
    return fit.params, bread @ meat @ bread


def test_glm_linear_combine_colliding_psu_labels():
    glm = SurveyGLM(model=ModelType.LINEAR)
    glm.estimate(
        y=Y,
        x=X,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    expected_beta, expected_cov = independent_sandwich(Y, X, WGT, sm.families.Gaussian())
    assert np.allclose(glm.beta["point_est"], expected_beta)
    assert np.allclose(glm.beta_cov, expected_cov)
    assert np.allclose(glm.beta["stderror"], np.sqrt(np.diag(expected_cov)))
    # the fit used every row: combining changed the variance design only
    assert glm.sample_size == 8


def test_glm_linear_combine_matches_collision_free_control():
    glm = SurveyGLM(model=ModelType.LINEAR)
    glm.estimate(
        y=Y,
        x=X,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    control = SurveyGLM(model=ModelType.LINEAR)
    control.estimate(
        y=Y,
        x=X,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_DISTINCT,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    assert np.allclose(glm.beta["point_est"], control.beta["point_est"])
    assert np.allclose(glm.beta_cov, control.beta_cov)
    assert np.allclose(glm.beta["stderror"], control.beta["stderror"])
    assert np.allclose(glm.beta["lower_ci"], control.beta["lower_ci"])
    assert np.allclose(glm.beta["upper_ci"], control.beta["upper_ci"])
    assert glm.sample_size == control.sample_size


def test_glm_logistic_combine_colliding_psu_labels():
    y_bin = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    glm = SurveyGLM(model=ModelType.LOGISTIC)
    glm.estimate(
        y=y_bin,
        x=X,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    expected_beta, expected_cov = independent_sandwich(Y, X, WGT, sm.families.Binomial())
    expected_beta_bin, expected_cov_bin = independent_sandwich(
        y_bin, X, WGT, sm.families.Binomial()
    )
    assert np.allclose(glm.beta["point_est"], expected_beta_bin)
    assert np.allclose(glm.beta_cov, expected_cov_bin)
    assert np.allclose(glm.beta["stderror"], np.sqrt(np.diag(expected_cov_bin)))
    assert np.allclose(glm.odds_ratio["point_est"], np.exp(expected_beta_bin))
    assert glm.sample_size == 8


def test_glm_combine_into_new_label_no_keyerror():
    # regression test: combining into a brand new stratum label used to raise
    # KeyError because the fpc dictionary was keyed by the pre-combine strata
    glm = SurveyGLM(model=ModelType.LINEAR)
    glm.estimate(
        y=Y,
        x=X,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb={"A": "BRAND_NEW", "B": "BRAND_NEW"},
    )
    expected_beta, _ = independent_sandwich(Y, X, WGT, sm.families.Gaussian())
    assert np.allclose(glm.beta["point_est"], expected_beta)


def test_glm_combine_into_existing_stratum():
    # merge singleton A into C (which has two PSUs); B is skipped... no: B is
    # also a singleton, so handle it with skip in the dict form
    glm = SurveyGLM(model=ModelType.LINEAR)
    glm.estimate(
        y=Y,
        x=X,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu={"A": SinglePSUEst.combine, "B": SinglePSUEst.skip},
        strata_comb={"A": "C"},
    )
    # B rows are dropped from the fit (existing skip semantics)
    assert glm.sample_size == 6


def test_glm_combine_still_single_psu_is_rejected():
    # combining A alone into the new stratum D leaves D with one real PSU:
    # must be refused, not silently turned into one PSU per observation
    glm = SurveyGLM(model=ModelType.LINEAR)
    with pytest.raises(ValueError, match="Only one PSU"):
        glm.estimate(
            y=Y,
            x=X,
            samp_weight=WGT,
            stratum=STRATUM,
            psu=PSU_COLLIDING,
            single_psu=SinglePSUEst.combine,
            strata_comb={"A": "D"},
        )


def test_glm_combine_invalid_mapping_raises_valueerror():
    for bad_mapping in (None, {}, {"A": "D", "ZZZ": "D"}, {"A": 9, "B": 9}):
        glm = SurveyGLM(model=ModelType.LINEAR)
        with pytest.raises(ValueError):
            glm.estimate(
                y=Y,
                x=X,
                samp_weight=WGT,
                stratum=STRATUM,
                psu=PSU_COLLIDING,
                single_psu=SinglePSUEst.combine,
                strata_comb=bad_mapping,
            )


def test_glm_combine_fpc_dict_keyed_by_final_strata():
    fpc = {"C": 0.9, "D": 0.8}  # final strata, insertion order must not matter
    glm1 = SurveyGLM(model=ModelType.LINEAR)
    glm1.estimate(
        y=Y,
        x=X,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
        fpc=fpc,
    )
    glm2 = SurveyGLM(model=ModelType.LINEAR)
    glm2.estimate(
        y=Y,
        x=X,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
        fpc={"D": 0.8, "C": 0.9},
    )
    assert np.allclose(glm1.beta_cov, glm2.beta_cov)
    # scalar fpc must scale every final stratum the same way
    glm3 = SurveyGLM(model=ModelType.LINEAR)
    glm3.estimate(
        y=Y,
        x=X,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
        fpc=0.5,
    )
    glm4 = SurveyGLM(model=ModelType.LINEAR)
    glm4.estimate(
        y=Y,
        x=X,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
        fpc={"C": 0.5, "D": 0.5},
    )
    assert np.allclose(glm3.beta_cov, glm4.beta_cov)


def test_glm_combine_fpc_must_not_reference_disappeared_strata():
    glm = SurveyGLM(model=ModelType.LINEAR)
    with pytest.raises(ValueError, match="final strata"):
        glm.estimate(
            y=Y,
            x=X,
            samp_weight=WGT,
            stratum=STRATUM,
            psu=PSU_COLLIDING,
            single_psu=SinglePSUEst.combine,
            strata_comb=COMB,
            fpc={"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6},  # A and B are gone
        )


def test_glm_combine_fpc_invalid_values_rejected():
    for bad_fpc in ({"C": 0.5, "D": True}, {"C": 0.5, "D": 1.5}, {"C": 0.5}, 2.0, False):
        glm = SurveyGLM(model=ModelType.LINEAR)
        with pytest.raises(ValueError):
            glm.estimate(
                y=Y,
                x=X,
                samp_weight=WGT,
                stratum=STRATUM,
                psu=PSU_COLLIDING,
                single_psu=SinglePSUEst.combine,
                strata_comb=COMB,
                fpc=bad_fpc,
            )


def test_glm_combine_failure_preserves_state_and_inputs():
    glm = SurveyGLM(model=ModelType.LINEAR)
    glm.estimate(
        y=Y,
        x=X,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    beta_saved = glm.beta["point_est"].copy()
    cov_saved = glm.beta_cov.copy()
    labels = ["const", "slope"]
    glm.x_labels_custom = labels  # caller-owned list must survive

    stratum_before = STRATUM.copy()
    psu_before = PSU_COLLIDING.copy()
    with pytest.raises(ValueError):
        glm.estimate(
            y=Y,
            x=X,
            samp_weight=WGT,
            stratum=STRATUM,
            psu=PSU_COLLIDING,
            single_psu=SinglePSUEst.combine,
            strata_comb={"A": "D", "NOPE": "D"},
        )
    assert np.array_equal(glm.beta["point_est"], beta_saved)
    assert np.array_equal(glm.beta_cov, cov_saved)
    assert (STRATUM == stratum_before).all()
    assert (PSU_COLLIDING == psu_before).all()

    # the next legal call behaves like a fresh estimator
    fresh = SurveyGLM(model=ModelType.LINEAR)
    fresh.estimate(
        y=Y,
        x=X,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    glm.estimate(
        y=Y,
        x=X,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    assert np.allclose(glm.beta["point_est"], fresh.beta["point_est"])
    assert np.allclose(glm.beta_cov, fresh.beta_cov)


def test_glm_combine_does_not_mutate_caller_x_labels():
    labels = ["const", "slope"]
    glm = SurveyGLM(model=ModelType.LINEAR)
    glm.estimate(
        y=Y,
        x=X,
        x_labels=labels,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    assert labels == ["const", "slope"]
    assert glm.x_labels == ["const", "slope"]


def test_glm_combine_invariant_to_observation_order():
    order = np.array([4, 0, 6, 2, 5, 1, 7, 3])
    glm = SurveyGLM(model=ModelType.LINEAR)
    glm.estimate(
        y=Y[order],
        x=X[order],
        samp_weight=WGT[order],
        stratum=STRATUM[order],
        psu=PSU_COLLIDING[order],
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    expected_beta, expected_cov = independent_sandwich(Y, X, WGT, sm.families.Gaussian())
    assert np.allclose(glm.beta["point_est"], expected_beta)
    assert np.allclose(glm.beta_cov, expected_cov)


def test_glm_combine_invariant_to_psu_renaming_within_strata():
    psu_renamed = np.array([11, 11, 22, 22, 33, 33, 44, 44])
    glm = SurveyGLM(model=ModelType.LINEAR)
    glm.estimate(
        y=Y,
        x=X,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=psu_renamed,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    expected_beta, expected_cov = independent_sandwich(Y, X, WGT, sm.families.Gaussian())
    assert np.allclose(glm.beta["point_est"], expected_beta)
    assert np.allclose(glm.beta_cov, expected_cov)


def test_glm_combine_long_labels_and_display():
    glm = SurveyGLM(model=ModelType.LINEAR)
    glm.estimate(
        y=Y,
        x=X,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb={"A": "A_VERY_LONG_COMBINED_STRATUM", "B": "A_VERY_LONG_COMBINED_STRATUM"},
    )
    expected_beta, _ = independent_sandwich(Y, X, WGT, sm.families.Gaussian())
    assert np.allclose(glm.beta["point_est"], expected_beta)
    text = str(glm)
    assert "LINEAR" in text
    assert "Sample size: 8" in text


def test_glm_combine_with_remove_nan_stays_in_sync():
    # one NaN row: the design processing and the fit must use the same rows
    y = np.array([2.0, np.nan, 1.0, 3.0, 5.0, 7.0, 8.0, 6.0])
    glm = SurveyGLM(model=ModelType.LINEAR)
    glm.estimate(
        y=y,
        x=X,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
        remove_nan=True,
    )
    keep = np.array([0, 2, 3, 4, 5, 6, 7])
    fit = sm.GLM(Y[keep], X[keep], freq_weights=WGT[keep], family=sm.families.Gaussian()).fit()
    assert glm.sample_size == 7
    assert np.allclose(glm.beta["point_est"], fit.params)


def test_glm_skip_keeps_fpc_keyed_by_original_strata():
    # skip semantics unchanged: the fpc dictionary is still keyed by the
    # original strata, including the skipped one
    glm = SurveyGLM(model=ModelType.LINEAR)
    glm.estimate(
        y=Y,
        x=X,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu={"A": SinglePSUEst.skip, "B": SinglePSUEst.skip},
        fpc={"A": 0.9, "B": 0.8, "C": 0.7},
    )
    assert glm.sample_size == 4  # only stratum C rows remain
    assert set(glm.fpc.keys()) == {"A", "B", "C"}
