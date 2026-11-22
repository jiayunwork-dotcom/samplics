"""Regression tests for combining strata with singleton PSUs (TaylorEstimator).

All expected values are computed by hand from the linearized scores aggregated
over the *real* survey units: a PSU is identified by the (original stratum, psu)
pair, so two PSUs that happen to share a label in different strata remain
distinct units after their strata are combined for variance estimation. No
expectation below is produced by calling samplics itself.
"""

import numpy as np
import pytest
from scipy.stats import t as student

from samplics.estimation import TaylorEstimator
from samplics.utils.types import PopParam, SinglePSUEst

T_QT_1 = student.ppf(0.975, df=1)  # 12.706204736432095


def between_psu_variance(*psu_score_sums):
    """Hand-computed stratified between-PSU variance for one final stratum."""
    sums = np.asarray(psu_score_sums, dtype=float)
    n_h = sums.size
    return n_h / (n_h - 1) * np.sum((sums - sums.mean()) ** 2)


# The reproduction case: two strata, one PSU each, and both PSUs happen to be
# labelled 1. Combining the strata must keep the two units distinct.
Y = np.array([10.0, 20.0, 30.0, 40.0])
WGT = np.ones(4)
STRATUM = np.array(["A", "A", "B", "B"])
PSU_COLLIDING = np.array([1, 1, 1, 1])  # same label "1" in strata A and B
PSU_DISTINCT = np.array([1, 1, 2, 2])  # collision-free control
COMB = {"A": "C", "B": "C"}


def test_combine_total_colliding_psu_labels():
    est = TaylorEstimator(PopParam.total)
    est.estimate(
        y=Y,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    # real PSUs: (A,1) -> 10+20 = 30 ; (B,1) -> 30+40 = 70
    expected_var = between_psu_variance(30.0, 70.0)  # 1600.0
    assert np.isclose(est.point_est, 100.0)
    assert np.isclose(est.variance, expected_var)
    assert np.isclose(est.variance, 1600.0)
    assert np.isclose(est.stderror, 40.0)
    assert est.nb_strata == 1
    assert est.nb_psus == 2
    assert est.degree_of_freedom == 1
    assert np.isclose(est.lower_ci, 100.0 - T_QT_1 * 40.0)
    assert np.isclose(est.upper_ci, 100.0 + T_QT_1 * 40.0)
    assert np.isclose(est.coef_var, 0.4)


def test_combine_total_matches_collision_free_control():
    """Only the unit labels differ: the analysis must be identical."""
    est = TaylorEstimator(PopParam.total)
    est.estimate(
        y=Y,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    control = TaylorEstimator(PopParam.total)
    control.estimate(
        y=Y,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_DISTINCT,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    assert np.isclose(est.point_est, control.point_est)
    assert np.isclose(est.variance, control.variance)
    assert np.isclose(est.stderror, control.stderror)
    assert est.degree_of_freedom == control.degree_of_freedom
    assert np.isclose(est.lower_ci, control.lower_ci)
    assert np.isclose(est.upper_ci, control.upper_ci)
    # identical results down to the text and dataframe displays
    assert str(est) == str(control)
    df_est = est.to_dataframe()
    df_control = control.to_dataframe()
    for col in ["_estimate", "_stderror", "_lci", "_uci", "_cv"]:
        assert np.allclose(df_est[col].to_numpy(float), df_control[col].to_numpy(float))


def test_combine_mean_colliding_psu_labels():
    est = TaylorEstimator(PopParam.mean)
    est.estimate(
        y=Y,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    # scores: w * (y - 25) / 4 -> PSU sums (A,1) = -5, (B,1) = 5
    expected_var = between_psu_variance(-5.0, 5.0)  # 100.0
    assert np.isclose(est.point_est, 25.0)
    assert np.isclose(est.variance, expected_var)
    assert np.isclose(est.variance, 100.0)
    assert est.degree_of_freedom == 1


def test_combine_ratio_colliding_psu_labels():
    x = np.array([1.0, 2.0, 1.0, 2.0])
    est = TaylorEstimator(PopParam.ratio)
    est.estimate(
        y=Y,
        x=x,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    # ratio = 100/6 = 50/3 ; scores: w * (y - x*50/3) / 6
    # PSU sums: (A,1) = -10/3, (B,1) = 10/3
    expected_var = between_psu_variance(-10.0 / 3.0, 10.0 / 3.0)  # 400/9
    assert np.isclose(est.point_est, 50.0 / 3.0)
    assert np.isclose(est.variance, expected_var)
    assert np.isclose(est.variance, 400.0 / 9.0)
    assert est.degree_of_freedom == 1


def test_combine_domain_total_keeps_full_design():
    domain = np.array([1, 1, 2, 2])
    est = TaylorEstimator(PopParam.total)
    est.estimate(
        y=Y,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        domain=domain,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    # domain 1 scores: [10, 20, 0, 0] -> PSU sums 30 and 0
    # domain 2 scores: [0, 0, 30, 40] -> PSU sums 0 and 70
    assert np.isclose(est.point_est[1], 30.0)
    assert np.isclose(est.point_est[2], 70.0)
    assert np.isclose(est.variance[1], between_psu_variance(30.0, 0.0))  # 900
    assert np.isclose(est.variance[2], between_psu_variance(0.0, 70.0))  # 4900
    # the domain estimation keeps the full design: both real PSUs counted once
    assert est.nb_psus == 2
    assert est.nb_strata == 1
    assert est.degree_of_freedom == 1


def test_combine_domain_ratio_colliding_psu_labels():
    x = np.array([2.0, 1.0, 3.0, 5.0])
    domain = np.array([1, 2, 1, 2])  # domains cut across the two PSUs
    est = TaylorEstimator(PopParam.ratio)
    est.estimate(
        y=Y,
        x=x,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        domain=domain,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    # domain 1 (rows 0, 2): ratio = 40/5 = 8 ; scores d*w*(y - 8x)/5
    #   row 0: -6/5 (PSU (A,1)), row 2: 6/5 (PSU (B,1))
    # domain 2 (rows 1, 3): ratio = 60/6 = 10 ; scores d*w*(y - 10x)/6
    #   row 1: 5/3 (PSU (A,1)), row 3: -5/3 (PSU (B,1))
    assert np.isclose(est.point_est[1], 8.0)
    assert np.isclose(est.point_est[2], 10.0)
    assert np.isclose(est.variance[1], between_psu_variance(-6.0 / 5.0, 6.0 / 5.0))
    assert np.isclose(est.variance[2], between_psu_variance(5.0 / 3.0, -5.0 / 3.0))
    assert est.degree_of_freedom == 1


def test_combine_into_existing_stratum_preserves_its_psus():
    # stratum B already has two PSUs; merging A into B must keep all three units
    stratum = np.array(["A", "A", "B", "B", "B", "B"])
    psu = np.array([1, 1, 1, 1, 2, 2])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    est = TaylorEstimator(PopParam.total)
    est.estimate(
        y=y,
        samp_weight=np.ones(6),
        stratum=stratum,
        psu=psu,
        single_psu=SinglePSUEst.combine,
        strata_comb={"A": "B"},
    )
    # real PSUs in final stratum B: (A,1)=30, (B,1)=70, (B,2)=110
    expected_var = between_psu_variance(30.0, 70.0, 110.0)  # 4800
    assert np.isclose(est.point_est, 210.0)
    assert np.isclose(est.variance, expected_var)
    assert np.isclose(est.variance, 4800.0)
    assert est.nb_psus == 3
    assert est.nb_strata == 1
    assert est.degree_of_freedom == 2


def test_combine_chain_mapping_is_not_transitive():
    # A -> B and B -> C in one simultaneous pass: rows of A land in B, rows of B
    # land in C; the rows moved out of A are NOT moved on to C.
    stratum = np.array(["A", "A", "A", "A", "B", "B", "C", "C", "C"])
    psu = np.array([1, 1, 2, 2, 1, 1, 1, 2, 2])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
    est = TaylorEstimator(PopParam.total)
    est.estimate(
        y=y,
        samp_weight=np.ones(9),
        stratum=stratum,
        psu=psu,
        single_psu=SinglePSUEst.combine,
        strata_comb={"A": "B", "B": "C"},
    )
    # final B: (A,1)=30, (A,2)=70 ; final C: (B,1)=110, (C,1)=70, (C,2)=170
    expected_var = between_psu_variance(30.0, 70.0) + between_psu_variance(110.0, 70.0, 170.0)
    assert np.isclose(est.point_est, 450.0)
    assert np.isclose(est.variance, expected_var)  # 1600 + 7600 = 9200
    assert np.isclose(est.variance, 9200.0)
    assert est.nb_psus == 5
    assert est.nb_strata == 2
    assert est.degree_of_freedom == 3


def test_combine_long_target_labels_not_truncated():
    stratum = np.array(["S1", "S1", "S2", "S2"])
    psu = np.array([1, 1, 1, 1])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    target = "A_MUCH_LONGER_STRATUM_NAME"
    est = TaylorEstimator(PopParam.total)
    est.estimate(
        y=y,
        samp_weight=np.ones(4),
        stratum=stratum,
        psu=psu,
        single_psu=SinglePSUEst.combine,
        strata_comb={"S1": target, "S2": target},
    )
    assert list(est.strata) == [target]
    assert est.nb_strata == 1
    assert est.nb_psus == 2
    # PSU sums: (S1,1)=3, (S2,1)=7
    assert np.isclose(est.variance, between_psu_variance(3.0, 7.0))  # 16.0


def test_combine_int_strata_and_int_labels():
    stratum = np.array([1, 1, 2, 2])
    psu = np.array([1, 1, 1, 1])
    est = TaylorEstimator(PopParam.total)
    est.estimate(
        y=Y,
        samp_weight=WGT,
        stratum=stratum,
        psu=psu,
        single_psu=SinglePSUEst.combine,
        strata_comb={1: 3, 2: 3},
    )
    assert np.isclose(est.point_est, 100.0)
    assert np.isclose(est.variance, 1600.0)
    assert est.degree_of_freedom == 1


def test_combine_dict_single_psu_form():
    est = TaylorEstimator(PopParam.total)
    est.estimate(
        y=Y,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu={("A", "B"): SinglePSUEst.combine},
        strata_comb=COMB,
    )
    assert np.isclose(est.point_est, 100.0)
    assert np.isclose(est.variance, 1600.0)
    assert est.degree_of_freedom == 1


def test_combine_same_psu_multiple_rows_one_unit():
    # many rows per PSU: the PSU total is what matters, not the row count
    stratum = np.array(["A", "A", "A", "B", "B", "B", "B"])
    psu = np.array([1, 1, 1, 1, 1, 1, 1])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    est = TaylorEstimator(PopParam.total)
    est.estimate(
        y=y,
        samp_weight=np.ones(7),
        stratum=stratum,
        psu=psu,
        single_psu=SinglePSUEst.combine,
        strata_comb={"A": "C", "B": "C"},
    )
    # (A,1) = 6, (B,1) = 22
    assert np.isclose(est.point_est, 28.0)
    assert np.isclose(est.variance, between_psu_variance(6.0, 22.0))  # 512
    assert est.nb_psus == 2


def test_combine_still_single_psu_is_rejected():
    # only A is combined away; B keeps a single real PSU -> clear refusal
    est = TaylorEstimator(PopParam.total)
    with pytest.raises(ValueError, match="Only one PSU"):
        est.estimate(
            y=Y,
            samp_weight=WGT,
            stratum=STRATUM,
            psu=PSU_COLLIDING,
            single_psu=SinglePSUEst.combine,
            strata_comb={"A": "C"},
        )


def test_combine_swap_does_not_merge_and_is_rejected():
    # swapping labels merges nothing: each final stratum keeps one real PSU
    est = TaylorEstimator(PopParam.total)
    with pytest.raises(ValueError, match="Only one PSU"):
        est.estimate(
            y=Y,
            samp_weight=WGT,
            stratum=STRATUM,
            psu=PSU_COLLIDING,
            single_psu=SinglePSUEst.combine,
            strata_comb={"A": "B", "B": "A"},
        )


def test_combine_invalid_mappings_raise_valueerror():
    base = dict(
        y=Y,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
    )
    for bad_mapping in (
        None,  # missing
        {},  # empty
        {"A": "C", "ZZZ": "C"},  # source not in the sample
        {"A": 3, "B": 3},  # int targets for string strata
        {"A": "C", "B": True},  # bool is not a valid label
    ):
        est = TaylorEstimator(PopParam.total)
        with pytest.raises(ValueError):
            est.estimate(**base, strata_comb=bad_mapping)


def test_combine_stringified_mapping_rejected_for_int_strata():
    est = TaylorEstimator(PopParam.total)
    with pytest.raises(ValueError):
        est.estimate(
            y=Y,
            samp_weight=WGT,
            stratum=np.array([1, 1, 2, 2]),
            psu=PSU_COLLIDING,
            single_psu=SinglePSUEst.combine,
            strata_comb={"1": "3", "2": "3"},  # strings for int strata
        )


def test_combine_fpc_scalar_applies_to_final_strata():
    est = TaylorEstimator(PopParam.total)
    est.estimate(
        y=Y,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
        fpc=0.5,
    )
    assert np.isclose(est.variance, 0.5 * 1600.0)
    assert est.fpc == {"C": 0.5}


def test_combine_fpc_dict_keyed_by_final_strata_any_order():
    stratum = np.array(["A", "A", "A", "A", "B", "B", "C", "C", "C"])
    psu = np.array([1, 1, 2, 2, 1, 1, 1, 2, 2])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
    comb = {"A": "B", "B": "C"}  # final strata: B and C
    kwargs = dict(
        y=y,
        samp_weight=np.ones(9),
        stratum=stratum,
        psu=psu,
        single_psu=SinglePSUEst.combine,
        strata_comb=comb,
    )
    est1 = TaylorEstimator(PopParam.total)
    est1.estimate(**kwargs, fpc={"B": 0.9, "C": 0.8})
    est2 = TaylorEstimator(PopParam.total)
    est2.estimate(**kwargs, fpc={"C": 0.8, "B": 0.9})  # reversed insertion order
    # 0.9 * 1600 + 0.8 * 7600 = 7520
    assert np.isclose(est1.variance, 0.9 * 1600.0 + 0.8 * 7600.0)
    assert np.isclose(est1.variance, est2.variance)
    assert np.isclose(est1.point_est, est2.point_est)
    assert list(est1.fpc.keys()) == list(est2.fpc.keys())


def test_combine_fpc_dict_must_not_reference_disappeared_strata():
    est = TaylorEstimator(PopParam.total)
    with pytest.raises(ValueError, match="final strata"):
        est.estimate(
            y=Y,
            samp_weight=WGT,
            stratum=STRATUM,
            psu=PSU_COLLIDING,
            single_psu=SinglePSUEst.combine,
            strata_comb=COMB,
            fpc={"A": 0.9, "B": 0.8, "C": 0.7},  # A and B no longer exist
        )


def test_combine_fpc_invalid_values_rejected():
    base = dict(
        y=Y,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    for bad_fpc in (
        {"C": 0.5, "D": 0.5},  # extra key
        {"C": True},  # bool is not a number
        {"C": 1.5},  # outside [0, 1]
        {"C": -0.1},  # outside [0, 1]
        {"C": float("nan")},  # not finite
        {"C": float("inf")},  # not finite
        True,  # bool scalar
        1.5,  # scalar outside [0, 1]
        float("nan"),  # scalar not finite
    ):
        est = TaylorEstimator(PopParam.total)
        with pytest.raises(ValueError):
            est.estimate(**base, fpc=bad_fpc)


def test_combine_failure_preserves_state_and_inputs():
    est = TaylorEstimator(PopParam.total)
    est.estimate(
        y=Y,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    saved = (est.point_est, est.variance, est.stderror, est.degree_of_freedom)

    stratum_before = STRATUM.copy()
    psu_before = PSU_COLLIDING.copy()
    bad_mapping = {"A": "C", "ZZZ": "C"}
    with pytest.raises(ValueError):
        est.estimate(
            y=Y,
            samp_weight=WGT,
            stratum=STRATUM,
            psu=PSU_COLLIDING,
            single_psu=SinglePSUEst.combine,
            strata_comb=bad_mapping,
        )
    # previous public results, caller arrays and the mapping are untouched
    assert (est.point_est, est.variance, est.stderror, est.degree_of_freedom) == saved
    assert (STRATUM == stratum_before).all()
    assert (PSU_COLLIDING == psu_before).all()
    assert bad_mapping == {"A": "C", "ZZZ": "C"}

    # a failing fpc validation must not clobber the results either
    with pytest.raises(ValueError):
        est.estimate(
            y=Y,
            samp_weight=WGT,
            stratum=STRATUM,
            psu=PSU_COLLIDING,
            single_psu=SinglePSUEst.combine,
            strata_comb=COMB,
            fpc={"C": 2.0},
        )
    assert (est.point_est, est.variance, est.stderror, est.degree_of_freedom) == saved

    # the next legal call behaves like a fresh estimator
    fresh = TaylorEstimator(PopParam.total)
    fresh.estimate(
        y=Y,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    est.estimate(
        y=Y,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    assert est.point_est == fresh.point_est
    assert est.variance == fresh.variance
    assert est.stderror == fresh.stderror
    assert est.degree_of_freedom == fresh.degree_of_freedom


def test_combine_invariant_to_observation_order():
    order = np.array([2, 0, 3, 1])
    est = TaylorEstimator(PopParam.total)
    est.estimate(
        y=Y[order],
        samp_weight=WGT[order],
        stratum=STRATUM[order],
        psu=PSU_COLLIDING[order],
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    assert np.isclose(est.point_est, 100.0)
    assert np.isclose(est.variance, 1600.0)
    assert est.degree_of_freedom == 1


def test_combine_invariant_to_psu_renaming_within_strata():
    psu_renamed = np.array([101, 101, 202, 202])  # per-stratum bijective renaming
    est = TaylorEstimator(PopParam.total)
    est.estimate(
        y=Y,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=psu_renamed,
        single_psu=SinglePSUEst.combine,
        strata_comb=COMB,
    )
    assert np.isclose(est.point_est, 100.0)
    assert np.isclose(est.variance, 1600.0)
    assert est.degree_of_freedom == 1


def test_combine_singleton_detection_uses_real_units():
    # without combining, the default is still to refuse singleton strata
    est = TaylorEstimator(PopParam.total)
    with pytest.raises(ValueError, match="Only one PSU"):
        est.estimate(y=Y, samp_weight=WGT, stratum=STRATUM, psu=PSU_COLLIDING)
    # and skip still drops the singleton strata from the variance only
    est_skip = TaylorEstimator(PopParam.total)
    est_skip.estimate(
        y=Y,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.skip,
    )
    assert np.isclose(est_skip.point_est, 100.0)
    assert np.isclose(est_skip.variance, 0.0)


def test_combine_dict_mix_certainty_and_combine():
    # stratum 1 handled with certainty (its two observations become the units),
    # stratum 2 combined into stratum 3
    y = np.array([2.0, 4.0, 1.0, 3.0, 5.0, 7.0])
    stratum = np.array([1, 1, 2, 2, 3, 3])
    psu = np.array([1, 1, 1, 1, 1, 2])
    est = TaylorEstimator(PopParam.total)
    est.estimate(
        y=y,
        samp_weight=np.ones(6),
        stratum=stratum,
        psu=psu,
        single_psu={1: SinglePSUEst.certainty, 2: SinglePSUEst.combine},
        strata_comb={2: 3},
    )
    # stratum 1: units (1,obs1)=2, (1,obs2)=4 -> var (2-4)^2 = 4
    # stratum 3: units (2,1)=1+3=4, (3,1)=5, (3,2)=7 -> 1.5*(16/9+1/9+25/9) = 7
    expected_var = (2.0 - 4.0) ** 2 + between_psu_variance(4.0, 5.0, 7.0)
    assert np.isclose(est.point_est, 22.0)
    assert np.isclose(est.variance, expected_var)  # 11.0
    assert np.isclose(est.variance, 11.0)
    assert est.nb_psus == 5
    assert est.degree_of_freedom == 3


def test_combine_with_remove_nan():
    y = np.array([10.0, np.nan, 30.0, 40.0])
    stratum = np.array(["A", "A", "B", "B"])
    psu = np.array([1, 1, 1, 1])
    est = TaylorEstimator(PopParam.total)
    est.estimate(
        y=y,
        samp_weight=np.array([1.0, 1.0, 1.0, 1.0]),
        stratum=stratum,
        psu=psu,
        single_psu=SinglePSUEst.combine,
        strata_comb={"A": "C", "B": "C"},
        remove_nan=True,
    )
    # row 1 dropped: (A,1)=10, (B,1)=70
    assert np.isclose(est.point_est, 80.0)
    assert np.isclose(est.variance, between_psu_variance(10.0, 70.0))  # 7200
    assert est.degree_of_freedom == 1


def test_skip_keeps_fpc_keyed_by_original_strata():
    # skip semantics unchanged: fpc dictionary keyed by the original strata
    est = TaylorEstimator(PopParam.total)
    est.estimate(
        y=Y,
        samp_weight=WGT,
        stratum=STRATUM,
        psu=PSU_COLLIDING,
        single_psu=SinglePSUEst.skip,
        fpc={"A": 0.9, "B": 0.8},
    )
    assert np.isclose(est.point_est, 100.0)
    assert np.isclose(est.variance, 0.0)  # both singleton strata skipped
