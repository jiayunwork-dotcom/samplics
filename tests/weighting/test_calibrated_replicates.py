"""Integration tests for calibrated replicates and the calibrated estimator.

A realistic stratified multi-PSU design drives jackknife, BRR and BRR-Fay
replicate generation; every replicate column is independently re-solved as a
ground truth, and total/mean/ratio point estimates, variances, domain results
and intervals are checked. A dedicated case shows that applying the
full-sample multiplier to every replicate gives the wrong variance.
"""

from __future__ import annotations

import numpy as np
import pytest

from samplics.estimation import ReplicateEstimator
from samplics.utils.types import PopParam, RepMethod
from samplics.weighting import (
    CalibratedReplicates,
    CalibrationError,
    ReplicateWeight,
)
from samplics.weighting.calibration import calibrate_bounded


def _make_design(rng, n_strata=5, psu_per_stratum=3, units_per_psu=8):
    total_psus = n_strata * psu_per_stratum
    n = total_psus * units_per_psu
    stratum = np.repeat([f"h{h}" for h in range(n_strata)], psu_per_stratum * units_per_psu)
    psu_parts = [
        np.repeat(
            [f"p{h}-{k}" for k in range(psu_per_stratum)], units_per_psu
        )
        for h in range(n_strata)
    ]
    psu = np.concatenate(psu_parts)
    weights = rng.uniform(1.0, 6.0, n)
    aux = np.column_stack(
        [np.ones(n), rng.normal(scale=10.0, size=n), rng.normal(size=n)]
    )
    y = rng.normal(50.0, 12.0, n)
    x = rng.normal(100.0, 20.0, n)
    domain = np.where(np.arange(n) % 2, "D1", "D2")
    control = aux.T @ weights * np.array([1.07, 0.94, 1.05])
    return weights, psu, stratum, aux, control, y, x, domain


def _independent_replicate_truth(raw_rep_weights, aux, control, bounds, tol, max_iter):
    """Solve every replicate column independently (the ground truth)."""

    r = raw_rep_weights.shape[1]
    out = np.zeros_like(raw_rep_weights, dtype=np.float64)
    for k in range(r):
        out[:, k] = calibrate_bounded(
            raw_rep_weights[:, k],
            aux,
            control,
            bounds=bounds,
            tol=tol,
            max_iter=max_iter,
            label=f"truth-{k}",
        )
    return out


@pytest.mark.parametrize(
    "method, kwargs",
    [
        (RepMethod.jackknife, {}),
        (RepMethod.brr, dict(nb_reps=8)),
        (RepMethod.brr, dict(nb_reps=8, fay_coef=0.3)),
    ],
)
def test_calibrated_replicates_match_independent_column_truth(method, kwargs):
    rng = np.random.default_rng(101)
    if method == RepMethod.brr:
        # BRR needs exactly two PSUs per stratum.
        weights, psu, stratum, aux, control, y, x, domain = _make_design(
            rng, n_strata=4, psu_per_stratum=2, units_per_psu=10
        )
        generator = ReplicateWeight(method=method, strat=True, **kwargs)
    else:
        weights, psu, stratum, aux, control, y, x, domain = _make_design(rng)
        generator = ReplicateWeight(method=method, strat=True)

    bounds = (0.3, 4.0)
    batch = generator.replicate(
        weights,
        psu,
        stratum,
        calibration=dict(aux_vars=aux, control=control, bounds=bounds),
        conservative=True,
    )
    assert isinstance(batch, CalibratedReplicates)

    # Full-sample truth.
    full_truth = calibrate_bounded(weights, aux, control, bounds=bounds)
    assert np.allclose(batch.full_weights, full_truth, rtol=1e-10, atol=1e-10)

    # Replicate column truth, solved one column at a time.
    raw_frame = generator._replicate_raw(weights, psu, stratum)
    prefix = generator._rep_prefix(None)
    raw_cols = [c for c in raw_frame.columns if c.startswith(prefix)]
    raw_rep = raw_frame[raw_cols].to_numpy(dtype=np.float64)
    truth = _independent_replicate_truth(raw_rep, aux, control, bounds, 1e-9, 500)
    assert np.allclose(batch.rep_weights, truth, rtol=1e-9, atol=1e-9)
    assert list(batch.replicate_ids) == raw_cols

    # Every column meets the same controls within tolerance; zero positions
    # follow each replicate's own base weights.
    col_tol = 1e-9 * np.maximum(1.0, np.abs(control))
    for k in range(batch.nb_reps):
        assert np.all(np.abs(aux.T @ batch.rep_weights[:, k] - control) <= col_tol)
        zero_base = raw_rep[:, k] == 0.0
        assert np.all(batch.rep_weights[zero_base, k] == 0.0)

    # Design metadata travels with the batch.
    assert batch.method == method
    assert len(batch.rep_coefs) == batch.nb_reps
    if method == RepMethod.brr:
        fay = kwargs.get("fay_coef", 0.0)
        assert np.isclose(batch.fay_coef, fay)
        assert np.allclose(
            batch.rep_coefs, 1.0 / (batch.nb_reps * (1.0 - fay) ** 2)
        )
        assert batch.degree_of_freedom == batch.nb_strata
    else:
        assert batch.fay_coef is None
        assert np.all((batch.rep_coefs > 0.0) & (batch.rep_coefs < 1.0))
        assert batch.degree_of_freedom == batch.nb_reps - batch.nb_strata


@pytest.mark.parametrize("conservative", [True, False])
@pytest.mark.parametrize("param", [PopParam.total, PopParam.mean, PopParam.ratio])
def test_estimate_calibrated_matches_hand_formulas(conservative, param):
    rng = np.random.default_rng(202)
    weights, psu, stratum, aux, control, y, x, domain = _make_design(rng)
    generator = ReplicateWeight(method=RepMethod.jackknife, strat=True)
    batch = generator.replicate(
        weights,
        psu,
        stratum,
        calibration=dict(aux_vars=aux, control=control, bounds=(0.3, 4.0)),
        conservative=conservative,
    )

    estimator = ReplicateEstimator(RepMethod.jackknife, param)
    xx = x if param == PopParam.ratio else None
    estimator.estimate_calibrated(
        y, batch, x=xx, domain=domain, conservative=conservative
    )

    # Point estimate from calibrated full-sample weights.
    if param == PopParam.total:
        point_ref = batch.full_weights @ y
    elif param == PopParam.mean:
        point_ref = (batch.full_weights @ y) / batch.full_weights.sum()
    else:
        point_ref = (batch.full_weights @ y) / (batch.full_weights @ x)
    domain_refs = {}
    for d_label in np.unique(domain):
        mask = domain == d_label
        if param == PopParam.total:
            ref = batch.full_weights[mask] @ y[mask]
        elif param == PopParam.mean:
            ref = (batch.full_weights[mask] @ y[mask]) / batch.full_weights[mask].sum()
        else:
            ref = (batch.full_weights[mask] @ y[mask]) / (
                batch.full_weights[mask] @ x[mask]
            )
        domain_refs[d_label] = ref
        assert np.isclose(estimator.point_est[d_label], ref)

    # Replicate estimates and variance by hand.
    if param == PopParam.total:
        rep_est = batch.rep_weights.T @ y
    elif param == PopParam.mean:
        rep_est = (batch.rep_weights.T @ y) / batch.rep_weights.sum(axis=0)
    else:
        rep_est = (batch.rep_weights.T @ y) / (batch.rep_weights.T @ x)

    if conservative:
        var_ref = np.sum(batch.rep_coefs * (rep_est - point_ref) ** 2)
    else:
        var_ref = np.sum(
            batch.rep_coefs * (rep_est - np.mean(rep_est)) ** 2
        )
    # Overall variance via the estimator's own no-domain run.
    overall = ReplicateEstimator(RepMethod.jackknife, param)
    overall.estimate_calibrated(y, batch, x=xx, conservative=conservative)
    assert np.isclose(overall.variance, var_ref)
    assert np.isclose(overall.stderror, np.sqrt(var_ref))
    assert overall.lower_ci < overall.point_est < overall.upper_ci

    # Domain variances reuse the same calibrated weights with masking only.
    for d_label in np.unique(domain):
        mask = domain == d_label
        sub_rep = batch.rep_weights[mask, :]
        if param == PopParam.total:
            rep_d = sub_rep.T @ y[mask]
            point_d = batch.full_weights[mask] @ y[mask]
        elif param == PopParam.mean:
            rep_d = (sub_rep.T @ y[mask]) / sub_rep.sum(axis=0)
            point_d = (batch.full_weights[mask] @ y[mask]) / batch.full_weights[
                mask
            ].sum()
        else:
            rep_d = (sub_rep.T @ y[mask]) / (sub_rep.T @ x[mask])
            point_d = (batch.full_weights[mask] @ y[mask]) / (
                batch.full_weights[mask] @ x[mask]
            )
        if conservative:
            v_d = np.sum(batch.rep_coefs * (rep_d - point_d) ** 2)
        else:
            v_d = np.sum(batch.rep_coefs * (rep_d - np.mean(rep_d)) ** 2)
        assert np.isclose(estimator.variance[d_label], v_d)
        assert np.isfinite(estimator.lower_ci[d_label])
        assert np.isfinite(estimator.upper_ci[d_label])

    frame = estimator.to_dataframe()
    assert set(frame["_domain"]) == set(np.unique(domain))
    assert frame.shape[0] == np.unique(domain).size


def test_shared_full_sample_multiplier_gives_wrong_variance():
    """The motivating counter-example: one multiplier on all replicates."""

    rng = np.random.default_rng(303)
    weights, psu, stratum, aux, control, y, x, domain = _make_design(
        rng, n_strata=6, psu_per_stratum=3, units_per_psu=6
    )
    generator = ReplicateWeight(method=RepMethod.jackknife, strat=True)
    batch = generator.replicate(
        weights,
        psu,
        stratum,
        calibration=dict(aux_vars=aux, control=control, bounds=(0.3, 4.0)),
        conservative=True,
    )

    raw_frame = generator._replicate_raw(weights, psu, stratum)
    raw_cols = [c for c in raw_frame.columns if c.startswith("_jk")]
    raw_rep = raw_frame[raw_cols].to_numpy(dtype=np.float64)
    naive = raw_rep * (batch.full_weights / weights)[:, None]

    # Naive replicates miss the controls by a wide margin on at least one column.
    naive_resid = max(
        np.max(np.abs(aux.T @ naive[:, k] - control)) for k in range(naive.shape[1])
    )
    good_resid = max(
        np.max(np.abs(aux.T @ batch.rep_weights[:, k] - control))
        for k in range(batch.nb_reps)
    )
    assert naive_resid > 1.0
    assert good_resid < 1e-7

    correct = ReplicateEstimator(RepMethod.jackknife, PopParam.total)
    correct.estimate_calibrated(y, batch, conservative=True)
    naive_est = naive.T @ y
    naive_var = float(
        np.sum(batch.rep_coefs * (naive_est - correct.point_est) ** 2)
    )
    # Same point estimate (both use calibrated full weights), different SE.
    assert np.isclose(correct.point_est, batch.full_weights @ y)
    assert not np.isclose(naive_var, correct.variance, rtol=0.05)


def test_whole_batch_failure_delivers_nothing_and_is_labeled():
    rng = np.random.default_rng(404)
    weights, psu, stratum, aux, control, y, x, domain = _make_design(rng)
    generator = ReplicateWeight(method=RepMethod.jackknife, strat=True)

    # Impossible total: no box can reach it.
    bad_control = control.copy()
    bad_control[0] = 1e12
    with pytest.raises(CalibrationError) as exc_info:
        generator.replicate(
            weights,
            psu,
            stratum,
            calibration=dict(aux_vars=aux, control=bad_control, max_iter=25),
        )
    err = exc_info.value
    assert err.target == "full"
    assert err.reason in ("infeasible", "iteration_limit")
    # A later legal call behaves like a fresh object.
    batch_a = generator.replicate(
        weights, psu, stratum, calibration=dict(aux_vars=aux, control=control)
    )
    batch_b = generator.replicate(
        weights, psu, stratum, calibration=dict(aux_vars=aux, control=control)
    )
    assert np.allclose(batch_a.full_weights, batch_b.full_weights)
    assert np.allclose(batch_a.rep_weights, batch_b.rep_weights)


def test_replicate_level_failure_identifies_the_column():
    rng = np.random.default_rng(505)
    weights, psu, stratum, aux, control, y, x, domain = _make_design(rng)
    generator = ReplicateWeight(method=RepMethod.jackknife, strat=True)

    # With a tight box, the full sample can reach a high intercept total but
    # some PSU-deleted replicates (narrower support) cannot.
    raw_frame = generator._replicate_raw(weights, psu, stratum)
    raw_cols = [c for c in raw_frame.columns if c.startswith("_jk")]
    raw_rep = raw_frame[raw_cols].to_numpy(dtype=np.float64)
    lo, hi = 0.9, 1.1

    def total_range(vec):
        pos = vec > 0
        return float(vec[pos].sum() * lo), float(vec[pos].sum() * hi)

    _, full_hi = total_range(weights)
    rep_hi = [total_range(raw_rep[:, k])[1] for k in range(raw_rep.shape[1])]
    smallest_rep_hi = min(rep_hi)
    ambitious = control.copy()
    ambitious[0] = 0.5 * (full_hi + smallest_rep_hi)
    assert smallest_rep_hi < ambitious[0] < full_hi

    with pytest.raises(CalibrationError) as exc_info:
        generator.replicate(
            weights,
            psu,
            stratum,
            calibration=dict(aux_vars=aux, control=ambitious, bounds=(lo, hi)),
        )
    err = exc_info.value
    assert err.target != "full"
    assert err.target in raw_cols
    assert err.reason == "infeasible"
    assert err.certificate is not None


def test_bootstrap_and_bad_options_are_rejected():
    rng = np.random.default_rng(606)
    weights, psu, stratum, aux, control, y, x, domain = _make_design(rng)
    boot = ReplicateWeight(method=RepMethod.bootstrap, strat=True, nb_reps=10)
    with pytest.raises(ValueError):
        boot.replicate(
            weights,
            psu,
            stratum,
            calibration=dict(aux_vars=aux, control=control),
        )

    jk = ReplicateWeight(method=RepMethod.jackknife, strat=True)
    with pytest.raises(ValueError):
        jk.replicate(
            weights,
            psu,
            stratum,
            rep_coefs=True,
            calibration=dict(aux_vars=aux, control=control),
        )
    with pytest.raises(ValueError):
        jk.replicate(
            weights,
            psu,
            stratum,
            calibration=dict(aux_vars=aux, control=control, strange_key=1),
        )
    with pytest.raises(ValueError):
        jk.replicate(
            weights, psu, stratum,
            calibration=dict(aux_vars=aux, control=control),
            conservative="yes",
        )
    with pytest.raises(ValueError):
        jk.replicate(weights, psu, stratum, calibration="not-a-mapping")


def test_legacy_uncalibrated_path_unchanged():
    rng = np.random.default_rng(707)
    weights, psu, stratum, aux, control, y, x, domain = _make_design(rng)
    import pandas as pd

    jk = ReplicateWeight(method=RepMethod.jackknife, strat=True)
    frame = jk.replicate(weights, psu, stratum)
    assert isinstance(frame, pd.DataFrame)
    assert any(c.startswith("_jk_wgt_") for c in frame.columns)


def test_estimator_rejections_and_state_persistence():
    rng = np.random.default_rng(808)
    weights, psu, stratum, aux, control, y, x, domain = _make_design(rng)
    jk = ReplicateWeight(method=RepMethod.jackknife, strat=True)
    batch = jk.replicate(
        weights,
        psu,
        stratum,
        calibration=dict(aux_vars=aux, control=control),
        conservative=True,
    )
    estimator = ReplicateEstimator(RepMethod.jackknife, PopParam.total)
    estimator.estimate_calibrated(y, batch, conservative=True)
    saved_point = estimator.point_est
    saved_var = estimator.variance

    # Method mismatch.
    with pytest.raises(ValueError):
        ReplicateEstimator(RepMethod.brr, PopParam.total).estimate_calibrated(
            y, batch, conservative=True
        )
    # Wrong length.
    with pytest.raises(ValueError):
        estimator.estimate_calibrated(y[:-1], batch, conservative=True)
    # Conservative convention mismatch.
    with pytest.raises(ValueError):
        estimator.estimate_calibrated(y, batch, conservative=False)
    # Raw array instead of a batch.
    with pytest.raises(ValueError):
        estimator.estimate_calibrated(y, batch.rep_weights, conservative=True)

    # State after the failed calls is still the last successful result.
    assert estimator.point_est == saved_point
    assert estimator.variance == saved_var

    # A subsequent legal call works and refreshes results.
    estimator.estimate_calibrated(y + 1.0, batch, conservative=True)
    assert estimator.point_est != saved_point


def test_fay_estimator_uses_batch_coefficients_and_rejects_mismatch():
    rng = np.random.default_rng(909)
    weights, psu, stratum, aux, control, y, x, domain = _make_design(
        rng, n_strata=4, psu_per_stratum=2, units_per_psu=10
    )
    fay = ReplicateWeight(
        method=RepMethod.brr, strat=True, nb_reps=8, fay_coef=0.3
    )
    batch = fay.replicate(
        weights,
        psu,
        stratum,
        calibration=dict(aux_vars=aux, control=control),
        conservative=True,
    )
    good = ReplicateEstimator(RepMethod.brr, PopParam.total, fay_coef=0.3)
    good.estimate_calibrated(y, batch, conservative=True)
    rep_est = batch.rep_weights.T @ y
    expected = np.sum(batch.rep_coefs * (rep_est - good.point_est) ** 2)
    assert np.isclose(good.variance, expected)
    assert good.degree_of_freedom == batch.degree_of_freedom

    with pytest.raises(ValueError):
        ReplicateEstimator(RepMethod.brr, PopParam.total, fay_coef=0.5).estimate_calibrated(
            y, batch, conservative=True
        )
