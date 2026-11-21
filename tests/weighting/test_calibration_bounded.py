"""Tests for bounded chi-square calibration.

The hand-enumerated boundary sets are checked row by row on tiny tables;
medium problems are compared against independent SLSQP optima that are
computed in the test itself rather than back-solved from the new solver.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from scipy.optimize import linprog, minimize

from samplics.weighting import CalibrationError, SampleWeight, calibrate_bounded


# ---------------------------------------------------------------------------
# Hand-enumerated small tables: every boundary set is enumerated explicitly.
# ---------------------------------------------------------------------------


def _enumerate_boundary_sets(d, x_col, target, lower, upper):
    """Brute-force optimum over all 3**m active-set assignments.

    For each choice of {lower-active, free, upper-active} per positive row,
    the free rows solve the equality QP; feasible assignments are compared on
    the chi-square objective. Returns the optimal weight vector.
    """

    pos = d > 0.0
    d_p = d[pos]
    x_p = np.asarray(x_col[pos], dtype=np.float64).reshape(-1)
    target_scalar = float(np.asarray(target).reshape(-1)[0])
    m = d_p.size
    best = None
    best_obj = np.inf
    for assignment in itertools.product(("lo", "free", "hi"), repeat=m):
        g = np.empty(m)
        free = [k for k, a in enumerate(assignment) if a == "free"]
        for k, a in enumerate(assignment):
            g[k] = lower if a == "lo" else upper if a == "hi" else g[k]
        if free:
            # g_k = 1 + lam * x_k on free rows; solve the single control.
            rhs = target_scalar - sum(
                d_p[k] * x_p[k] * g[k] for k in range(m) if k not in free
            )
            denom = sum(d_p[k] * x_p[k] ** 2 for k in free)
            if abs(denom) <= 1e-14:
                continue
            lam = (rhs - sum(d_p[k] * x_p[k] for k in free)) / denom
            for k in free:
                g[k] = 1.0 + lam * x_p[k]
            if any(g[k] < lower - 1e-10 or g[k] > upper + 1e-10 for k in free):
                continue
        elif abs(x_p @ (d_p * g) - target_scalar) > 1e-9 * max(
            1.0, abs(target_scalar)
        ):
            continue
        w = d_p * g
        if abs(x_p @ w - target_scalar) > 1e-9 * max(1.0, abs(target_scalar)):
            continue
        obj = float(np.sum(d_p * (g - 1.0) ** 2))
        if obj < best_obj:
            best_obj, best = obj, w
    assert best is not None, "enumeration found no feasible point"
    full = np.zeros_like(d)
    full[pos] = best
    return full, best_obj


def test_hand_enumerated_boundary_sets_single_column():
    d = np.array([1.0, 2.0, 0.0, 3.0, 4.0])  # one structural zero
    x = np.array([2.0, -1.0, 5.0, 0.5, 1.5]).reshape(-1, 1)
    lower, upper = 0.6, 1.8
    # Target chosen so the optimum mixes bound and free rows.
    target = np.array([14.0])
    w = calibrate_bounded(d, x, target, bounds=(lower, upper))

    # Structural zero is never revived.
    assert w[2] == 0.0
    # Box holds on every positive row.
    g = w[d > 0] / d[d > 0]
    assert np.all(g >= lower - 1e-12) and np.all(g <= upper + 1e-12)

    w_ref, obj_ref = _enumerate_boundary_sets(d, x.reshape(-1), target, lower, upper)
    assert np.allclose(w, w_ref, rtol=1e-7, atol=1e-9)
    assert np.isclose(
        np.sum((w[d > 0] - d[d > 0]) ** 2 / d[d > 0]), obj_ref, rtol=1e-7
    )
    # Control met within the per-column allowance.
    assert abs(x.T @ w - target)[0] <= 1e-9 * max(1.0, abs(target[0]))


def test_hand_enumerated_all_rows_bound():
    # Only one feasible point: every row must sit on the lower bound.
    d = np.array([2.0, 3.0, 5.0])
    x = np.ones((3, 1))
    lower, upper = 0.4, 2.5
    target = np.array([lower * float(np.sum(d))])
    w = calibrate_bounded(d, x, target, bounds=(lower, upper))
    assert np.allclose(w, lower * d)
    assert np.allclose(w / d, lower)


def test_hand_enumerated_two_columns_all_active_patterns():
    rng = np.random.default_rng(11)
    d = rng.uniform(0.5, 4.0, size=5)
    x = np.column_stack([np.ones(5), rng.normal(size=5)])
    lower, upper = 0.5, 2.0
    # Feasible target built from a factor vector touching both bounds.
    g0 = np.array([lower, upper, 1.1, lower, 1.3])
    target = x.T @ (d * g0)
    w = calibrate_bounded(d, x, target, bounds=(lower, upper))
    assert np.max(np.abs(x.T @ w - target)) <= 1e-9 * np.max(
        np.maximum(1.0, np.abs(target))
    )
    # The chi-square optimum cannot be worse than the feasible g0.
    assert np.sum(d * (w / d - 1.0) ** 2) <= np.sum(d * (g0 - 1.0) ** 2) + 1e-12
    factors = w / d
    assert np.all(factors >= lower - 1e-10) and np.all(factors <= upper + 1e-10)
    # Construct a target on the boundary of the feasible box by maximizing one
    # auxiliary total over g in [lower, upper] (an LP vertex), which forces
    # bound-active rows at the chi-square optimum too.
    dx = d[:, None] * x
    edge_g = linprog(c=-dx[:, 0], bounds=[(lower, upper)] * 5, method="highs").x
    control_edge = dx.T @ np.clip(edge_g, lower, upper)
    w_edge = calibrate_bounded(
        d, x, control_edge, bounds=(lower, upper)
    )
    w_ref, _ = _slsqp_solution(d, x, control_edge, lower, upper)
    assert np.allclose(w_edge, w_ref, rtol=1e-6, atol=1e-7)
    g_edge = w_edge / d
    # A box-vertex target leaves at least one row on a bound at the optimum.
    assert np.any(
        np.isclose(g_edge, lower, atol=1e-7) | np.isclose(g_edge, upper, atol=1e-7)
    )


# ---------------------------------------------------------------------------
# Independent optimum cross-checks (SLSQP, not derived from the new solver).
# ---------------------------------------------------------------------------


def _slsqp_solution(d, x, control, lower, upper):
    """Independent bounded optimum via SciPy (a different algorithm)."""

    n, p = x.shape
    constraints = [
        {"type": "eq", "fun": (lambda g, j=j: x[:, j] @ (d * g) - control[j])}
        for j in range(p)
    ]
    result = minimize(
        lambda g: float(np.sum(d * (g - 1.0) ** 2)),
        np.clip(np.ones(n), lower, upper),
        jac=lambda g: 2.0 * d * (g - 1.0),
        method="SLSQP",
        bounds=[(lower, upper)] * n,
        constraints=constraints,
        options={"maxiter": 10000, "ftol": 1e-14},
    )
    if not result.success:
        # Trust-constr as an independent fallback for harder binding cases.
        from scipy.optimize import LinearConstraint
        from scipy.optimize import minimize as _min

        lc = LinearConstraint(x.T * d, control - 1e-12, control + 1e-12)
        result = _min(
            lambda g: float(np.sum(d * (g - 1.0) ** 2)),
            np.clip(np.ones(n), lower, upper),
            jac=lambda g: 2.0 * d * (g - 1.0),
            method="trust-constr",
            bounds=[(lower, upper)] * n,
            constraints=[lc],
            options={"maxiter": 2000, "gtol": 1e-12},
        )
        assert result.success, result.message
    return d * result.x, result.fun


def test_medium_conditioned_matches_independent_optimum():
    rng = np.random.default_rng(7)
    n, p = 60, 6
    d = rng.uniform(0.5, 6.0, n)
    x = np.column_stack(
        [np.ones(n), rng.normal(scale=4.0, size=(n, p - 1))]
    )
    # Interior feasible target generated from in-box factors.
    g0 = np.clip(1.0 + 0.15 * rng.normal(size=n), 0.6, 4.5)
    control = x.T @ (d * g0)
    w = calibrate_bounded(d, x, control)
    w_ref, obj_ref = _slsqp_solution(d, x, control, 0.2, 5.0)
    assert np.allclose(w, w_ref, rtol=1e-7, atol=1e-9)
    assert np.isclose(
        np.sum((w - d) ** 2 / d), obj_ref, rtol=1e-7, atol=1e-9
    )
    col_tol = 1e-9 * np.maximum(1.0, np.abs(control))
    assert np.all(np.abs(x.T @ w - control) <= col_tol)


def test_medium_binding_matches_independent_optimum():
    rng = np.random.default_rng(19)
    n, p = 80, 5
    d = rng.uniform(1.0, 9.0, n)
    x = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    # Near-edge feasible target so several rows bind: build it directly from
    # a factor vector with rows on both bounds.
    g0 = np.where(
        rng.random(n) < 0.3,
        0.8,
        np.where(rng.random(n) < 0.45, 1.25, 0.9 + 0.2 * rng.random(n)),
    )
    control = x.T @ (d * g0)
    w = calibrate_bounded(d, x, control, bounds=(0.8, 1.25))
    w_ref, obj_ref = _slsqp_solution(d, x, control, 0.8, 1.25)
    assert np.allclose(w, w_ref, rtol=1e-6, atol=1e-7)
    assert np.isclose(np.sum((w - d) ** 2 / d), obj_ref, rtol=1e-6)
    factors = w / d
    assert np.all(factors >= 0.8 - 1e-10) and np.all(factors <= 1.25 + 1e-10)


# ---------------------------------------------------------------------------
# Invariance: row order, column order/units, redundant columns.
# ---------------------------------------------------------------------------


def test_row_and_column_permutations_align_by_identity():
    rng = np.random.default_rng(3)
    n, p = 40, 4
    d = rng.uniform(0.5, 5.0, n)
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    g0 = np.clip(1.0 + 0.2 * rng.normal(size=n), 0.75, 1.35)
    control = x.T @ (d * g0)
    bounds = (0.7, 1.4)
    w_ref = calibrate_bounded(d, x, control, bounds=bounds)

    row_perm = rng.permutation(n)
    w_rows = calibrate_bounded(d[row_perm], x[row_perm], control, bounds=bounds)
    assert np.allclose(w_rows, w_ref[row_perm], rtol=1e-10, atol=1e-8)

    col_perm = rng.permutation(p)
    w_cols = calibrate_bounded(
        d, x[:, col_perm], control[col_perm], bounds=bounds
    )
    assert np.allclose(w_cols, w_ref, rtol=1e-10, atol=1e-8)

    # Unit rescaling of a column (and its total) leaves weights unchanged.
    scales = np.array([1.0, 1e6, 0.01, 10.0])[:p]
    x_scaled = x * scales
    control_scaled = control * scales
    w_scaled = calibrate_bounded(
        d, x_scaled, control_scaled, bounds=bounds
    )
    assert np.allclose(w_scaled, w_ref, rtol=1e-8, atol=1e-8)

    # Redundant (duplicate and linear-combo) columns with consistent totals.
    x_dup = np.column_stack([x, x[:, 0], 2.0 * x[:, 1] - x[:, 2]])
    c_dup = np.concatenate(
        [control, [control[0], 2.0 * control[1] - control[2]]]
    )
    w_dup = calibrate_bounded(d, x_dup, c_dup, bounds=bounds)
    assert np.allclose(w_dup, w_ref, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# Failure classification.
# ---------------------------------------------------------------------------


def test_structural_zeros_are_kept_and_cannot_be_revived():
    d = np.array([1.0, 1.0, 0.0, 0.0])
    x = np.ones((4, 1))
    w = calibrate_bounded(d, x, np.array([3.0]), bounds=(0.5, 2.0))
    assert np.all(w[2:] == 0.0)
    assert np.isclose(x.T @ w, 3.0)


def test_infeasible_target_has_verifiable_farkas_certificate():
    d = np.ones(3)
    x = np.ones((3, 1))
    with pytest.raises(CalibrationError) as exc_info:
        calibrate_bounded(d, x, np.array([100.0]), bounds=(0.5, 2.0))
    err = exc_info.value
    assert err.reason == "infeasible"
    assert err.target == "full"
    assert err.certificate is not None
    y, gap = err.certificate
    # Independently re-check the Farkas inequality on the weighted D X matrix.
    dx = (d.reshape(-1, 1) * x) @ y
    support = np.sum(
        np.maximum(dx, 0.0) * 2.0 + np.minimum(dx, 0.0) * 0.5
    )
    assert y @ np.array([100.0]) - support == gap
    assert gap > 0.0
    assert err.iterations >= 0 and err.residual > 0.0


def test_inconsistent_redundant_column_is_infeasible():
    rng = np.random.default_rng(5)
    n = 30
    d = rng.uniform(1.0, 4.0, n)
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    control = x.T @ d * np.array([1.05, 0.97, 1.02])
    x_bad = np.column_stack([x, x[:, 0]])
    c_bad = np.concatenate([control, [control[0] + 10.0]])
    with pytest.raises(CalibrationError) as exc_info:
        calibrate_bounded(d, x_bad, c_bad)
    assert exc_info.value.reason == "infeasible"
    assert exc_info.value.certificate is not None


def test_iteration_budget_is_never_called_infeasibility():
    rng = np.random.default_rng(21)
    n = 100
    d = rng.uniform(1.0, 10.0, n)
    x = np.column_stack([np.ones(n), rng.normal(scale=3.0, size=(n, 5))])
    lam = 0.05 * np.array([2.0, -1.5, 1.2, -0.8, 1.0, 0.5])
    g_star = 1.0 + x @ lam
    control = x.T @ (d * np.clip(g_star, 0.2, 5.0))
    # A problem known to need several active-set iterations (verified in dev).
    with pytest.raises(CalibrationError) as exc_info:
        calibrate_bounded(d, x, control, bounds=(0.5, 2.0), max_iter=1)
    err = exc_info.value
    assert err.reason == "iteration_limit"
    assert err.certificate is None
    assert err.iterations == 1
    # The same problem converges with the default budget.
    w = calibrate_bounded(d, x, control, bounds=(0.5, 2.0))
    assert np.max(np.abs(x.T @ w - control)) <= 1e-9 * np.max(
        np.maximum(1.0, np.abs(control))
    )


def test_numerical_failure_class_exists_and_singular_does_not_fake_infeasible():
    # A rank-deficient but feasible design must solve, never report infeasible.
    rng = np.random.default_rng(9)
    n = 40
    d = rng.uniform(0.5, 5.0, n)
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    g0 = np.clip(1.0 + 0.1 * rng.normal(size=n), 0.8, 1.2)
    control = x.T @ (d * g0)
    x_dep = np.column_stack([x, x[:, 1], x[:, 2] + x[:, 0]])
    c_dep = np.concatenate(
        [control, [control[1], control[2] + control[0]]]
    )
    w = calibrate_bounded(d, x_dep, c_dep, bounds=(0.75, 1.25))
    assert np.all(np.isfinite(w))
    assert np.max(np.abs(x_dep.T @ w - c_dep)) <= 1e-9 * np.max(
        np.maximum(1.0, np.abs(c_dep))
    )
    # The public reason strings are stable.
    assert CalibrationError.VALID_REASONS == (
        "infeasible",
        "iteration_limit",
        "numerical_failure",
    )


# ---------------------------------------------------------------------------
# Input validation and immutability.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bounds=(2.0, 0.5)),
        dict(bounds=(1.0, 1.0)),
        dict(bounds=(-0.1, 2.0)),
        dict(tol=0.0),
        dict(tol=float("inf")),
        dict(max_iter=0),
        dict(max_iter=True),
    ],
)
def test_illegal_options_raise_value_error(kwargs):
    d = np.ones(4)
    x = np.ones((4, 1))
    with pytest.raises(ValueError):
        calibrate_bounded(d, x, np.array([5.0]), **kwargs)


def test_bad_input_types_and_shapes_raise_value_error():
    d = np.ones(4)
    x = np.ones((4, 1))
    bad_inputs = [
        (np.array([True, False, True, False]), x, np.array([1.0])),
        (d, x.astype(complex), np.array([1.0])),
        (d, x, np.array([1.0 + 0j])),
        (d, np.ones((4, 2, 1)), np.ones(2)),
        (d, np.ones((3, 1)), np.array([1.0])),
        (d, x, np.array([1.0, 2.0])),
        (np.zeros(4), x, np.array([1.0])),
        (d, np.full((4, 1), np.nan), np.array([1.0])),
    ]
    for dd, xx, cc in bad_inputs:
        with pytest.raises(ValueError):
            calibrate_bounded(dd, xx, cc)


def test_inputs_are_not_modified_and_output_is_independent():
    rng = np.random.default_rng(1)
    d = rng.uniform(1.0, 4.0, 8)
    x = np.column_stack([np.ones(8), rng.normal(size=(8, 2))])
    control = x.T @ d * np.array([1.05, 0.95, 1.02])
    d_copy, x_copy, c_copy = d.copy(), x.copy(), control.copy()
    w1 = calibrate_bounded(d, x, control)
    w2 = calibrate_bounded(d, x, control)
    assert np.array_equal(d, d_copy)
    assert np.array_equal(x, x_copy)
    assert np.array_equal(control, c_copy)
    assert np.array_equal(w1, w2)
    w1[0] += 99.0
    assert calibrate_bounded(d, x, control)[0] == w2[0]


def test_no_op_when_controls_already_met():
    rng = np.random.default_rng(2)
    d = rng.uniform(1.0, 4.0, 10)
    x = np.column_stack([np.ones(10), rng.normal(size=(10, 2))])
    control = x.T @ d
    w = calibrate_bounded(d, x, control)
    assert np.array_equal(w, d)


def test_sampleweight_class_method_matches_function():
    rng = np.random.default_rng(4)
    d = rng.uniform(1.0, 4.0, 12)
    x = np.column_stack([np.ones(12), rng.normal(size=(12, 2))])
    control = x.T @ d * np.array([1.04, 0.98, 1.03])
    w_function = calibrate_bounded(d, x, control)
    w_method = SampleWeight.calibrate_bounded(d, x, control)
    assert np.array_equal(w_function, w_method)


def test_numerical_failure_is_distinct_from_infeasibility(monkeypatch):
    # If the independent phase-1 LP itself breaks down, the outcome is a
    # numerical failure, never a claimed infeasibility.
    import samplics.weighting.calibration as calibration_module

    def boom(*args, **kwargs):
        raise ValueError("simulated LP breakdown")

    monkeypatch.setattr(calibration_module, "linprog", boom)
    d = np.ones(3)
    x = np.ones((3, 1))
    with pytest.raises(CalibrationError) as exc_info:
        calibrate_bounded(d, x, np.array([100.0]), bounds=(0.5, 2.0))
    err = exc_info.value
    assert err.reason == "numerical_failure"
    assert err.certificate is None
    assert "simulated LP breakdown" in str(err)


def test_diagnostic_message_documents_iterations_and_residual():
    d = np.ones(3)
    x = np.ones((3, 1))
    with pytest.raises(CalibrationError) as exc_info:
        calibrate_bounded(d, x, np.array([100.0]), bounds=(0.5, 2.0))
    message = str(exc_info.value)
    assert "infeasible" in message
    assert "Farkas" in message
    assert isinstance(exc_info.value.iterations, int)
    assert exc_info.value.residual > 0.0


def test_extreme_column_scales_are_equilibrated():
    rng = np.random.default_rng(0)
    n = 50
    d = rng.uniform(1.0, 5.0, n)
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    x_scaled = x.copy()
    x_scaled[:, 1] = 1.0e120 * x[:, 1]
    control = np.array(
        [
            1.01 * d.sum(),
            (x_scaled[:, 1] * d).sum(),
            1.01 * (x[:, 2] * d).sum(),
        ]
    )
    w = calibrate_bounded(d, x_scaled, control)
    relative = np.abs(x_scaled.T @ w - control) / np.maximum(1.0, np.abs(control))
    assert np.all(relative <= 1e-9)
