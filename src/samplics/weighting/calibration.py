"""Bounded chi-square distance calibration.

This module solves the bounded calibration problem

    minimize   sum_{i: d_i > 0} (w_i - d_i) ** 2 / d_i
    subject to X.T @ w = control,
               lower * d_i <= w_i <= upper * d_i   for d_i > 0,
               w_i = 0                               for d_i = 0,

where ``d`` is the vector of base (design) weights, ``X`` is the n-by-p
matrix of auxiliary variables and ``control`` holds the p known population
totals.  The algorithm is a primal-dual active-set quadratic program written
in terms of the multiplicative adjustment factors ``g_i = w_i / d_i`` (for
``d_i > 0``), which keeps every linear algebra object of size at most
O(n p) / O(p**2): no n-by-n diagonal matrices are materialized and no
nR-by-nR problem is ever assembled.

A phase-1 linear program (HiGHS through :func:`scipy.optimize.linprog`) is only
invoked when the active-set solver cannot produce a feasible point; it provides
an independently checkable Farkas certificate of infeasibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from scipy.optimize import linprog


__all__ = ["CalibrationError", "CalibrationResult", "calibrate_bounded"]


class CalibrationError(Exception):
    """Raised when a bounded calibration cannot be completed.

    Attributes:
        reason: one of ``"infeasible"``, ``"iteration_limit"`` or
            ``"numerical_failure"``.
        iterations: number of active-set iterations actually performed.
        residual: feasibility residual (maximum absolute control discrepancy)
            observed at the stopping point.
        target: which weight vector failed (``"full"`` or a replicate label).
        certificate: when ``reason == "infeasible"``, a tuple
            ``(y, gap)`` with the Farkas certificate ``y`` such that no point
            inside the box can meet the controls and ``gap`` the positive
            numerical gap; ``None`` otherwise.
    """

    VALID_REASONS = ("infeasible", "iteration_limit", "numerical_failure")

    def __init__(
        self,
        reason: str,
        message: str,
        iterations: int,
        residual: float,
        target: str = "full",
        certificate: Optional[tuple[np.ndarray, float]] = None,
    ) -> None:
        if reason not in self.VALID_REASONS:
            raise ValueError(f"reason must be one of {self.VALID_REASONS}")
        self.reason = reason
        self.iterations = int(iterations)
        self.residual = float(residual)
        self.target = target
        self.certificate = certificate
        super().__init__(message)


@dataclass
class CalibrationResult:
    """Outcome of one bounded calibration solve.

    Attributes:
        factors: multiplicative factors ``g`` on the positive-weight rows.
        iterations: number of active-set iterations consumed.
        feasibility_residual: max absolute discrepancy against the controls.
        optimality_residual: dimensionless KKT/optimality measure at the
            solution (see :func:`calibrate_bounded` for its definition).
        lower_active: rows (indices within the d_i > 0 block) pinned at the
            lower bound.
        upper_active: rows pinned at the upper bound.
    """

    factors: np.ndarray
    iterations: int
    feasibility_residual: float
    optimality_residual: float
    lower_active: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    upper_active: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))


def _as_1d_real(name: str, arr: object) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype.kind == "b":
        raise ValueError(f"{name} must not contain boolean values.")
    if a.dtype.kind == "c":
        raise ValueError(f"{name} must not contain complex values.")
    if not np.issubdtype(a.dtype, np.floating) and not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"{name} must be a numeric array of finite real values.")
    a = a.astype(np.float64, copy=True)
    if a.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain finite values only.")
    return a


def _validate_inputs(
    samp_weight: np.ndarray,
    aux_vars: np.ndarray,
    control: np.ndarray,
    bounds: tuple[float, float],
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Validate and normalize all inputs; raises ValueError on any problem."""

    if isinstance(samp_weight, bool) or isinstance(aux_vars, bool) or isinstance(control, bool):
        raise ValueError("Boolean inputs are not accepted.")

    d = _as_1d_real("samp_weight", samp_weight)
    if np.any(d < 0.0):
        raise ValueError("samp_weight must be non-negative.")
    if not np.any(d > 0.0):
        raise ValueError("samp_weight must contain at least one positive weight.")

    x = np.asarray(aux_vars)
    if x.dtype.kind == "b":
        raise ValueError("aux_vars must not contain boolean values.")
    if x.dtype.kind == "c":
        raise ValueError("aux_vars must not contain complex values.")
    if not np.issubdtype(x.dtype, np.floating) and not np.issubdtype(x.dtype, np.integer):
        raise ValueError("aux_vars must be a numeric matrix of finite real values.")
    x = x.astype(np.float64, copy=True)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError("aux_vars must be a two-dimensional n-by-p matrix.")
    if not np.all(np.isfinite(x)):
        raise ValueError("aux_vars must contain finite values only.")
    n, p = x.shape
    if n <= 0 or p <= 0:
        raise ValueError("aux_vars must have n > 0 rows and p > 0 columns.")
    if d.shape[0] != n:
        raise ValueError("samp_weight and aux_vars have incompatible numbers of rows.")

    ctl = _as_1d_real("control", control)
    if ctl.shape[0] != p:
        raise ValueError("control must have one entry per column of aux_vars.")

    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise ValueError("bounds must be a (lower, upper) pair of finite positive reals.")
    lower, upper = float(bounds[0]), float(bounds[1])
    if not (np.isfinite(lower) and np.isfinite(upper)):
        raise ValueError("bounds must be finite.")
    if lower <= 0.0 or upper <= 0.0:
        raise ValueError("bounds must be positive.")
    if not (lower <= 1.0 <= upper):
        raise ValueError("bounds must satisfy lower <= 1 <= upper.")
    if not (lower < upper):
        raise ValueError("bounds must satisfy lower < upper.")

    if isinstance(tol, bool) or not isinstance(tol, (int, float)):
        raise ValueError("tol must be a finite positive number.")
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and strictly positive.")

    if isinstance(max_iter, bool) or not isinstance(max_iter, int):
        raise ValueError("max_iter must be a non-boolean positive integer.")
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")

    return d, x, ctl, lower, upper, tol


def _farkas_certificate(
    x_pos: np.ndarray,
    d_pos: np.ndarray,
    g_lo: np.ndarray,
    g_hi: np.ndarray,
    target: np.ndarray,
    col_tol: np.ndarray,
    label: str,
) -> Optional[tuple[np.ndarray, float]]:
    """Independently verify infeasibility with a phase-1 LP and Farkas y.

    The calibration constraints in factor form are
    ``(D X).T @ g = target`` with ``D = diag(d_pos)``. The phase-1 program
    minimizes the total violation ``sum(u + v)`` of
    ``(D X).T @ g + u - v = target`` under the box on g. If its optimum stays
    above the per-column tolerances, the equality dual y gives the checkable
    Farkas gap
    ``y @ target - sum max(D X y, 0) * g_hi - sum min(D X y, 0) * g_lo``.
    """

    m, p = x_pos.shape
    dx = d_pos[:, None] * x_pos
    c = np.concatenate([np.zeros(m), np.ones(p), np.ones(p)])
    a_eq = np.block([dx.T, np.eye(p), -np.eye(p)])
    lp_bounds = [(float(g_lo[k]), float(g_hi[k])) for k in range(m)] + [(0.0, None)] * (2 * p)
    try:
        lp = linprog(c=c, A_eq=a_eq, b_eq=target, bounds=lp_bounds, method="highs")
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        raise CalibrationError(
            "numerical_failure",
            f"Phase-1 LP used for the infeasibility check of '{label}' failed: {exc!r}.",
            0,
            float(np.max(np.abs(target))),
            label,
        ) from exc
    if not lp.success:
        return None
    if float(lp.fun) <= float(np.max(col_tol)):
        # A feasible point exists; the active-set QP simply failed to find it.
        return None
    y = np.asarray(lp.eqlin.marginals, dtype=np.float64)
    dxy = dx @ y
    # Support function of the box [g_lo, g_hi] along direction D X y.
    box_support = float(np.sum(np.maximum(dxy, 0.0) * g_hi + np.minimum(dxy, 0.0) * g_lo))
    gap = float(y @ target - box_support)
    if gap > 0.0 and np.all(np.isfinite(y)):
        return y, gap
    return None


def _equality_qp(
    x_f: np.ndarray, d_f: np.ndarray, rhs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the equality-constrained QP on a free set.

    Returns (g_free, lambda). Stationarity of the chi-square objective in
    factors g under the equality ``X_F.T D_F g_F = rhs`` gives
    ``g_f = 1 + X_f @ lambda`` with

        (X_f.T D_f X_f) lambda = rhs - X_f.T D_f 1.

    The Gram matrix may be rank deficient (duplicate or dependent auxiliary
    columns); a minimum-norm least-squares solution is used in that case,
    which yields the unique factors g even when lambda is not unique.
    """

    gdx = d_f[:, None] * x_f
    gram = x_f.T @ gdx
    lam, *_ = np.linalg.lstsq(gram, rhs - gdx.T @ np.ones(x_f.shape[0]), rcond=None)
    g_free = 1.0 + x_f @ lam
    return g_free, lam


def _active_set_solve(
    x_pos: np.ndarray,
    d_pos: np.ndarray,
    target: np.ndarray,
    g_lo: np.ndarray,
    g_hi: np.ndarray,
    col_tol: np.ndarray,
    tol: float,
    max_iter: int,
    label: str,
) -> CalibrationResult:
    """Primal-dual active-set QP on the factor vector g.

    Each iteration solves the equality-constrained QP obtained by freezing the
    active rows, takes a fraction-to-the-boundary step, adds the blocking
    constraint, and releases one active constraint whose reduced multiplier
    has the wrong sign.  A visited-set guard prevents active-set cycling.
    """

    m, p = x_pos.shape
    tol_level = float(np.max(col_tol))

    g = np.clip(np.ones(m), g_lo, g_hi)
    eps_box = 1e-12 * np.maximum(1.0, np.maximum(np.abs(g_lo), np.abs(g_hi)))
    active_lo = g <= g_lo + eps_box
    active_hi = g >= g_hi - eps_box
    free = ~(active_lo | active_hi)

    def residual_of(gg: np.ndarray) -> float:
        return float(np.max(np.abs(x_pos.T @ (d_pos * gg) - target)))

    # Cycling guard: never revisit a working set already encountered.
    visited: set[bytes] = set()
    iterations = 0
    last_residual = residual_of(g)
    stuck = 0

    for iterations in range(1, max_iter + 1):
        free_idx = np.flatnonzero(free)
        active_idx = np.flatnonzero(~free)
        rhs = target - x_pos[active_idx].T @ (d_pos[active_idx] * g[active_idx])

        if free_idx.size == 0:
            g_equality = g.copy()
            lam = np.zeros(p)
            singular = False
        else:
            try:
                g_free, lam = _equality_qp(x_pos[free_idx], d_pos[free_idx], rhs)
                singular = not np.all(np.isfinite(g_free)) or not np.all(np.isfinite(lam))
            except np.linalg.LinAlgError:
                singular = True
            if singular:
                # Extreme numerical degeneracy: pin one row at its current
                # value and retry next round.
                pin = free_idx[-1]
                free[pin] = False
                if g[pin] >= 0.5 * (g_lo[pin] + g_hi[pin]):
                    active_hi[pin] = True
                    g[pin] = g_hi[pin]
                else:
                    active_lo[pin] = True
                    g[pin] = g_lo[pin]
                continue
            g_equality = g.copy()
            g_equality[free_idx] = g_free

        outside = (g_equality < g_lo - eps_box) | (g_equality > g_hi + eps_box)
        if np.any(outside):
            # Fraction-to-boundary step: closest blocking constraint.
            alpha = 1.0
            blocking = -1
            for k in free_idx:
                delta = g_equality[k] - g[k]
                if abs(delta) <= 1e-15:
                    continue
                if g_equality[k] < g_lo[k]:
                    a = (g_lo[k] - g[k]) / delta
                    if 0.0 <= a < alpha:
                        alpha, blocking = a, k
                elif g_equality[k] > g_hi[k]:
                    a = (g_hi[k] - g[k]) / delta
                    if 0.0 <= a < alpha:
                        alpha, blocking = a, k
            g = g + alpha * (g_equality - g)
            g = np.clip(g, g_lo, g_hi)
            if blocking >= 0:
                free[blocking] = False
                if g_equality[blocking] > g_hi[blocking]:
                    active_hi[blocking] = True
                else:
                    active_lo[blocking] = True
        else:
            g = np.clip(g_equality, g_lo, g_hi)
            # Stationarity is g - 1 = X lam, so the reduced multiplier of an
            # active inequality is mu = 1 + X lam - g (positive => active).
            # On free rows KKT requires mu = 0; the same quantity on an active
            # row tells whether the bound should be released.
            mu = (1.0 + x_pos @ lam) - g
            scale = max(1.0, float(np.max(np.abs(1.0 + x_pos @ lam))))
            kkt_free = mu[free_idx] if free_idx.size else np.zeros(0)
            opt_measure = float(np.max(np.abs(kkt_free))) / scale if free_idx.size else 0.0

            # Release at most one active constraint with a wrong-signed
            # multiplier (most violated first). A lower bound g >= g_lo stays
            # active while the objective slope in the only feasible direction
            # (upward) is non-negative, i.e. mu <= 0; release it when mu > 0.
            # The upper-bound multiplier is -mu, so an upper bound g <= g_hi
            # is released when mu < 0.
            release = -1
            lo_candidates = np.flatnonzero(active_lo & (mu > 1e-11))
            if lo_candidates.size:
                release = int(lo_candidates[np.argmax(mu[lo_candidates])])
                active_lo[release] = False
            else:
                hi_candidates = np.flatnonzero(active_hi & (mu < -1e-11))
                if hi_candidates.size:
                    release = int(hi_candidates[np.argmin(mu[hi_candidates])])
                    active_hi[release] = False
            if release >= 0:
                free[release] = True

            feas = residual_of(g)
            if release < 0 and feas <= tol_level and opt_measure <= max(tol, 1e-14):
                return CalibrationResult(
                    factors=g,
                    iterations=iterations,
                    feasibility_residual=feas,
                    optimality_residual=opt_measure,
                    lower_active=np.flatnonzero(active_lo),
                    upper_active=np.flatnonzero(active_hi),
                )

        # Cycling guard on the resulting working set.
        signature = np.packbits(np.concatenate([active_lo, active_hi])).tobytes()
        if signature in visited:
            break
        visited.add(signature)

        cur = residual_of(g)
        if abs(last_residual - cur) <= 1e-14 * max(1.0, last_residual):
            stuck += 1
        else:
            stuck = 0
        last_residual = cur
        if stuck >= max(2 * p + 5, 10):
            break

    feas = residual_of(g)
    budget_exhausted = iterations >= max_iter
    if feas <= tol_level:
        raise CalibrationError(
            "iteration_limit",
            f"Calibration for '{label}' reached max_iter={max_iter} with a feasible "
            f"point (residual={feas:.3e} <= {tol_level:.3e}) but the KKT optimality "
            f"measure did not meet tol={tol:.3e}.",
            iterations,
            feas,
            label,
        )

    # A spent iteration budget can never be called infeasibility: only a
    # stalled/cycled solver is allowed to seek a Farkas certificate.
    if budget_exhausted:
        raise CalibrationError(
            "iteration_limit",
            f"Calibration for '{label}' exhausted max_iter={max_iter} before reaching "
            f"feasibility; residual={feas:.3e} > {tol_level:.3e}.",
            iterations,
            feas,
            label,
        )

    cert = _farkas_certificate(x_pos, d_pos, g_lo, g_hi, target, col_tol, label)
    if cert is not None:
        y, gap = cert
        raise CalibrationError(
            "infeasible",
            f"Calibration for '{label}' is infeasible: the phase-1 linear program "
            f"cannot meet the controls within tolerance (best violation "
            f"{feas:.3e} > {tol_level:.3e}); verified Farkas gap={gap:.3e}.",
            iterations,
            feas,
            label,
            certificate=(y, gap),
        )
    raise CalibrationError(
        "iteration_limit",
        f"Calibration for '{label}' did not converge within max_iter={max_iter}; "
        f"residual={feas:.3e} and no infeasibility certificate could be established.",
        iterations,
        feas,
        label,
    )


def calibrate_bounded(
    samp_weight: np.ndarray,
    aux_vars: np.ndarray,
    control: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.2, 5.0),
    tol: float = 1e-9,
    max_iter: int = 500,
    label: str = "full",
) -> np.ndarray:
    """Bounded chi-square calibration of sample weights.

    Solves, for base weights ``d`` and auxiliary matrix ``X`` (n by p),

    .. math::
        \\min_{w}\\;\\sum_{i:d_i>0}\\frac{(w_i-d_i)^2}{d_i}
        \\quad\\text{s.t.}\\quad
        X^{\\top}w = \\text{control},\\;
        l\\,d_i\\le w_i\\le u\\,d_i,\\;
        w_i=0\\;(d_i=0).

    Args:
        samp_weight: length-n array of finite non-negative base weights; at
            least one must be strictly positive. Rows with ``d_i = 0`` are
            structural zeros and are returned as zero - they can never be
            revived to meet a total.
        aux_vars: n-by-p matrix of finite real auxiliary values. Columns may
            have mixed signs, very different scales, duplicates or be linearly
            dependent.
        control: length-p array of population totals, one per auxiliary
            column.
        bounds: ``(lower, upper)`` finite positive pair with
            ``lower <= 1 <= upper`` and ``lower < upper``.
        tol: strictly positive finite convergence tolerance. The per-column
            feasibility allowance is ``tol * max(1, abs(control[j]))``. The
            same order of tolerance governs the optimality stopping measure
            defined below.
        max_iter: non-boolean positive integer; the active-set iteration
            budget, counted separately for every weight vector.
        label: diagnostic label used in :class:`CalibrationError` messages
            (``"full"`` or a replicate identifier).

    Optimality stopping scale:
        With multipliers ``lambda`` of the control constraints, the KKT
        condition is ``g_i = 1 + x_i @ lambda`` on every free row, where
        ``g_i = w_i / d_i``. The optimality measure is the maximum absolute
        free-row KKT residual divided by
        ``max(1, ||1 + X lambda||_inf)``; it is dimensionless. The solve
        stops only when both the feasibility residual (measured against the
        per-column allowances ``tol * max(1, |control_j|)``) and this
        optimality measure are no larger than ``tol``.

    Returns:
        A new, independent length-n ``float64`` array of calibrated weights.
        The input arrays are never modified.

    Raises:
        ValueError: invalid inputs (booleans, complex or non-finite values,
            mismatched dimensions, illegal options).
        CalibrationError: the solve could not be completed; ``reason`` is
            ``"infeasible"`` (with a checkable Farkas certificate),
            ``"iteration_limit"`` or ``"numerical_failure"``.
    """

    d, x, ctl, lower, upper, tol = _validate_inputs(
        samp_weight, aux_vars, control, bounds, tol, max_iter
    )

    n = d.shape[0]
    out = np.zeros(n, dtype=np.float64)
    pos = d > 0.0

    d_pos = d[pos]
    x_pos = x[pos]
    target = ctl.copy()

    # Column equilibration: rescale every auxiliary column (and its total) to
    # unit max-abs magnitude. The feasible set in factors g and the objective
    # are unchanged, but the Gram matrix X'DX becomes well conditioned even
    # when the original columns have wildly different units.
    col_scale = np.maximum(np.abs(x_pos).max(axis=0), 1e-300)
    x_pos = x_pos / col_scale
    target = target / col_scale
    col_tol = tol * np.maximum(1.0, np.abs(target))

    # Cheap no-op check: unadjusted weights already meet every target.
    if float(np.max(np.abs(x_pos.T @ d_pos - target))) <= float(np.max(col_tol)):
        out[pos] = d_pos
        return out

    g_lo = np.full(d_pos.shape, lower, dtype=np.float64)
    g_hi = np.full(d_pos.shape, upper, dtype=np.float64)

    try:
        result = _active_set_solve(
            x_pos, d_pos, target, g_lo, g_hi, col_tol, tol, max_iter, label=label
        )
    except CalibrationError:
        raise
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise CalibrationError(
            "numerical_failure",
            f"Numerical failure while calibrating '{label}': {exc!r}.",
            0,
            float(np.max(np.abs(x_pos.T @ d_pos - target))),
            label,
        ) from exc

    w_pos = d_pos * result.factors
    resid = float(np.max(np.abs(x_pos.T @ w_pos - target)))
    if resid > float(np.max(col_tol)) + 1e-12:
        raise CalibrationError(
            "numerical_failure",
            f"Reconstructed weights for '{label}' failed the control check "
            f"(residual={resid:.3e} > {float(np.max(col_tol)):.3e}).",
            result.iterations,
            resid,
            label,
        )
    # Enforce the box one last time to suppress rounding-level excursions.
    w_pos = np.clip(w_pos, lower * d_pos, upper * d_pos)
    out[pos] = w_pos
    return out
