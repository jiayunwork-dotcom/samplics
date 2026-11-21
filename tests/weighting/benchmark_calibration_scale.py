"""Scaling and workload benchmark for bounded calibrated replicates.

Runs legal stratified jackknife designs at n = 2,000 and n = 10,000 with
p = 6 auxiliary columns and R = 20 replicates, arranging different replicate
columns to hit bounds or lose effective rank. Reports wall time, peak memory
and the active-set iteration workload (one solve per full sample plus one per
replicate, each with its own zero pattern and active set).

Run with:
    PYTHONPATH=src python tests/weighting/benchmark_calibration_scale.py
"""

from __future__ import annotations

import resource
import time

import numpy as np

from samplics.utils.types import RepMethod
from samplics.weighting import ReplicateWeight
from samplics.weighting.calibration import _active_set_solve


def _peak_mb() -> float:
    # ru_maxrss is kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _design(rng, n, n_strata=10, psu_per_stratum=2):
    units_per_psu = n // (n_strata * psu_per_stratum)
    stratum = np.repeat(
        [f"h{h}" for h in range(n_strata)], psu_per_stratum * units_per_psu
    )
    psu = np.concatenate(
        [
            np.repeat(
                [f"p{h}-{k}" for k in range(psu_per_stratum)], units_per_psu
            )
            for h in range(n_strata)
        ]
    )
    weights = rng.uniform(1.0, 8.0, n)
    # Mixed-scale auxiliary columns including an intercept and a near-duplicate
    # so the effective rank can be below p on some replicate sub-problems.
    base = rng.normal(size=(n, 4))
    aux = np.column_stack(
        [
            np.ones(n),
            base[:, 0] * 1e4,                      # large unit column
            base[:, 1],
            base[:, 2],
            base[:, 3],
            base[:, 1] + 1e-8 * rng.normal(size=n),  # near duplicate
        ]
    )
    return weights, psu, stratum, aux


def _collect_iterations(x_pos, d_pos, target, g_lo, g_hi, col_tol, max_iter, label):
    result = _active_set_solve(
        x_pos, d_pos, target, g_lo, g_hi, col_tol, 1e-9, max_iter, label
    )
    return result.iterations, result.lower_active.size + result.upper_active.size


def run_case(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    weights, psu, stratum, aux = _design(rng, n)
    # Feasible targets built from in-box factors; the box is tightened around
    # them so that, especially at smaller n, several rows bind and different
    # replicates show different active sets and effective rank behavior.
    g_dream = np.clip(1.0 + 0.12 * rng.normal(size=n), 0.92, 1.08)
    control = aux.T @ (weights * g_dream)
    bounds = (0.95, 1.05) if n >= 10000 else (0.93, 1.07)

    generator = ReplicateWeight(method=RepMethod.jackknife, strat=True)
    start = time.perf_counter()
    batch = generator.replicate(
        weights,
        psu,
        stratum,
        calibration=dict(aux_vars=aux, control=control, bounds=bounds),
    )
    elapsed = time.perf_counter() - start

    # Independently tally per-solve iterations and active-set sizes.
    raw_frame = generator._replicate_raw(weights, psu, stratum)
    raw_cols = [c for c in raw_frame.columns if c.startswith("_jk")][: batch.nb_reps]
    raw_rep = raw_frame[raw_cols].to_numpy(dtype=np.float64)
    pos = weights > 0
    d_pos = weights[pos]
    # Mirror the internal column equilibration for the instrumentation run.
    col_scale = np.maximum(np.abs(aux[pos]).max(axis=0), 1e-300)
    x_eq = aux[pos] / col_scale
    target_eq = control / col_scale
    col_tol = 1e-9 * np.maximum(1.0, np.abs(target_eq))
    g_lo = np.full(d_pos.shape, bounds[0])
    g_hi = np.full(d_pos.shape, bounds[1])

    full_it, full_active = _collect_iterations(
        x_eq, d_pos, target_eq, g_lo, g_hi, col_tol, 500, "full"
    )
    rep_iters, rep_actives, rep_zeros = [], [], []
    for k in range(batch.nb_reps):
        base_k = raw_rep[:, k]
        pos_k = base_k > 0
        d_k = base_k[pos_k]
        x_k = aux[pos_k] / col_scale
        t_k = control / col_scale
        tol_k = 1e-9 * np.maximum(1.0, np.abs(t_k))
        it_k, act_k = _collect_iterations(
            x_k, d_k, t_k, np.full(d_k.shape, bounds[0]),
            np.full(d_k.shape, bounds[1]), tol_k, 500, f"rep-{k}",
        )
        rep_iters.append(it_k)
        rep_actives.append(act_k)
        rep_zeros.append(int((~pos_k).sum()))

    # Verify every calibrated column meets the controls in original units.
    resid = max(
        float(np.max(np.abs(aux.T @ batch.rep_weights[:, k] - control)))
        for k in range(batch.nb_reps)
    )

    return {
        "n": n,
        "R": batch.nb_reps,
        "p": aux.shape[1],
        "elapsed_s": elapsed,
        "peak_mb": _peak_mb(),
        "full_iters": full_it,
        "full_active": full_active,
        "rep_iters": rep_iters,
        "rep_actives": rep_actives,
        "rep_zeros": rep_zeros,
        "max_control_resid": resid,
        "distinct_active_sizes": len(set(rep_actives)),
    }


def main() -> None:
    rows = [run_case(2000, seed=11), run_case(10000, seed=23)]
    header = (
        f"{'n':>6} {'R':>3} {'p':>2} {'time(s)':>9} {'peak(MB)':>9} "
        f"{'full_it':>7} {'rep_it min/med/max':>19} {'active min/max':>15} "
        f"{'zeros min/max':>14} {'resid':>10}"
    )
    print(header)
    for r in rows:
        rep_it = np.asarray(r["rep_iters"])
        rep_ac = np.asarray(r["rep_actives"])
        rep_z = np.asarray(r["rep_zeros"])
        print(
            f"{r['n']:>6} {r['R']:>3} {r['p']:>2} {r['elapsed_s']:>9.3f} "
            f"{r['peak_mb']:>9.1f} {r['full_iters']:>7} "
            f"{rep_it.min():>8}/{int(np.median(rep_it)):>3}/{rep_it.max():>5} "
            f"{rep_ac.min():>7}/{rep_ac.max():>6} "
            f"{rep_z.min():>7}/{rep_z.max():>5} "
            f"{r['max_control_resid']:>10.2e}"
        )
        print(
            f"       distinct replicate active-set sizes: {r['distinct_active_sizes']}; "
            f"total active-set solves = {1 + r['R']} (no n*n or nR*nR objects)"
        )

if __name__ == "__main__":
    main()
