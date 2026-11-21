# Bounded calibration of full-sample and replicate weights

This note records the numerical design behind `SampleWeight.calibrate_bounded`,
the calibrated branch of `ReplicateWeight.replicate` and
`ReplicateEstimator.estimate_calibrated`. It explains how the shared auxiliary
matrix is reused across weight vectors that have different zero patterns,
active sets and effective ranks, how infeasibility is distinguished from a
spent budget, and how replicate identities and design coefficients reach the
estimator.

## 1. Problem in factor form

For base weights `d`, auxiliary matrix `X` (n by p) and population totals
`control`, each calibration solves

```
min  sum_{i: d_i > 0} (w_i - d_i)^2 / d_i
s.t. X' w = control,
     lower * d_i <= w_i <= upper * d_i   (d_i > 0),
     w_i = 0                              (d_i = 0).
```

Writing `g_i = w_i / d_i` on the positive rows turns the box into fixed
bounds `[lower, upper]` on `g`, the objective into
`sum d_i (g_i - 1)^2` and the equalities into `(D X)' g = control` with
`D = diag(d_pos)`. Rows with `d_i = 0` are dropped from the solve entirely;
they can never be revived.

## 2. Why an active-set QP, and what is reused

A primal-dual active-set method works directly in factor space. With free set
`F` and the active rows frozen, stationarity `g - 1 = X lambda` and the
equalities give, on every outer iteration,

```
(X_F' D_F X_F) lambda = rhs - X_F' D_F 1,        g_F = 1 + X_F lambda.
```

Only a p-by-p Gram matrix is factorized per iteration, so the working memory is
`O(n p + p^2)` and never `O(n^2)`. There is no n-by-n diagonal matrix and no
nR-by-nR aggregate problem; the R replicate solves are R independent
n-by-p problems that reuse the same *column-equilibrated* auxiliary matrix.

### Shared preprocessing, per-vector working sets

The pieces shared across the full sample and all R replicates are:

* the auxiliary matrix `X` and its per-column equilibration scales
  `s_j = max_i |X_ij|`; dividing column `j` (and total `j`) by `s_j` leaves
  the feasible factor set and the objective unchanged but makes the Gram
  matrix well conditioned even when columns differ by orders of magnitude;
* the control vector and the per-column tolerance
  `tol * max(1, |control_j|)`.

The pieces that deliberately stay **per weight vector** are:

* the positive-row mask (a jackknife/BRR replicate has its own zero pattern);
* the base weights `d^(r)` and hence the Gram matrix `X' D_r X`;
* the lower/upper active sets, the multiplier `lambda^(r)` and the iteration
  counter (`max_iter` is counted separately for the full sample and every
  replicate, as required).

Reusing a constraint factorization across replicates is therefore limited to
the column scaling: because `D_r`, the active set and the effective rank all
change between columns, a cached factorization would be stale almost
immediately. Independent solves are also what make a single failed replicate
attributable and independently retriable without contaminating the others.

### Rank deficiency

Duplicate or linearly dependent columns make `X' D X` singular. The free-row
step uses a minimum-norm least-squares solution (`numpy.linalg.lstsq`), which
returns the unique factors `g = 1 + X lambda` even when `lambda` itself is not
unique. Inconsistent redundant totals (the same column asked to hit two
different controls) are real infeasibilities and are caught below, not hidden
by the least-squares step.

## 3. Optimality and feasibility stopping scales

* **Feasibility** is measured per column against
  `tol * max(1, |control_j|)`.
* **Optimality** uses the KKT residual on the free rows,
  `max |g_F - (1 + X_F lambda)|`, divided by
  `max(1, ||1 + X lambda||_inf)` so the measure is dimensionless. A solve is
  accepted only when both scales are at most `tol`.
* Bound-only feasible points (every row pinned) are handled because the
  active set is allowed to cover all rows; the free block is then empty and
  the controls are checked directly.

## 4. Telling infeasibility apart from "not finished"

The active-set loop can stop for three reasons, mapped onto
`CalibrationError.reason`:

1. `iteration_limit` — the per-vector budget `max_iter` is spent, or the
   working-set cycling guard/stall rule trips. A spent budget is **never**
   upgraded to infeasibility; the exception always carries the current
   residual.
2. `infeasible` — only after an independent phase-1 linear program
   (`scipy.optimize.linprog`, HiGHS) on the *weighted* constraints
   `(D X)' g + u - v = control` proves that the minimum total violation
   exceeds the tolerance. The LP equality dual `y` is returned as a
   checkable Farkas certificate with the positive gap
   `y' control - sum max(DX y, 0) g_hi - sum min(DX y, 0) g_lo`.
3. `numerical_failure` — the phase-1 LP itself raises, or a reconstructed
   point fails its own control check.

A singular Gram matrix, a temporarily stalled residual or an exhausted budget
can therefore never masquerade as a proof of infeasibility.

## 5. Cost of the repeated solves

Each outer active-set iteration is dominated by an `O(|F| p^2)` Gram
assembly and an `O(p^3)` factorization. With `p` small (here `p = 6`) and the
number of bound activations typically in the low tens, the whole batch of
`1 + R` solves stays sub-second at n = 10,000 in the benchmark
(`tests/weighting/benchmark_calibration_scale.py`). Stored state per batch is
the n-by-R weight matrix and the n-by-p auxiliary matrix plus `O(p^2)`
scratch; per-solve bookkeeping (active masks, the visited-set signatures) is
`O(n)` and does not grow with `n^2`.

## 6. How replicate identity and design coefficients flow to the estimator

`ReplicateWeight.replicate` first runs the unchanged generator to obtain the
raw replicate dataframe and its column names (e.g. `_jk_wgt_1`, `_brr_wgt_3`,
`_fay_wgt_2`). The calibrated branch:

1. keeps those column names verbatim as `CalibratedReplicates.replicate_ids`,
   so column `r` always refers to the same design replicate before and after
   calibration;
2. keeps the design's `rep_coefs`, `fay_coef`, `degree_of_freedom` and
   `nb_strata` exactly as generated — calibration changes weights, not the
   sampling design;
3. calibrates the full sample and then each replicate against its own base
   weights and zeros;
4. hands the whole immutable batch to
   `ReplicateEstimator.estimate_calibrated`, which validates method, Fay
   coefficient, coefficient/Fay consistency and the degrees of freedom against
   its own construction before any statistic is computed.

The point estimate uses the calibrated full-sample weights; the variance uses
the calibrated replicate columns with the design coefficients. Domains only
mask the same already-calibrated weights — they are never re-calibrated on a
sub-table. Undefined variances (a zero mean/ratio denominator on the full
sample, any replicate or any reported domain) raise instead of being written
as zero, and a failed call leaves the estimator's previous successful result
intact.
