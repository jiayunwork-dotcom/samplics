# Exact conditional Poisson selection and design-based estimation

This note records the numerical design behind `SelectMethod.conditional_poisson`,
`SampleSelection.prepare_design`, `ConditionalPoissonDesign`,
`SelectionResult`, `SampleWeight.from_selection` and
`TaylorEstimator.estimate_design`. It explains how exact sampling,
first-order probabilities and in-sample second-order queries share one
numerical core, why the chosen representation is used, and how the sample ->
weight -> variance chain stays tied to unit identity.

## 1. Model

Each frame row `i` carries a Bernoulli parameter `p_i` in `(0, 1]`. Inside a
stratum, the independent Bernoulli experiments are conditioned on a fixed
number of selections `n_s`. A legal subset `A` of non-certainty units has
probability proportional to

```
prod_{i in A} p_i / (1 - p_i)
```

and certainty units (parameter exactly `1`) belong to every legal subset. The
result is the conditional Poisson (rejective) design, the maximum-entropy
fixed-size design with the prescribed first-order structure.

Parameter validation, at `prepare_design` time and before any random number is
drawn:

* `samp_unit` is one-dimensional, non-empty and unique; strings or integers.
* `probabilities` are finite, real (booleans rejected), aligned row by row, in
  `(0, 1]`. They are *pre-conditioning* Bernoulli parameters, not the final
  inclusion probabilities.
* Without strata, `samp_size` is an integer.  With strata it is a mapping that
  covers exactly the observed strata, with integer sizes in
  `[1, population]`.
* With `c_i` certainty units and `m_s` optional units in stratum `s`, the
  random part `r_s = n_s - c_s` must satisfy `0 <= r_s <= m_s`; and whenever
  `m_s > 0`, `r_s >= 1`, so no frame unit is left with zero inclusion
  probability.
* `mos`, `samp_rate`, `probs` and `remove_nan=True` are rejected explicitly.

## 2. Shared numerical core

Everything is driven by the elementary symmetric polynomials of the stratum
log-odds `t_i = log(p_i / (1 - p_i))`:

```
e_d(t)  =  sum over d-subsets of prod t_i in the subset.
```

Stored tables (per stratum, indexed by local stratum position, certainty rows
inactive):

* `prefix[k, d] = log e_d(t_i : i < k)` and
* `suffix[k, d] = log e_d(t_i : i >= k)`,

each `(K + 1) x (r + 1)`. They are built by the standard recurrence
`e_d(t_0, ..., t_k) = e_d(t_0, ..., t_{k-1}) + t_k e_{d-1}(t_0, ..., t_{k-1})`,
carried out with `logaddexp`, so products never overflow or underflow and equal
parameters are not treated as a special case.

From these tables the three quantities share one source of truth:

* **First order.** `pi_i = exp(t_i + log e_{r-1}(t_{-i}) - log e_r)`, with
  `log e_{r-1}(t_{-i})` recovered from the prefix and suffix rows in
  `O(r)` per unit.
* **Exact draw.** A single forward walk selects unit `i` with the conditional
  probability `exp(t_i + suffix[i+1, r_remaining - 1] - suffix[i, r_remaining])`.
  This is an exact inverse-CDF walk of the conditional distribution, so the
  number of random numbers is deterministic and the walk uses `O(N r)` time.
* **Second order in sample.** For a queried pair `i < j` in one stratum,
  `pi_ij = exp(t_i + t_j + log e_{r-2}(t excluding i and j) - log e_r)`. The
  middle polynomial is obtained by sweeping the prefix of `i` forward over the
  units before `j`, reusing the stored suffix column at `j + 1`. A query over
  `n` units costs `O(n K r)` time and produces an `n x n` matrix; no `N x N`
  matrix is ever built.

Certainty units use the closed forms `pi_i = 1` and `pi_ij = pi_j`; census
strata have `pi = 1` and `pi_ij = 1`; cross-stratum pairs are products of
first-order probabilities.

## 3. Why this representation

Alternatives considered:

* **Direct products of odds.** Cheap but overflows for parameters near
  `1 - 1e-12` and loses all precision for `1e-12`. Rejected.
* **Normalised odds vectors and sequential division.** Requires dividing
  polynomials by `(1 + odds_i x)`; with coefficients spanning many orders of
  magnitude this loses sign information and produced negative probabilities in
  a prototype. Rejected.
* **Per-query polynomial rebuild from scratch.** Correct but `O(n^2 K r^2)`.
  The sweep above reuses the prefix table and the stored suffix, giving
  `O(n K r)` while keeping the same tables that yield the first-order
  probabilities and the sampler.
* **Caching subset probabilities or an `N x N` pair matrix.** Rejected: the
  design would need `O(N^2)` memory for a quantity only ever queried on the
  sample, and repeated draws would share mutable large arrays.

The two tables are the only "expensive" state: `2 (N + 1)(n + 1)` float64s, so
`N = 2000, n = 80` is about 2.6 MiB. A draw allocates only `O(N)` working
arrays and an `n x n` result matrix is added only when a joint query is made.

## 4. Randomness

`draw(*, random_state=...)` accepts a non-negative integer or a NumPy
`Generator`. An integer builds a fresh `default_rng` and is reproducible; a
`Generator` is advanced in place. The global NumPy state is never read or
modified. Repeated draws reuse the immutable tables and each result owns its
selection arrays, so mutating one result cannot affect another.

## 5. Design-based weights and variance

`SampleWeight.from_selection(result)` returns `1 / pi_i` in result order, where
`pi_i` is the *final* design inclusion probability, never the pre-conditioning
Bernoulli parameter.

`TaylorEstimator.estimate_design(y, selection, *, x=None, domain=None)`:

* `total`: Horvitz-Thompson total `sum y_i / pi_i`.
* `mean`: HT total divided by the design's known frame size. In a domain, it
  divides by the number of frame rows carrying that domain label, so `domain`
  must cover the full frame; the sample domain is looked up by row number.
* `ratio`: `sum w y / sum w x`, overall or by domain, with `sum w x` the
  estimated denominator.

Variance is the stratified Sen-Yates-Grundy estimator

```
V(s) = 1/2 sum_{i != j in s} (pi_i pi_j - pi_ij) / pi_ij
                    * (z_i / pi_i - z_j / pi_j)**2
```

where `z` is `y` for totals, `y` for means (scaled by `1/N^2`), and
`y - R x` for ratios (scaled by the squared estimated denominator). Domain
observations outside the domain enter every domain's linearised variable as
zero. For a fixed-size design `pi_ij <= pi_i pi_j`, so every coefficient is
non-negative; this form avoids catastrophic cancellation.

Certainty units and census strata contribute exactly zero random variance.
A non-census stratum with a single random selection makes the variance
inestimable and the whole call is rejected; no stratum is silently dropped or
zeroed. The same rejection covers zero denominators, invalid input and
over-large confidence requests. On any failure the estimator keeps its last
successful results. Degrees of freedom are the sum over strata of
`random selections - 1`; deterministic strata contribute zero. A degenerate
interval is only allowed when all degrees of freedom are zero *and* the
variance is zero.

`to_dataframe()` reports the usual columns; `mean` here is the design-based
`total / known frame size`, whereas the legacy `estimate()` defines `mean` as
`sum w y / sum w`. The legacy entry point is unchanged.

## 6. Validation summary

* Exhaustive enumeration of every legal subset for 4 to 8 units reproduces the
  sample distribution, first-order probabilities and second-order matrix to
  `rtol=1e-8, atol=1e-11`.
* The design expectation of the HT total, its true design variance, and the
  design expectation of the variance estimator are checked against that same
  enumeration (the variance-estimator expectation matches the true variance to
  `1e-10` relative on well-conditioned examples).
* Equal parameters reduce exactly to simple random sampling; stratified
  designs are independent across strata; certainty units and census strata are
  handled in closed form.
* Fixed-size draws are checked by subset frequency, not just row sums, with a
  stated simulation count and tolerance.
* A `N = 2000, n = 80` single-stratum frame is drawn 10 times with an in-sample
  joint query after each draw; the extra peak memory stays below 96 MiB
  (measured 0.27 MiB on Python 3.11 / NumPy 2.2.2; the two tables themselves
  are about 2.6 MiB).
* Parameters spanning `1e-12` to `1 - 1e-12`, duplicated parameters and
  near-census designs produce no NaN, no negative probabilities and correct
  certainty handling.

Command used for the validation (project editable install in `.venv`):

```
$ .venv/bin/python -m pytest tests/sampling tests/estimation -q
398 passed, 8 xfailed, 2 xpassed
$ .venv/bin/python -m pytest -q
662 passed, 9 skipped, 42 xfailed, 2 xpassed
```

The same new tests were also run under Python 3.12.16 with the project's
locked NumPy 2.2.2 and SciPy 1.15.1:

```
$ /tmp/venv312/bin/python -m pytest tests/sampling/test_conditional_poisson.py \
      tests/sampling/test_conditional_poisson_performance.py tests/estimation -q
215 passed, 3 xfailed, 2 xpassed
```

`tests/sampling/test_selection.py` fails to *collect* under Python 3.12 with
the modular `pandas` versions newer than the lockfile because it mutates a
read-only Series; that is a pre-existing test issue, not part of this change.

