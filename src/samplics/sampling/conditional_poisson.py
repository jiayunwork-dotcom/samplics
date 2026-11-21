"""Exact conditional Poisson sampling and its design-based probabilities.

The module provides :class:`ConditionalPoissonDesign` and
:class:`SelectionResult`.  A design is built from a frame of unit identifiers,
a per-unit Bernoulli parameter (the pre-conditioning probability), and a fixed
sample size (globally or by stratum).  Within every stratum the Bernoulli
experiments are conditioned on the specified number of selected units, which
gives the *conditional Poisson* (rejective) design, also known as maximum
entropy sampling with fixed size.

Numerical strategy
------------------
First-order inclusion probabilities and the joint probabilities queried through
:meth:`ConditionalPoissonDesign.joint_inclusion` come from elementary symmetric
polynomials of the stratum log-odds

    t_i = log(p_i / (1 - p_i)).

All polynomial arithmetic is done in log space with ``logaddexp``.  For a
stratum with ``K`` non-certainty units and a random sample size ``r`` the
dynamic programs store

* a prefix table ``P[k, d] = log e_d(t_i : i < k)`` and
* a suffix table ``S[k, d] = log e_d(t_i : i >= k)``,

each of shape ``(K + 1, r + 1)`` when indexed by local stratum position.  Units
declared certain (parameter exactly ``1``) are always selected and are skipped
by the polynomial recurrences; census strata (every unit certain or sample size
equal to the population) are flagged deterministic.

Sampling walks the frame once, at each unit drawing with the exact conditional
probability ``exp(t_i + S[i + 1, r - 1] - S[i, r])``.  This is an exact
inverse-CDF walk of the conditional distribution, not a rejection loop, so the
number of random numbers consumed is deterministic.

Joint probabilities are materialised only for the requested identifiers.  For a
pair ``i < j`` in one stratum the polynomial over the units strictly between
them is accumulated by sweeping the prefix table forward from ``i`` to ``j``.
A query over ``n`` sampled units therefore costs ``O(n K r)`` time and produces
an ``n x n`` matrix, but never constructs an ``N x N`` matrix.
Certainty units use ``pi_i = 1`` and ``pi_ij = pi_j``; census strata have
``pi = 1`` and ``pi_ij = 1``; cross-stratum pairs are products of first-order
probabilities.

Stability and cost
------------------
Log-space arithmetic avoids the overflow/underflow a direct product of odds
suffers when parameters span many orders of magnitude (say ``1e-12`` through
``1 - 1e-12``), and keeps equal or nearly equal parameters from losing
precision.  The main cost is the two ``(K + 1) x (r + 1)`` float64 tables per
stratum: for ``N = 2000`` and ``n = 80`` that is about 2.6 MiB.  A draw costs
``O(N r)`` for the walk plus ``O(n r)`` to assemble the result, and addition of
an ``n x n`` result matrix.  Repeated draws reuse the immutable tables; each
draw owns its own selection arrays and the NumPy global random state is never
touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Union

import numpy as np
from scipy.special import logsumexp

__all__ = ["ConditionalPoissonDesign", "SelectionResult"]


_NEG_INF = -np.inf


def _ordered_unique(values: np.ndarray) -> list:
    """Unique values of ``values`` in first-appearance order, as Python scalars."""
    seen: list = []
    for value in values:
        value = value.item() if hasattr(value, "item") else value
        if value not in seen:
            seen.append(value)
    return seen


def _logsumexp_or_neginf(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).any():
        return float(_NEG_INF)
    return float(logsumexp(values))


# ---------------------------------------------------------------------------
# Elementary symmetric polynomial helpers (all in natural-log space)
# ---------------------------------------------------------------------------


def _extend_logpoly(poly: np.ndarray, log_weight: float) -> np.ndarray:
    """Multiply a log-polynomial by ``(1 + exp(log_weight) x)``."""
    out = poly.copy()
    if poly.size > 1:
        out[1:] = np.logaddexp(poly[1:], log_weight + poly[:-1])
    return out


def _prefix_logpolys(
    log_odds: np.ndarray, active: np.ndarray, degree: int
) -> np.ndarray:
    """``P[k, d] = log e_d`` over active units with local position ``< k``."""
    nb = log_odds.size
    table = np.full((nb + 1, degree + 1), _NEG_INF)
    table[0, 0] = 0.0
    for k in range(nb):
        table[k + 1] = table[k].copy()
        if active[k] and degree >= 1:
            table[k + 1, 1:] = np.logaddexp(
                table[k, 1:], log_odds[k] + table[k, :-1]
            )
    return table


def _suffix_logpolys(
    log_odds: np.ndarray, active: np.ndarray, degree: int
) -> np.ndarray:
    """``S[k, d] = log e_d`` over active units with local position ``>= k``."""
    nb = log_odds.size
    table = np.full((nb + 1, degree + 1), _NEG_INF)
    table[nb, 0] = 0.0
    for k in range(nb - 1, -1, -1):
        table[k] = table[k + 1].copy()
        if active[k] and degree >= 1:
            table[k, 1:] = np.logaddexp(
                table[k + 1, 1:], log_odds[k] + table[k + 1, :-1]
            )
    return table


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SelectionResult:
    """One realised draw of a :class:`ConditionalPoissonDesign`.

    Attributes:
        samp_unit: selected unit identifiers, in original frame order.
        indices: original frame row numbers of the selected units, in the same
            order as ``samp_unit``.
        stratum: stratum label of every selected unit, or ``None`` when the
            design is not stratified.
        design: design that produced the draw.  Keeping the reference makes the
            frame -> sample -> weight -> variance chain traceable by unit
            identity.
    """

    samp_unit: np.ndarray
    indices: np.ndarray
    stratum: Optional[np.ndarray]
    design: "ConditionalPoissonDesign" = field(repr=False)

    def __post_init__(self) -> None:
        self.samp_unit = np.array(self.samp_unit, copy=True)
        self.indices = np.array(self.indices, dtype=np.intp, copy=True)
        if self.stratum is not None:
            self.stratum = np.array(self.stratum, copy=True)

    @property
    def sample_size(self) -> int:
        return int(self.indices.size)

    @property
    def inclusion_probabilities(self) -> np.ndarray:
        """Final first-order inclusion probabilities in sample order."""
        return np.array(
            self.design.inclusion_probabilities[self.indices], dtype=float, copy=True
        )

    def to_dataframe(self):
        """Return the sample as a pandas data frame."""
        import pandas as pd

        data: dict[str, Any] = {
            "_samp_unit": self.samp_unit,
            "_index": self.indices,
            "_probs": self.inclusion_probabilities,
        }
        if self.stratum is not None:
            data["_stratum"] = self.stratum
        return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Internal per-stratum tables
# ---------------------------------------------------------------------------


class _StratumDesign:
    """Immutable tables for one stratum, indexed by local stratum position."""

    __slots__ = (
        "label",
        "positions",
        "sample_size",
        "bernoulli",
        "mandatory",
        "active",
        "log_odds",
        "random_size",
        "log_e_r",
        "prefix",
        "suffix",
        "pi",
        "deterministic",
    )

    def __init__(
        self,
        label: Any,
        positions: np.ndarray,
        bernoulli: np.ndarray,
        sample_size: int,
    ) -> None:
        self.label = label
        self.positions = np.asarray(positions, dtype=np.intp)
        self.sample_size = int(sample_size)
        self.bernoulli = np.array(bernoulli, dtype=float, copy=True)

        self.active = self.bernoulli < 1.0
        self.mandatory = ~self.active
        nb_active = int(self.active.sum())
        nb_mandatory = int(self.mandatory.sum())
        self.random_size = self.sample_size - nb_mandatory
        if self.random_size < 0:
            raise ValueError(
                "samp_size cannot be smaller than the number of certainty units "
                f"in stratum {label!r}."
            )
        if self.random_size > nb_active:
            raise ValueError(
                f"samp_size exceeds the number of non-certainty units in stratum {label!r}."
            )

        self.log_odds = np.zeros(self.positions.size, dtype=float)
        self.log_odds[self.active] = np.log(self.bernoulli[self.active]) - np.log1p(
            -self.bernoulli[self.active]
        )

        degree = self.random_size
        self.prefix = _prefix_logpolys(self.log_odds, self.active, degree)
        self.suffix = _suffix_logpolys(self.log_odds, self.active, degree)
        self.log_e_r = float(self.suffix[0, degree]) if degree <= nb_active else _NEG_INF

        self.pi = np.ones(self.positions.size, dtype=float)
        if nb_active > 0 and self.random_size < nb_active:
            self.pi[self.active] = self._first_order_active()
        self.deterministic = bool(nb_active == 0 or self.random_size == nb_active)

    def _first_order_active(self) -> np.ndarray:
        active_local = np.flatnonzero(self.active)
        nb_active = active_local.size
        r = self.random_size
        prefix = self.prefix
        suffix = self.suffix
        log_e_r = self.log_e_r
        probabilities = np.empty(nb_active)
        for compact, k in enumerate(active_local):
            low = max(0, r - 1 - (nb_active - 1 - compact))
            high = min(r - 1, compact)
            column_low = r - 1 - high
            column_high = r - 1 - low
            terms = (
                prefix[k, low : high + 1]
                + suffix[k + 1, column_low : column_high + 1][::-1]
            )
            probabilities[compact] = np.exp(
                self.log_odds[k] + _logsumexp_or_neginf(terms) - log_e_r
            )
        return probabilities


# ---------------------------------------------------------------------------
# Public design object
# ---------------------------------------------------------------------------


class ConditionalPoissonDesign:
    """A fixed-size conditional Poisson design over a finite frame.

    Instances are produced by :meth:`samplics.sampling.SampleSelection.prepare_design`.
    They are immutable with respect to the caller's arrays: the frame, the
    stratum labels and the Bernoulli parameters are copied on construction, so
    later mutation of the original arrays has no effect.
    """

    def __init__(
        self,
        samp_unit: np.ndarray,
        probabilities: np.ndarray,
        sample_size: Union[int, Mapping[Any, int]],
        stratum: Optional[np.ndarray] = None,
        *,
        stratified: bool = False,
    ) -> None:
        self.samp_unit = np.array(samp_unit, copy=True)
        self.probabilities = np.array(probabilities, dtype=float, copy=True)
        self.stratum = None if stratum is None else np.array(stratum, copy=True)
        self.stratified = bool(stratified)
        self.frame_size = int(self.samp_unit.size)
        self._id_to_position = {key: pos for pos, key in enumerate(self.samp_unit)}
        self._sample_size: Union[int, dict[Any, int]]

        if self.stratum is None:
            labels = np.zeros(self.frame_size, dtype=np.intp)
            self._sample_size = int(sample_size)  # type: ignore[arg-type]
        else:
            labels = self.stratum
            self._sample_size = {
                key: int(value) for key, value in sample_size.items()  # type: ignore[union-attr]
            }

        self._strata: list[_StratumDesign] = []
        for key in _ordered_unique(labels):
            mask = labels == key
            position = np.flatnonzero(mask)
            size = (
                self._sample_size
                if not isinstance(self._sample_size, dict)
                else self._sample_size[key]
            )
            self._strata.append(
                _StratumDesign(
                    label=key,
                    positions=position,
                    bernoulli=self.probabilities[position],
                    sample_size=size,
                )
            )

        pi = np.empty(self.frame_size, dtype=float)
        self._position_to_stratum = np.empty(self.frame_size, dtype=np.intp)
        self._position_to_local = np.empty(self.frame_size, dtype=np.intp)
        for index, stratum_design in enumerate(self._strata):
            pi[stratum_design.positions] = stratum_design.pi
            self._position_to_stratum[stratum_design.positions] = index
            self._position_to_local[stratum_design.positions] = np.arange(
                stratum_design.positions.size
            )
        self._pi = pi

    # -- basic properties ---------------------------------------------------

    @property
    def inclusion_probabilities(self) -> np.ndarray:
        """Final first-order inclusion probabilities in original frame order."""
        return np.array(self._pi, dtype=float, copy=True)

    @property
    def sample_size(self) -> Union[int, dict[Any, int]]:
        return (
            int(self._sample_size)
            if not isinstance(self._sample_size, dict)
            else dict(self._sample_size)
        )

    @property
    def strata(self) -> list[Any]:
        return [stratum.label for stratum in self._strata]

    @property
    def random_sample_sizes(self) -> dict[Any, int]:
        """Number of non-certainty units selected in each stratum."""
        return {stratum.label: stratum.random_size for stratum in self._strata}

    @property
    def deterministic_strata(self) -> list[Any]:
        """Strata in which every unit is selected with probability one."""
        return [stratum.label for stratum in self._strata if stratum.deterministic]

    def is_deterministic(self, label: Any) -> bool:
        return self.stratum_design(label).deterministic

    def stratum_design(self, label: Any) -> _StratumDesign:
        for stratum_design in self._strata:
            if stratum_design.label == label:
                return stratum_design
        raise KeyError(label)

    def __len__(self) -> int:
        return self.frame_size

    def __repr__(self) -> str:
        return (
            f"ConditionalPoissonDesign(n={self.frame_size}, "
            f"strata={len(self._strata)})"
        )

    # -- drawing ------------------------------------------------------------

    def draw(
        self, *, random_state: Optional[Union[int, np.random.Generator]] = None
    ) -> SelectionResult:
        """Draw one sample from the design.

        Args:
            random_state: non-negative integer seed or a NumPy ``Generator``.
                An integer seed creates a fresh generator and is reproducible;
                a ``Generator`` is advanced in place.  The NumPy global random
                state is never used or modified.

        Returns:
            SelectionResult with selected units in original frame order.
        """
        rng = _as_generator(random_state)

        parts = [
            stratum_design.positions[_draw_stratum(stratum_design, rng)]
            for stratum_design in self._strata
        ]
        selected_positions = (
            np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=np.intp)
        )
        stratum_values = (
            None
            if self.stratum is None
            else np.array(self.stratum[selected_positions], copy=True)
        )
        return SelectionResult(
            samp_unit=np.array(self.samp_unit[selected_positions], copy=True),
            indices=selected_positions,
            stratum=stratum_values,
            design=self,
        )

    # -- design-based variance ingredients ----------------------------------

    def _stratum_joint_matrix(self, positions: np.ndarray) -> np.ndarray:
        """Exact ``pi_ij`` matrix for frame positions inside one stratum."""
        positions = np.asarray(positions, dtype=np.intp)
        if positions.size == 0:
            return np.zeros((0, 0))
        stratum_index = self._position_to_stratum[positions]
        if np.unique(stratum_index).size != 1:
            raise ValueError("Positions must belong to a single stratum.")
        stratum_design = self._strata[int(stratum_index[0])]
        local = self._position_to_local[positions]
        matrix = np.zeros((positions.size, positions.size))
        pi = self._pi[positions]
        matrix[np.diag_indices(positions.size)] = pi
        if stratum_design.deterministic:
            matrix[:] = 1.0
            return matrix
        index = np.arange(positions.size)
        self._fill_same_stratum(matrix, index, local, stratum_design, pi)
        np.fill_diagonal(matrix, pi)
        return matrix

    def sen_yates_grundy_terms(
        self, positions: np.ndarray, values: np.ndarray
    ) -> list[tuple[float, int]]:
        """Per-stratum Sen-Yates-Grundy variance contributions.

        Args:
            positions: frame positions of the observed units (any order).
            values: linearised values aligned with ``positions``.

        Returns:
            List of ``(variance_estimate, random_count)`` tuples, one per
            stratum that contains at least two random selections.  Each
            contribution is

                ``1/2 * sum_{i != j} (pi_i pi_j - pi_ij) / pi_ij *
                (z_i / pi_i - z_j / pi_j)**2``

            with ``z`` the supplied linearised value.  For a fixed-size design
            ``pi_ij <= pi_i pi_j``, so every coefficient is non-negative and
            the pairwise-difference form is numerically stable.  Deterministic
            (census/certainty-only) strata are omitted: they contribute exactly
            zero random variance.
        """
        positions = np.asarray(positions, dtype=np.intp)
        values = np.asarray(values, dtype=float)
        if positions.size != values.size:
            raise ValueError("positions and values must have the same length.")
        if positions.size == 0:
            return []

        stratum_index = self._position_to_stratum[positions]
        results: list[tuple[float, int]] = []
        for index, stratum_design in enumerate(self._strata):
            if stratum_design.deterministic:
                continue
            mask = stratum_index == index
            if not mask.any():
                continue
            stratum_positions = positions[mask]
            stratum_values = values[mask]
            # Certainty units have pi_i = 1 and pi_ij = pi_j, so every pair
            # involving them has a zero coefficient; they are kept for clarity.
            joint = self._stratum_joint_matrix(stratum_positions)
            pi = joint.diagonal()
            if (pi <= 0.0).any():
                raise ValueError("Encountered a non-positive inclusion probability.")
            if (joint <= 0.0).any():
                # Both units were observed, so pi_ij is strictly positive; a
                # zero here means numerical underflow of an extremely rare pair.
                raise ValueError(
                    "A sampled pair has an unusable joint inclusion probability."
                )
            weights = stratum_values / pi
            numerator = np.outer(pi, pi) - joint
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                coefficients = numerator / joint
            np.fill_diagonal(coefficients, 0.0)
            if not np.all(np.isfinite(coefficients)):
                raise ValueError("Variance coefficients are not finite.")
            differences = weights[:, None] - weights[None, :]
            quadratic = 0.5 * float(np.sum(coefficients * differences**2))
            if not np.isfinite(quadratic):
                raise ValueError("The variance contribution is not finite.")
            results.append((max(quadratic, 0.0), stratum_design.random_size))
        return results

    def syg_variance(self, positions: np.ndarray, values: np.ndarray) -> float:
        """Total Sen-Yates-Grundy variance across strata.

        Every non-deterministic stratum must have at least two random
        selections; otherwise the variance is not estimable and ``ValueError``
        is raised.  This is a property of the design, so it is checked for all
        such strata, whether or not the supplied positions cover them.
        """
        positions = np.asarray(positions, dtype=np.intp)
        if positions.size == 0:
            raise ValueError("At least one sampled unit is required.")
        for index, stratum_design in enumerate(self._strata):
            if stratum_design.deterministic:
                continue
            if stratum_design.random_size < 2:
                raise ValueError(
                    "Variance is not estimable: a non-census stratum has a single "
                    f"randomly selected unit (stratum {stratum_design.label!r})."
                )
        terms = self.sen_yates_grundy_terms(positions, values)
        total = float(sum(term for term, _ in terms))
        if not np.isfinite(total):
            raise ValueError("The variance estimate is not finite.")
        return total

    def design_degrees_of_freedom(self, positions: np.ndarray) -> int:
        """Sum over strata of ``random selections - 1`` (deterministic = 0)."""
        positions = np.asarray(positions, dtype=np.intp)
        stratum_index = self._position_to_stratum[positions]
        observed = set(stratum_index.tolist())
        degree = 0
        for index, stratum_design in enumerate(self._strata):
            if stratum_design.deterministic:
                continue
            if index in observed:
                degree += max(stratum_design.random_size - 1, 0)
        return degree

    # -- joint inclusion ----------------------------------------------------

    def joint_inclusion(self, ids: Any) -> np.ndarray:
        """Return the second-order inclusion matrix for ``ids``.

        Rows and columns follow ``ids`` exactly.  The diagonal holds first-order
        inclusion probabilities.  Pairs from different strata are products of
        their first-order probabilities.  Unknown and repeated identifiers are
        rejected with ``ValueError``.
        """
        query = self._query_positions(ids)
        size = query.size
        matrix = np.zeros((size, size), dtype=float)
        if size == 0:
            return matrix

        stratum_of = self._position_to_stratum[query]
        local_of = self._position_to_local[query]
        pi = self._pi[query]
        matrix[np.diag_indices(size)] = pi

        for index, stratum_design in enumerate(self._strata):
            in_stratum = np.flatnonzero(stratum_of == index)
            if in_stratum.size >= 2:
                self._fill_same_stratum(
                    matrix, in_stratum, local_of, stratum_design, pi
                )

        cross = stratum_of[:, None] != stratum_of[None, :]
        matrix[cross] = (pi[:, None] * pi[None, :])[cross]
        np.fill_diagonal(matrix, pi)
        return matrix

    def _fill_same_stratum(
        self,
        matrix: np.ndarray,
        in_stratum: np.ndarray,
        local_of: np.ndarray,
        stratum_design: _StratumDesign,
        pi: np.ndarray,
    ) -> None:
        if stratum_design.deterministic:
            for offset_a, a in enumerate(in_stratum):
                for b in in_stratum[offset_a + 1 :]:
                    matrix[a, b] = matrix[b, a] = 1.0
            return

        local = local_of[in_stratum]
        mandatory = stratum_design.mandatory[local]

        # Pairs involving a certainty unit are known in closed form.
        for offset_a, a in enumerate(in_stratum):
            for offset_b in range(offset_a + 1, in_stratum.size):
                b = in_stratum[offset_b]
                if mandatory[offset_a] and mandatory[offset_b]:
                    matrix[a, b] = matrix[b, a] = 1.0
                elif mandatory[offset_a]:
                    matrix[a, b] = matrix[b, a] = pi[b]
                elif mandatory[offset_b]:
                    matrix[a, b] = matrix[b, a] = pi[a]

        active_mask = ~mandatory
        if not active_mask.any():
            return
        positions = in_stratum[active_mask]
        active_local = local[active_mask]
        order = np.argsort(active_local)
        positions = positions[order]
        active_local = active_local[order]

        r = stratum_design.random_size
        degree = r - 2
        if degree < 0:
            # At most one random selection: no two active units co-occur.
            for a in range(positions.size):
                for b in range(a + 1, positions.size):
                    matrix[positions[a], positions[b]] = 0.0
                    matrix[positions[b], positions[a]] = 0.0
            return

        logE = stratum_design.log_e_r
        log_odds = stratum_design.log_odds
        active = stratum_design.active
        prefix = stratum_design.prefix
        suffix_columns = stratum_design.suffix[:, degree::-1]
        nb_local = log_odds.size
        queried_local = set(active_local.tolist())
        local_to_query = {int(local): int(pos) for local, pos in zip(active_local, positions)}

        # For pair (a, b) the polynomial runs over active units strictly
        # between their local positions.  Starting from the prefix table of a
        # and sweeping forward one unit at a time makes a full query
        # O(n * K * r) time while never touching the N x N space.
        for offset_a in range(positions.size - 1):
            a = int(positions[offset_a])
            local_a = int(active_local[offset_a])
            running = prefix[local_a, : degree + 1].copy()
            for unit in range(local_a + 1, nb_local):
                if unit in queried_local:
                    b = local_to_query[unit]
                    terms = running + suffix_columns[unit + 1]
                    value = np.exp(
                        log_odds[local_a]
                        + log_odds[unit]
                        + _logsumexp_or_neginf(terms)
                        - logE
                    )
                    value = float(np.clip(value, 0.0, 1.0))
                    matrix[a, b] = matrix[b, a] = value
                if active[unit]:
                    running = _extend_logpoly(running, log_odds[unit])

    # -- helpers ------------------------------------------------------------

    def _query_positions(self, ids: Any) -> np.ndarray:
        values = np.asarray(ids)
        if values.ndim != 1:
            raise ValueError("ids must be a one-dimensional array of unit identifiers.")
        resolved = np.empty(values.size, dtype=np.intp)
        seen: set[Any] = set()
        for index, value in enumerate(values):
            key = value.item() if hasattr(value, "item") else value
            if isinstance(key, (bool, np.bool_)):
                raise ValueError("Unit identifiers cannot be booleans.")
            if key in seen:
                raise ValueError(f"Duplicate unit identifier: {key!r}.")
            seen.add(key)
            try:
                resolved[index] = self._id_to_position[key]
            except (KeyError, TypeError):
                raise ValueError(f"Unknown unit identifier: {key!r}.") from None
        return resolved


# ---------------------------------------------------------------------------
# Low-level sampling and generator handling
# ---------------------------------------------------------------------------


def _as_generator(
    random_state: Optional[Union[int, np.random.Generator]],
) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise ValueError(
            "random_state must be a non-negative integer or a NumPy Generator."
        )
    if int(random_state) < 0:
        raise ValueError("random_state must be a non-negative integer.")
    return np.random.default_rng(int(random_state))


def _draw_stratum(stratum_design: _StratumDesign, rng: np.random.Generator) -> np.ndarray:
    """Return the local positions selected inside one stratum."""
    mandatory = stratum_design.mandatory
    if stratum_design.deterministic:
        return np.flatnonzero(np.ones(mandatory.size, dtype=bool))

    active_local = np.flatnonzero(stratum_design.active)
    random_size = stratum_design.random_size
    suffix = stratum_design.suffix
    log_odds = stratum_design.log_odds

    selected_active = np.zeros(active_local.size, dtype=bool)
    remaining = random_size
    for compact, unit in enumerate(active_local):
        if remaining == 0:
            break
        if remaining == active_local.size - compact:
            selected_active[compact:] = True
            break
        conditional = np.exp(
            log_odds[unit] + suffix[unit + 1, remaining - 1] - suffix[unit, remaining]
        )
        conditional = float(np.clip(conditional, 0.0, 1.0))
        if rng.random() < conditional:
            selected_active[compact] = True
            remaining -= 1

    selected = mandatory.copy()
    selected[active_local] = selected_active
    return np.flatnonzero(selected)
