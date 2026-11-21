"""Tests for exact conditional Poisson selection and its design probabilities."""

import itertools
import warnings

import numpy as np
import pytest
from scipy.special import logsumexp

from samplics.sampling import (ConditionalPoissonDesign, SampleSelection,
                               SelectionResult)
from samplics.utils.errors import MethodError
from samplics.utils.types import SelectMethod

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Independent reference: exhaustive enumeration of all legal subsets
# ---------------------------------------------------------------------------


def _enumerate_reference(probabilities, sample_size, mandatory):
    """Return subset probabilities and exact first/second-order truth.

    Only the non-certainty units are enumerated; the legal subsets are the
    combinations of that size among them.  The reference normalises the
    ``prod odds`` weights independently of any code under test.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    optional = np.flatnonzero(probabilities < 1.0)
    odds = probabilities[optional] / (1.0 - probabilities[optional])
    r = sample_size - int(mandatory.sum())
    subsets = list(itertools.combinations(range(optional.size), r))
    weights = np.array([np.prod(odds[list(subset)]) for subset in subsets])
    probabilities_subset = weights / weights.sum()

    first_order = np.zeros(optional.size)
    second_order = np.zeros((optional.size, optional.size))
    for subset, weight in zip(subsets, probabilities_subset):
        for i in subset:
            first_order[i] += weight
            for j in subset:
                second_order[i, j] += weight
    return subsets, probabilities_subset, first_order, second_order, optional


def _reference_frame(probabilities, sample_size, mandatory=None):
    """Expand the compact reference to the full frame, including certainties."""
    probabilities = np.asarray(probabilities, dtype=float)
    sample_size = int(sample_size)
    optional = np.flatnonzero(probabilities < 1.0)
    mandatory_mask = probabilities == 1.0
    _, _, first_optional, second_optional, _ = _enumerate_reference(
        probabilities, sample_size, mandatory_mask
    )
    first = np.ones(probabilities.size)
    first[optional] = first_optional
    second = np.ones((probabilities.size, probabilities.size))
    second[np.ix_(optional, optional)] = second_optional
    for a in range(probabilities.size):
        for b in range(probabilities.size):
            if a == b:
                continue
            if mandatory_mask[a] and not mandatory_mask[b]:
                second[a, b] = second[b, a] = first[b]
            elif mandatory_mask[b] and not mandatory_mask[a]:
                second[a, b] = second[b, a] = first[a]
    np.fill_diagonal(second, first)
    return first, second


def _assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-11)


# ---------------------------------------------------------------------------
# Exhaustive subset validation
# ---------------------------------------------------------------------------


WELL_CONDITIONED = {
    "mild": (np.array([0.10, 0.25, 0.40, 0.55, 0.70, 0.85]), 3),
    "equal": (np.array([0.35, 0.35, 0.35, 0.35, 0.35]), 2),
    "some_certainty": (np.array([1.0, 0.20, 0.40, 0.60, 0.80, 1.0]), 3),
    "one_random": (np.array([0.20, 0.40, 0.60, 0.80]), 1),
}


@pytest.mark.parametrize("name", list(WELL_CONDITIONED))
def test_design_probabilities_match_enumeration(name):
    probabilities, sample_size = WELL_CONDITIONED[name]
    units = np.arange(probabilities.size) * 7 + 3
    selection = SampleSelection(SelectMethod.conditional_poisson)
    design = selection.prepare_design(units, sample_size, probabilities=probabilities)

    first_true, second_true = _reference_frame(probabilities, sample_size)
    _assert_close(design.inclusion_probabilities, first_true)
    _assert_close(design.joint_inclusion(units), second_true)

    # Sum of first-order probabilities equals the fixed sample size.
    assert design.inclusion_probabilities.sum() == pytest.approx(sample_size, abs=1e-10)


def test_exact_subset_frequencies_match_odds_products():
    """The whole conditional distribution, not just its margins, is correct."""
    probabilities = np.array([0.15, 0.30, 0.45, 0.65, 0.85])
    sample_size = 3
    selection = SampleSelection(SelectMethod.conditional_poisson)
    design = selection.prepare_design(np.arange(5), sample_size, probabilities=probabilities)

    subsets, distribution, _, _, _ = _enumerate_reference(
        probabilities, sample_size, np.zeros(5, dtype=bool)
    )
    lookup = {subset: probability for subset, probability in zip(subsets, distribution)}

    n_simulations = 120_000
    generator = np.random.default_rng(4242)
    counts = {subset: 0 for subset in subsets}
    for _ in range(n_simulations):
        result = design.draw(random_state=generator)
        counts[tuple(result.indices.tolist())] += 1

    for subset in subsets:
        observed = counts[subset] / n_simulations
        expected = lookup[subset]
        standard_error = np.sqrt(expected * (1 - expected) / n_simulations)
        # Four sigma; the enumerator has 10 cells and fixed-size dependence.
        assert abs(observed - expected) < 4.0 * standard_error + 1e-12


def test_equal_parameters_reduce_to_simple_random_sampling():
    size = 7
    sample_size = 3
    probabilities = np.full(size, 0.42)
    selection = SampleSelection(SelectMethod.conditional_poisson)
    design = selection.prepare_design(np.arange(size), sample_size, probabilities=probabilities)

    _assert_close(design.inclusion_probabilities, np.full(size, sample_size / size))
    joint = design.joint_inclusion(np.arange(size))
    expected = sample_size * (sample_size - 1) / (size * (size - 1))
    off_diagonal = joint[~np.eye(size, dtype=bool)]
    _assert_close(off_diagonal, np.full(off_diagonal.size, expected))


def test_stratified_design_is_independent_across_strata():
    probabilities = np.array([0.10, 0.40, 0.70, 0.20, 0.50, 0.80, 0.30, 0.60])
    strata = np.array(["a", "a", "a", "b", "b", "b", "c", "c"])
    sizes = {"a": 2, "b": 2, "c": 2}
    selection = SampleSelection(SelectMethod.conditional_poisson)
    design = selection.prepare_design(
        np.arange(8), sizes, probabilities=probabilities, stratum=strata
    )

    first_true, second_true = _reference_frame(probabilities, 0)
    for label in np.unique(strata):
        mask = strata == label
        index = np.flatnonzero(mask)
        reference_first, reference_second = _reference_frame(
            probabilities[mask], sizes[label]
        )
        _assert_close(design.inclusion_probabilities[mask], reference_first)
        _assert_close(design.joint_inclusion(index), reference_second)

    joint = design.joint_inclusion(np.arange(8))
    first = design.inclusion_probabilities
    for a in range(8):
        for b in range(8):
            if strata[a] != strata[b]:
                assert joint[a, b] == pytest.approx(first[a] * first[b], rel=1e-12)


def test_stratified_design_with_integer_labels():
    probabilities = np.array([0.1, 0.4, 0.7, 0.2, 0.5, 0.8])
    strata = np.array([1, 1, 1, 2, 2, 2])
    selection = SampleSelection(SelectMethod.conditional_poisson)
    design = selection.prepare_design(
        np.arange(6), {1: 2, 2: 2}, probabilities=probabilities, stratum=strata
    )
    assert design.strata == [1, 2]
    for label in (1, 2):
        mask = strata == label
        _assert_close(
            design.inclusion_probabilities[mask],
            _reference_frame(probabilities[mask], 2)[0],
        )


def test_certainty_units_and_census_stratum():
    # Frame: three certainty census units, three mixed units.
    probabilities = np.array([1.0, 1.0, 0.3, 0.5, 0.7, 1.0])
    labels = np.array(["census", "census", "mixed", "mixed", "mixed", "census"])
    selection = SampleSelection(SelectMethod.conditional_poisson)
    design = selection.prepare_design(
        np.arange(6),
        {"census": 3, "mixed": 2},
        probabilities=probabilities,
        stratum=labels,
    )
    # The census stratum selects all three units with probability one.
    assert design.is_deterministic("census")
    _assert_close(design.inclusion_probabilities[[0, 1, 5]], np.ones(3))
    # The mixed stratum draws two of three; its units are not certain.
    assert not design.is_deterministic("mixed")
    assert design.inclusion_probabilities[4] < 1.0
    _assert_close(
        design.inclusion_probabilities[2:5],
        _reference_frame(probabilities[2:5], 2)[0],
    )
    # A fully sampled stratum is deterministic.
    full = selection.prepare_design(
        np.arange(4), 4, probabilities=np.array([0.1, 0.4, 0.7, 0.9])
    )
    _assert_close(full.inclusion_probabilities, np.ones(4))

    # Certainty units mixed with random ones: pair rules hold.
    mixed = selection.prepare_design(
        np.arange(4), 3, probabilities=np.array([1.0, 0.2, 0.5, 0.9])
    )
    first = mixed.inclusion_probabilities
    joint = mixed.joint_inclusion(np.arange(4))
    assert first[0] == 1.0
    _assert_close(joint[0, 1:], first[1:])
    _assert_close(joint[1:, 1:], _reference_frame(np.array([0.2, 0.5, 0.9]), 2)[1])


# ---------------------------------------------------------------------------
# Stability for extreme parameters
# ---------------------------------------------------------------------------


def test_extreme_parameters_are_finite_and_monotone():
    probabilities = np.array([1e-12, 1e-8, 0.01, 0.2, 0.5, 0.9, 0.999999, 1 - 1e-12])
    selection = SampleSelection(SelectMethod.conditional_poisson)
    design = selection.prepare_design(np.arange(8), 3, probabilities=probabilities)

    first = design.inclusion_probabilities
    assert np.all(np.isfinite(first))
    assert np.all(first > 0.0) and np.all(first < 1.0)
    assert np.all(np.diff(first) > 0.0)  # monotone ordering is preserved
    assert first.sum() == pytest.approx(3.0, rel=1e-12)

    joint = design.joint_inclusion(np.arange(8))
    assert np.all(np.isfinite(joint))
    assert np.all(joint >= 0.0)
    assert np.allclose(np.diag(joint), first)
    assert np.allclose(joint, joint.T)
    # pi_ij <= pi_i pi_j for a fixed-size design, up to round-off relative to
    # the product scale (which can be as small as 1e-26 here).  The diagonal
    # equals pi_i by definition and is checked separately.
    strict = joint[~np.eye(8, dtype=bool)]
    bound = np.outer(first, first)[~np.eye(8, dtype=bool)]
    np.testing.assert_array_equal(strict <= bound * (1 + 1e-12) + 1e-300, True)

    first_true, second_true = _reference_frame(probabilities, 3)
    np.testing.assert_allclose(first, first_true, rtol=1e-8, atol=1e-25)
    np.testing.assert_allclose(joint, second_true, rtol=1e-8, atol=1e-25)


def test_repeated_and_near_census_parameters():
    probabilities = np.array([0.999999999999] * 3 + [0.2, 0.2, 0.4])
    selection = SampleSelection(SelectMethod.conditional_poisson)
    design = selection.prepare_design(np.arange(6), 3, probabilities=probabilities)
    first = design.inclusion_probabilities
    assert np.all(np.isfinite(first))
    assert first.sum() == pytest.approx(3.0, rel=1e-11)
    # The three duplicated high parameters are nearly forced.
    assert np.all(first[:3] > 0.999)

    # All-parameter-equal near census.
    equal = np.full(5, 0.9999)
    design_equal = selection.prepare_design(np.arange(5), 4, probabilities=equal)
    _assert_close(design_equal.inclusion_probabilities, np.full(5, 0.8))

    # Product underflow would occur with the raw odds; log space must not.
    tiny = np.array([1e-12] * 4 + [0.5, 0.5])
    design_tiny = selection.prepare_design(np.arange(6), 3, probabilities=tiny)
    joint = design_tiny.joint_inclusion(np.arange(6))
    assert np.all(np.isfinite(joint))
    assert joint[0, 1] >= 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _bad_cases():
    units = np.arange(6)
    probs = np.array([0.2, 0.4, 0.6, 0.8, 0.5, 0.9])
    return [
        # probabilities out of range / zero / NaN / bool
        dict(samp_unit=units, samp_size=3, probabilities=np.array([0.0, .4, .6, .8, .5, .9])),
        dict(samp_unit=units, samp_size=3, probabilities=np.array([-.1, .4, .6, .8, .5, .9])),
        dict(samp_unit=units, samp_size=3, probabilities=np.array([1.1, .4, .6, .8, .5, .9])),
        dict(samp_unit=units, samp_size=3, probabilities=np.array([np.nan, .4, .6, .8, .5, .9])),
        dict(samp_unit=units, samp_size=3, probabilities=np.array([True, False, True, False, True, False])),
        dict(samp_unit=units, samp_size=3, probabilities=[True, 0.5, 0.5, 0.5, 0.5, 0.5]),
        dict(samp_unit=units, samp_size=3, probabilities=probs[:-1]),
        # duplicate / empty / 2-D unit ids
        dict(samp_unit=np.array([1, 2, 2, 4]), samp_size=2, probabilities=np.array([.2, .4, .6, .8])),
        dict(samp_unit=np.array([]), samp_size=1, probabilities=np.array([])),
        # sample size problems
        dict(samp_unit=units, samp_size=0, probabilities=probs),
        dict(samp_unit=units, samp_size=7, probabilities=probs),
        dict(samp_unit=units, samp_size=3.0, probabilities=probs),
        dict(samp_unit=units, samp_size=True, probabilities=probs),
        # sampling rate / mos / probs conflicts
        dict(samp_unit=units, samp_size=3, probabilities=probs, samp_rate=0.5),
        dict(samp_unit=units, samp_size=3, probabilities=probs, mos=np.ones(6)),
        dict(samp_unit=units, samp_size=3, probabilities=probs, probs=np.ones(6)),
        dict(samp_unit=units, samp_size=3, probabilities=probs, remove_nan=True),
        # zero remaining random selections with optional units left
        dict(samp_unit=units, samp_size=2, probabilities=np.array([1.0, 1.0, .6, .8, .5, .9])),
        # too many certainty units for the sample size
        dict(samp_unit=units, samp_size=1, probabilities=np.array([1.0, 1.0, .6, .8, .5, .9])),
        # stratum of wrong length / missing / incomplete mapping
        dict(samp_unit=units, samp_size={"a": 2}, probabilities=probs, stratum=np.array(["a", "a", "b", "b", "b", "b"])),
        dict(samp_unit=units, samp_size={"a": 2, "b": 2}, probabilities=probs, stratum=np.array(["a", "a", "b", "b", "b"])),
        dict(samp_unit=units, samp_size={"a": 2, "b": 2}, probabilities=probs, stratum=np.array(["a", "a", "b", "b", "b", None])),
        dict(samp_unit=units, samp_size=3, probabilities=probs, stratum=np.array(["a", "a", "b", "b", "b", "b"])),
        # stratified sample size outside [1, population]
        dict(samp_unit=units, samp_size={"a": 0, "b": 2}, probabilities=probs, stratum=np.array(["a", "a", "a", "b", "b", "b"])),
        dict(samp_unit=units, samp_size={"a": 4, "b": 2}, probabilities=probs, stratum=np.array(["a", "a", "a", "b", "b", "b"])),
    ]


@pytest.mark.parametrize("case", _bad_cases())
def test_invalid_designs_raise_value_error(case):
    selection = SampleSelection(SelectMethod.conditional_poisson)
    units_before = np.array(case["samp_unit"], copy=True)
    probs_before = np.array(case["probabilities"], copy=True)
    with pytest.raises(ValueError):
        selection.prepare_design(**case)
    # Inputs are not rewritten.
    np.testing.assert_array_equal(np.array(case["samp_unit"]), units_before)
    np.testing.assert_array_equal(np.array(case["probabilities"]), probs_before)


def test_prepare_design_rejects_other_methods():
    with pytest.raises(MethodError):
        SampleSelection(SelectMethod.srs_wor).prepare_design(
            np.arange(5), 2, probabilities=np.full(5, 0.4)
        )


# ---------------------------------------------------------------------------
# Draw semantics
# ---------------------------------------------------------------------------


def test_draw_is_reproducible_and_isolated():
    selection = SampleSelection(SelectMethod.conditional_poisson)
    probabilities = np.array([0.2, 0.4, 0.6, 0.8, 0.5, 0.9, 0.3, 0.7])
    design = selection.prepare_design(np.arange(8), 3, probabilities=probabilities)

    first = design.draw(random_state=1234)
    second = design.draw(random_state=1234)
    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_array_equal(first.samp_unit, second.samp_unit)
    assert first.sample_size == 3

    # Each draw owns its selection arrays.
    first.indices[0] = -1
    assert -1 not in second.indices.tolist()
    third = design.draw(random_state=1234)
    assert -1 not in third.indices.tolist()

    # Generator is advanced in place and never touches the global state.
    generator = np.random.default_rng(7)
    before = np.random.get_state()[1].copy()
    a = design.draw(random_state=generator)
    b = design.draw(random_state=generator)
    assert np.array_equal(np.random.get_state()[1], before)
    assert not np.array_equal(a.indices, b.indices) or probabilities.size < 4

    # Same generator state -> same first draw.
    g1 = np.random.default_rng(11)
    g2 = np.random.default_rng(11)
    np.testing.assert_array_equal(
        design.draw(random_state=g1).indices, design.draw(random_state=g2).indices
    )

    with pytest.raises(ValueError):
        design.draw(random_state=-1)
    with pytest.raises(ValueError):
        design.draw(random_state=True)


def test_draw_returns_frame_order_and_result_container():
    selection = SampleSelection(SelectMethod.conditional_poisson)
    units = np.array(["u9", "u3", "u5", "u1", "u7", "u2"])
    probabilities = np.array([0.2, 0.4, 0.6, 0.8, 0.5, 0.9])
    design = selection.prepare_design(units, 3, probabilities=probabilities)
    result = design.draw(random_state=21)

    assert isinstance(result, SelectionResult)
    assert result.design is design
    assert np.all(np.diff(result.indices) > 0)
    np.testing.assert_array_equal(result.samp_unit, units[result.indices])
    np.testing.assert_array_equal(
        result.inclusion_probabilities, design.inclusion_probabilities[result.indices]
    )
    frame = result.to_dataframe()
    assert list(frame["_samp_unit"]) == result.samp_unit.tolist()


def test_design_copies_inputs_and_is_order_invariant():
    probabilities = np.array([0.2, 0.4, 0.6, 0.8, 0.5, 0.9])
    units = np.arange(6)
    selection = SampleSelection(SelectMethod.conditional_poisson)

    mutable_units = units.copy()
    mutable_probs = probabilities.copy()
    design = selection.prepare_design(mutable_units, 3, probabilities=mutable_probs)
    before = design.inclusion_probabilities.copy()
    mutable_units[:] = 99
    mutable_probs[:] = 0.01
    np.testing.assert_array_equal(design.inclusion_probabilities, before)

    # Reorder the frame; align by identifier and the probabilities agree.
    order = np.array([4, 0, 3, 1, 5, 2])
    reordered = selection.prepare_design(units[order], 3, probabilities=probabilities[order])
    aligned = reordered.inclusion_probabilities[np.argsort(order)]
    np.testing.assert_allclose(aligned, before, rtol=1e-12, atol=1e-14)

    joint = design.joint_inclusion([3, 0, 5])
    reordered_joint = reordered.joint_inclusion([3, 0, 5])
    _assert_close(joint, reordered_joint)


def test_joint_inclusion_query_order_and_errors():
    probabilities = np.array([0.2, 0.4, 0.6, 0.8, 0.5, 0.9])
    units = np.array([77, -4, 19, 1000, 3, 555])
    selection = SampleSelection(SelectMethod.conditional_poisson)
    design = selection.prepare_design(units, 2, probabilities=probabilities)

    query = np.array([555, 77, -4])
    matrix = design.joint_inclusion(query)
    assert matrix.shape == (3, 3)
    expected = design.joint_inclusion(units)[np.ix_([5, 0, 1], [5, 0, 1])]
    _assert_close(matrix, expected)
    _assert_close(np.diag(matrix), design.inclusion_probabilities[[5, 0, 1]])

    with pytest.raises(ValueError):
        design.joint_inclusion([77, 77])
    with pytest.raises(ValueError):
        design.joint_inclusion([77, 12345])
    with pytest.raises(ValueError):
        design.joint_inclusion([True])


# ---------------------------------------------------------------------------
# Frequency checks against the exact distribution
# ---------------------------------------------------------------------------


def test_draw_frequencies_match_exact_probabilities():
    probabilities = np.array([0.05, 0.10, 0.25, 0.40, 0.60, 0.75, 0.90])
    sample_size = 3
    selection = SampleSelection(SelectMethod.conditional_poisson)
    design = selection.prepare_design(np.arange(7), sample_size, probabilities=probabilities)
    truth = design.inclusion_probabilities

    n_simulations = 120_000
    generator = np.random.default_rng(20240517)
    counts = np.zeros(7)
    pair_counts = np.zeros((7, 7))
    for _ in range(n_simulations):
        result = design.draw(random_state=generator)
        counts[result.indices] += 1
        block = np.ix_(result.indices, result.indices)
        pair_counts[block] += 1

    estimated = counts / n_simulations
    # Standard error of a Bernoulli(cell) frequency at this sample size.
    standard_error = np.sqrt(truth * (1 - truth) / n_simulations)
    z_scores = np.abs(estimated - truth) / standard_error
    # Five sigma; with 7 cells and correlated draws this is a conservative
    # false-alarm threshold.
    assert np.all(z_scores < 5.0), z_scores

    joint_truth = design.joint_inclusion(np.arange(7))
    pair_estimated = pair_counts / n_simulations
    for a in range(7):
        for b in range(7):
            if a != b:
                se = np.sqrt(joint_truth[a, b] * (1 - joint_truth[a, b]) / n_simulations)
                assert abs(pair_estimated[a, b] - joint_truth[a, b]) < 5.0 * se + 1e-12

    # Sample size is fixed, which a row-sum-only probability check would miss.
    assert np.all(counts.sum() == n_simulations * sample_size)


def test_draw_frequencies_for_stratified_design():
    probabilities = np.array([0.2, 0.4, 0.6, 0.1, 0.3, 0.8])
    strata = np.array(["a", "a", "a", "b", "b", "b"])
    sizes = {"a": 2, "b": 1}
    selection = SampleSelection(SelectMethod.conditional_poisson)
    design = selection.prepare_design(
        np.arange(6), sizes, probabilities=probabilities, stratum=strata
    )
    truth = design.inclusion_probabilities

    n_simulations = 60_000
    generator = np.random.default_rng(998877)
    counts = np.zeros(6)
    for _ in range(n_simulations):
        result = design.draw(random_state=generator)
        counts[result.indices] += 1
        stratum_counts = {label: np.sum(result.stratum == label) for label in sizes}
        for label, expected in sizes.items():
            assert stratum_counts[label] == expected

    estimated = counts / n_simulations
    se = np.sqrt(truth * (1 - truth) / n_simulations)
    assert np.all(np.abs(estimated - truth) / se < 5.0)
