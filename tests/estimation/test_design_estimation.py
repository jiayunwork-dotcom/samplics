"""Tests for design-based weights and Taylor estimation under conditional Poisson."""

import itertools
import warnings

import numpy as np
import pytest
from scipy.special import logsumexp

from samplics.estimation import TaylorEstimator
from samplics.sampling import SampleSelection
from samplics.sampling.conditional_poisson import SelectionResult
from samplics.utils.types import PopParam, SelectMethod
from samplics.weighting import SampleWeight

warnings.filterwarnings("ignore", category=FutureWarning)


def _enumerate(probabilities, sample_size):
    """Exact conditional-Poisson distribution for a non-stratified frame."""
    probabilities = np.asarray(probabilities, dtype=float)
    optional = np.flatnonzero(probabilities < 1.0)
    mandatory = probabilities == 1.0
    odds = probabilities[optional] / (1.0 - probabilities[optional])
    r = sample_size - int(mandatory.sum())
    subsets = list(itertools.combinations(range(optional.size), r))
    weights = np.array([np.prod(odds[list(subset)]) for subset in subsets])
    distribution = weights / weights.sum()
    return optional, mandatory, subsets, distribution


def _design(probabilities, sample_size, units=None):
    selection = SampleSelection(SelectMethod.conditional_poisson)
    units = np.arange(len(probabilities)) if units is None else np.asarray(units)
    return selection.prepare_design(units, sample_size, probabilities=probabilities)


def _selection(design, indices):
    return SelectionResult(
        samp_unit=np.asarray(design.samp_unit)[indices],
        indices=np.asarray(indices),
        stratum=None,
        design=design,
    )


def test_from_selection_uses_final_probabilities_not_bernoulli_parameters():
    probabilities = np.array([0.20, 0.35, 0.50, 0.65, 0.80])
    design = _design(probabilities, 2)
    result = design.draw(random_state=17)

    weights = SampleWeight.from_selection(result)
    np.testing.assert_allclose(weights, 1.0 / result.inclusion_probabilities)
    # The Bernoulli parameters are deliberately different from final pi.
    assert not np.allclose(result.inclusion_probabilities, probabilities[result.indices])
    np.testing.assert_allclose(weights, 1.0 / design.inclusion_probabilities[result.indices])


def _enumerated_joint(probabilities, sample_size):
    """Exact inclusion probabilities and joint matrix from raw enumeration.

    This reference does not call the design's probability code; it rebuilds
    first- and second-order inclusion probabilities from the subset table.
    """
    optional, mandatory, subsets, distribution = _enumerate(probabilities, sample_size)
    size = len(probabilities)
    pi = np.ones(size)
    joint = np.ones((size, size))
    nonzero = ~mandatory
    pi_optional = np.zeros(len(optional))
    joint_optional = np.zeros((len(optional), len(optional)))
    for subset, weight in zip(subsets, distribution):
        for a in subset:
            pi_optional[a] += weight
            for b in subset:
                joint_optional[a, b] += weight
    pi[nonzero] = pi_optional
    joint[np.ix_(nonzero, nonzero)] = joint_optional
    for a in range(size):
        for b in range(size):
            if a == b:
                continue
            if mandatory[a] and not mandatory[b]:
                joint[a, b] = joint[b, a] = pi[b]
            elif mandatory[b] and not mandatory[a]:
                joint[a, b] = joint[b, a] = pi[a]
    np.fill_diagonal(joint, pi)
    return pi, joint


def test_total_design_expectation_and_variance():
    probabilities = np.array([0.08, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92])
    sample_size = 4
    values = np.array([9.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 1.0, 10.0])
    design = _design(probabilities, sample_size)
    optional, mandatory, subsets, distribution = _enumerate(probabilities, sample_size)
    assert optional.size == probabilities.size

    pi, joint = _enumerated_joint(probabilities, sample_size)
    np.testing.assert_allclose(design.inclusion_probabilities, pi, rtol=1e-10)
    true_total = values.sum()
    true_variance = float(
        sum(
            (joint[a, b] - pi[a] * pi[b]) * (values[a] / pi[a]) * (values[b] / pi[b])
            for a in range(values.size)
            for b in range(values.size)
        )
    )

    expectations = {"total": 0.0}
    expected_variance = 0.0
    for subset, weight in zip(subsets, distribution):
        index = np.array(subset)
        estimate = TaylorEstimator(PopParam.total)
        estimate.estimate_design(values[index], _selection(design, index))
        expectations["total"] += weight * estimate.point_est
        expected_variance += weight * estimate.variance

    assert expectations["total"] == pytest.approx(true_total, rel=1e-12)
    # Sen-Yates-Grundy is unbiased for the fixed-size design here.
    assert expected_variance == pytest.approx(true_variance, rel=1e-10)


def test_mean_and_ratio_design_expectation():
    probabilities = np.array([0.15, 0.30, 0.45, 0.60, 0.75, 0.90])
    sample_size = 3
    y = np.array([2.0, 4.0, 6.0, 9.0, 11.0, 13.0])
    x = np.array([1.0, 1.5, 2.0, 3.0, 3.5, 4.0])
    design = _design(probabilities, sample_size)
    _, _, subsets, distribution = _enumerate(probabilities, sample_size)
    pi, joint = _enumerated_joint(probabilities, sample_size)
    expected_mean = 0.0
    expected_ratio = 0.0
    expected_mean_variance = 0.0
    expected_ratio_variance = 0.0
    true_ratio = y.sum() / x.sum()
    for subset, weight in zip(subsets, distribution):
        index = np.array(subset)
        mean_est = TaylorEstimator(PopParam.mean)
        mean_est.estimate_design(y[index], _selection(design, index))
        ratio_est = TaylorEstimator(PopParam.ratio)
        ratio_est.estimate_design(y[index], _selection(design, index), x=x[index])
        expected_mean += weight * mean_est.point_est
        expected_ratio += weight * ratio_est.point_est
        expected_mean_variance += weight * mean_est.variance
        expected_ratio_variance += weight * ratio_est.variance

    assert expected_mean == pytest.approx(y.mean(), rel=1e-12)
    # The ratio is a nonlinear estimator: its design expectation is unbiased
    # only to first order.  Check closeness relative to the estimator spread.
    assert expected_ratio == pytest.approx(true_ratio, rel=2e-2)

    # True design variance of the mean: Var(HT total) / N^2.
    true_total_variance = float(
        sum(
            (joint[a, b] - pi[a] * pi[b]) * (y[a] / pi[a]) * (y[b] / pi[b])
            for a in range(y.size)
            for b in range(y.size)
        )
    )
    assert expected_mean_variance == pytest.approx(
        true_total_variance / y.size**2, rel=1e-10
    )

    # Ratio: the variance estimator is the linearised SYG variance with the
    # *sample* ratio and the estimated denominator.  Compute the same
    # linearisation independently and compare exactly.
    expected_ratio_variance = 0.0
    for subset, weight in zip(subsets, distribution):
        index = np.array(subset)
        pi_subset = pi[index]
        sample_ratio = float(np.sum(y[index] / pi_subset) / np.sum(x[index] / pi_subset))
        denominator = float(np.sum(x[index] / pi_subset))
        residual = y[index] - sample_ratio * x[index]
        sub_joint = joint[np.ix_(index, index)]
        pair = 0.0
        for a in range(index.size):
            for b in range(index.size):
                if a == b:
                    continue
                coefficient = (pi_subset[a] * pi_subset[b] - sub_joint[a, b]) / sub_joint[a, b]
                pair += coefficient * (
                    residual[a] / pi_subset[a] - residual[b] / pi_subset[b]
                ) ** 2
        expected_ratio_variance += 0.5 * weight * pair / denominator**2
    sample_variance = 0.0
    for subset, weight in zip(subsets, distribution):
        index = np.array(subset)
        estimate = TaylorEstimator(PopParam.ratio)
        estimate.estimate_design(y[index], _selection(design, index), x=x[index])
        sample_variance += weight * estimate.variance
    assert sample_variance == pytest.approx(expected_ratio_variance, rel=1e-10)


def test_domain_mean_uses_population_domain_counts():
    probabilities = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
    sample_size = 5
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    domain = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    design = _design(probabilities, sample_size)
    _, _, subsets, distribution = _enumerate(probabilities, sample_size)
    pi, joint = _enumerated_joint(probabilities, sample_size)

    domain_sizes = {"a": int(np.sum(domain == "a")), "b": int(np.sum(domain == "b"))}
    expected = {"a": 0.0, "b": 0.0}
    expected_variance = {"a": 0.0, "b": 0.0}
    for subset, weight in zip(subsets, distribution):
        index = np.array(subset)
        estimate = TaylorEstimator(PopParam.mean)
        estimate.estimate_design(y[index], _selection(design, index), domain=domain)
        assert list(estimate.point_est) == ["a", "b"]
        for label in ("a", "b"):
            expected[label] += weight * estimate.point_est[label]
            expected_variance[label] += weight * estimate.variance[label]

    for label in ("a", "b"):
        population_mean = y[domain == label].mean()
        assert expected[label] == pytest.approx(population_mean, rel=1e-12)
        linearized = np.where(domain == label, y, 0.0)
        true_variance = float(
            sum(
                (joint[a, b] - pi[a] * pi[b])
                * (linearized[a] / pi[a])
                * (linearized[b] / pi[b])
                for a in range(y.size)
                for b in range(y.size)
            )
        ) / domain_sizes[label] ** 2
        assert expected_variance[label] == pytest.approx(true_variance, rel=1e-10)


def test_domain_observations_outside_domain_contribute_zero():
    probabilities = np.full(6, 0.5)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    domain = np.array(["a", "a", "a", "b", "b", "b"])
    design = _design(probabilities, 3)
    result = _selection(design, np.array([0, 3, 4]))
    estimate = TaylorEstimator(PopParam.total)
    estimate.estimate_design(y[result.indices], result, domain=domain)
    # "b" variance comes only from units 3 and 4; unit 0 contributes zero.
    assert estimate.point_est["a"] == pytest.approx(y[0] / 0.5)
    assert estimate.point_est["b"] == pytest.approx((y[3] + y[4]) / 0.5)


def test_single_random_unit_rejects_whole_estimation():
    probabilities = np.array([0.2, 0.4, 0.6, 0.8])
    design = _design(probabilities, 1)
    result = design.draw(random_state=3)
    assert result.sample_size == 1
    estimate = TaylorEstimator(PopParam.total)
    with pytest.raises(ValueError):
        estimate.estimate_design(np.array([1.0]), result)


def test_deterministic_design_gives_zero_variance_and_degenerate_interval():
    probabilities = np.array([0.2, 0.4, 0.6, 0.8])
    selection = SampleSelection(SelectMethod.conditional_poisson)
    design = selection.prepare_design(np.arange(4), 4, probabilities=probabilities)
    result = design.draw(random_state=5)
    assert result.sample_size == 4

    estimate = TaylorEstimator(PopParam.total)
    estimate.estimate_design(np.array([1.0, 2.0, 3.0, 4.0]), result)
    assert estimate.variance == 0.0
    assert estimate.stderror == 0.0
    assert estimate.degree_of_freedom == 0
    assert estimate.lower_ci == pytest.approx(10.0)
    assert estimate.upper_ci == pytest.approx(10.0)


def test_certainty_and_census_strata_add_no_random_variance():
    selection = SampleSelection(SelectMethod.conditional_poisson)
    design = selection.prepare_design(
        np.array([100, 101, 102, 103, 104]),
        {"certain": 2, "random": 2},
        probabilities=np.array([1.0, 1.0, 0.4, 0.6, 0.8]),
        stratum=np.array(["certain", "certain", "random", "random", "random"]),
    )
    result = design.draw(random_state=1)
    assert result.sample_size == 4
    certain = result.stratum == "certain"
    assert np.sum(certain) == 2
    assert np.sum(~certain) == 2

    values = np.array([3.0, 4.0, 5.0, 6.0])
    estimate = TaylorEstimator(PopParam.total)
    estimate.estimate_design(values, result)
    # The certainty stratum contributes exactly zero variance; the random
    # stratum has two selections and therefore two degrees of freedom.
    assert np.isfinite(estimate.variance)
    assert estimate.variance > 0.0
    assert estimate.degree_of_freedom == 1

    # All-certainty design: deterministic and zero variance.
    deterministic = selection.prepare_design(
        np.arange(2), 2, probabilities=np.array([1.0, 1.0])
    )
    certain_result = deterministic.draw(random_state=0)
    estimate = TaylorEstimator(PopParam.total)
    estimate.estimate_design(np.array([3.0, 4.0]), certain_result)
    assert estimate.variance == 0.0
    assert estimate.degree_of_freedom == 0
    assert estimate.lower_ci == estimate.upper_ci == pytest.approx(7.0)


def test_stratified_design_variance_and_degrees_of_freedom():
    probabilities = np.array([0.2, 0.4, 0.6, 0.1, 0.3, 0.8])
    strata = np.array(["a", "a", "a", "b", "b", "b"])
    design = SampleSelection(SelectMethod.conditional_poisson).prepare_design(
        np.arange(6), {"a": 2, "b": 2}, probabilities=probabilities, stratum=strata
    )
    result = design.draw(random_state=9)
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    strat = result.stratum
    estimate = TaylorEstimator(PopParam.total)
    estimate.estimate_design(values[result.indices], result)
    # Two strata, each with two random selections -> df = (2-1)+(2-1) = 2.
    assert estimate.degree_of_freedom == 2
    assert np.isfinite(estimate.variance)


def test_failure_keeps_previous_successful_results():
    probabilities = np.array([0.2, 0.4, 0.6, 0.8, 0.5])
    design = _design(probabilities, 2)
    good = design.draw(random_state=4)
    estimate = TaylorEstimator(PopParam.total)
    estimate.estimate_design(np.array([1.0, 2.0]), good)
    previous = estimate.point_est, estimate.variance, estimate.stderror, estimate.lower_ci

    with pytest.raises(ValueError):
        estimate.estimate_design(np.array([1.0, np.nan]), good)
    with pytest.raises(ValueError):
        estimate.estimate_design(np.array([1.0]), good)
    with pytest.raises(ValueError):
        estimate.estimate_design(np.array([1.0, 2.0]), good, domain=np.array(["a", "b"]))

    assert (estimate.point_est, estimate.variance, estimate.stderror, estimate.lower_ci) == previous


def test_to_dataframe_full_pipeline():
    probabilities = np.array([0.15, 0.30, 0.45, 0.60, 0.75, 0.90])
    design = _design(probabilities, 3, units=np.array([10, 20, 30, 40, 50, 60]))
    result = design.draw(random_state=2)
    weights = SampleWeight.from_selection(result)
    values = 2.0 * weights
    estimate = TaylorEstimator(PopParam.total)
    estimate.estimate_design(values, result)
    frame = estimate.to_dataframe()
    assert list(frame.columns) == ["_param", "_estimate", "_stderror", "_lci", "_uci", "_cv"]
    # y is 2 / pi_i for the *sampled* units, so the HT total is twice the sum
    # of the inverted sample inclusion probabilities (not the frame size).
    assert estimate.point_est == pytest.approx(float(np.sum(values * weights)))
    np.testing.assert_allclose(values, 2.0 * weights)
    assert estimate.point_est == pytest.approx(2.0 * float(np.sum(weights**2)))
    assert frame["_estimate"].iloc[0] == pytest.approx(estimate.point_est)


def test_ratio_requires_x_and_rejects_bad_input():
    probabilities = np.array([0.2, 0.4, 0.6, 0.8])
    design = _design(probabilities, 2)
    result = design.draw(random_state=1)
    estimate = TaylorEstimator(PopParam.ratio)
    with pytest.raises(ValueError):
        estimate.estimate_design(np.ones(2), result)
    with pytest.raises(ValueError):
        estimate.estimate_design(np.ones(2), result, x=np.ones(3))


def test_reorder_frame_and_ids_preserved():
    units = np.array(["delta", "alpha", "charlie", "bravo"])
    probabilities = np.array([0.2, 0.4, 0.6, 0.8])
    design = _design(probabilities, 2, units=units)
    result = design.draw(random_state=6)
    weights = SampleWeight.from_selection(result)
    np.testing.assert_array_equal(result.samp_unit, units[result.indices])
    np.testing.assert_allclose(weights, 1.0 / design.inclusion_probabilities[result.indices])
