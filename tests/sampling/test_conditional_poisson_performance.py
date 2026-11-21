"""Performance and memory guardrails for conditional Poisson selection."""

import tracemalloc
import warnings

import numpy as np
import pytest

from samplics.estimation import TaylorEstimator
from samplics.sampling import SampleSelection
from samplics.utils.types import PopParam, SelectMethod

warnings.filterwarnings("ignore", category=FutureWarning)


def _build_large_design():
    frame_size = 2000
    sample_size = 80
    generator = np.random.default_rng(2024)
    probabilities = generator.uniform(0.02, 0.98, frame_size)
    selection = SampleSelection(SelectMethod.conditional_poisson)
    design = selection.prepare_design(
        np.arange(frame_size), sample_size, probabilities=probabilities
    )
    return design, frame_size, sample_size


@pytest.mark.slow
def test_large_frame_memory_and_repeated_draws():
    tracemalloc.start()
    design, frame_size, sample_size = _build_large_design()
    build_current, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    baseline, _ = tracemalloc.get_traced_memory()

    results = []
    for seed in range(10):
        result = design.draw(random_state=seed)
        assert result.sample_size == sample_size
        joint = design.joint_inclusion(result.samp_unit)
        assert joint.shape == (sample_size, sample_size)
        assert np.allclose(joint, joint.T)
        assert np.allclose(np.diag(joint), result.inclusion_probabilities)
        assert np.all(joint >= 0.0)
        results.append(result)

    peak, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # New peak memory after the design exists must stay below 96 MiB.
    extra = (peak - baseline) / 2**20
    assert extra < 96.0, f"peak extra memory {extra:.2f} MiB"

    # Each draw owns independent arrays.
    results[0].indices[0] = -1
    assert -1 not in results[1].indices.tolist()

    # First-order probabilities sum to the sample size by construction.
    assert design.inclusion_probabilities.sum() == pytest.approx(sample_size, rel=1e-10)


def test_no_pairwise_population_matrix_is_built():
    """The design stores O(N) and O(N * n) tables, not an N x N matrix."""
    design, frame_size, sample_size = _build_large_design()
    for stratum in design._strata:
        assert stratum.prefix.shape == (frame_size + 1, sample_size + 1)
        assert stratum.suffix.shape == (frame_size + 1, sample_size + 1)
    assert not hasattr(design, "joint_matrix")
    assert design.inclusion_probabilities.shape == (frame_size,)


def test_large_design_estimation_smoke():
    design, frame_size, sample_size = _build_large_design()
    result = design.draw(random_state=7)
    values = np.random.default_rng(9).normal(50.0, 10.0, sample_size)
    estimate = TaylorEstimator(PopParam.total)
    estimate.estimate_design(values, result)
    assert np.isfinite(estimate.point_est)
    assert np.isfinite(estimate.variance)
    assert estimate.variance >= 0.0
    assert estimate.degree_of_freedom == sample_size - 1
