# Using the conditional Poisson design end to end

```python
import numpy as np

from samplics.estimation import TaylorEstimator
from samplics.sampling import SampleSelection
from samplics.utils.types import PopParam, SelectMethod
from samplics.weighting import SampleWeight

frame_ids = np.array(["u01", "u02", "u03", "u04", "u05", "u06", "u07", "u08"])
bernoulli = np.array([0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95])

design = SampleSelection(SelectMethod.conditional_poisson).prepare_design(
    samp_unit=frame_ids,
    samp_size=3,
    probabilities=bernoulli,          # pre-conditioning, not final pi
)

# Final inclusion probabilities in frame order (exact, no simulation).
print(design.inclusion_probabilities)

# One draw; integer seeds are reproducible and do not touch the global RNG.
result = design.draw(random_state=2024)
print(result.samp_unit, result.indices)

# In-sample second-order probabilities, ordered like the sample.
joint = design.joint_inclusion(result.samp_unit)
assert np.allclose(np.diag(joint), result.inclusion_probabilities)

# Base weights come from the *design*, i.e. 1 / final pi_i.
base_weights = SampleWeight.from_selection(result)

y = np.array([5.0, 7.0, 11.0])          # aligned with the sample order

total = TaylorEstimator(PopParam.total)
total.estimate_design(y, result)
print(total.point_est, total.variance, total.degree_of_freedom)
print(total.to_dataframe())

ratio = TaylorEstimator(PopParam.ratio)
ratio.estimate_design(y, result, x=np.array([1.0, 2.0, 3.0]))
```

Stratified frames use a mapping for `samp_size`; `domain` in `estimate_design`
must cover the whole frame because domain means/ratios divide by the known
number of frame units in each domain:

```python
strata = np.array(["north", "south", "north", "south", "north", "south", "north", "south"])
design = SampleSelection(SelectMethod.conditional_poisson).prepare_design(
    frame_ids,
    {"north": 2, "south": 2},
    probabilities=bernoulli,
    stratum=strata,
)
result = design.draw(random_state=7)

domain = np.array(["a", "a", "b", "b", "a", "b", "a", "b"])   # one label per frame row
estimate = TaylorEstimator(PopParam.mean)
estimate.estimate_design(
    y=np.arange(4.0),           # sample order
    selection=result,
    domain=domain,
)
print(estimate.point_est)       # dict keyed by every domain present in the frame
```

A non-census stratum with a single random selection makes the variance
inestimable; `estimate_design` then rejects the whole call rather than dropping
the stratum or recording a zero variance.
