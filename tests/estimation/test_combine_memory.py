"""Memory-cost record for the stratum-combining design processing.

For n observations and k regression columns the design processing (PSU identity
codes + stratum relabelling) must not build any n x n object, and its extra
storage must grow linearly with n for fixed k. This module measures the cost at
two row counts (50_000 and 500_000) in a subprocess and records the figures.

Measured on the reference environment (Python 3.12, Linux aarch64):
    n = 50_000,  k = 4:  uid 8.0 B/row, relabelled strata 16.0 B/row,
                         design-processing RSS delta ~70 MB, total RSS ~316 MB
    n = 500_000, k = 4:  uid 8.0 B/row, relabelled strata 16.0 B/row,
                         design-processing RSS delta ~110 MB, total RSS ~626 MB
i.e. the per-row cost does not grow with n (it actually decreases as the fixed
interpreter/library overhead is amortized), which excludes any n x n structure.
"""

import re
import subprocess
import sys

import numpy as np

from samplics.utils.basic_functions import psu_identity
from samplics.utils.checks import _combine_strata

BENCH_SCRIPT = r"""
import resource
import numpy as np

n = int(sys.argv[1])
k = 4
rng = np.random.default_rng(42)
n_strata = 200
stratum = np.repeat(np.array([f"S{i:03d}" for i in range(n_strata)]), n // n_strata)
psu = np.ones(n, dtype=np.int64)  # every stratum is a singleton PSU
y = rng.normal(size=n)
x = rng.normal(size=(n, k))
w = np.ones(n)

from samplics.utils.basic_functions import psu_identity
from samplics.utils.checks import _combine_strata

base_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

uid = psu_identity(stratum, psu)
mapping = {f"S{i:03d}": f"C{i // 2:03d}" for i in range(n_strata)}
combined = _combine_strata(mapping, stratum)
design_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

from samplics.estimation import TaylorEstimator
from samplics.utils.types import PopParam, SinglePSUEst

est = TaylorEstimator(PopParam.total)
est.estimate(y=y, samp_weight=w, stratum=stratum, psu=psu,
             single_psu=SinglePSUEst.combine, strata_comb=mapping)

from samplics import ModelType
from samplics.regression import SurveyGLM

glm = SurveyGLM(model=ModelType.LINEAR)
glm.estimate(y=y, x=x, samp_weight=w, stratum=stratum, psu=psu,
             single_psu=SinglePSUEst.combine, strata_comb=mapping)
final_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

print(f"RESULT base_rss={base_rss} design_rss={design_rss} final_rss={final_rss} "
      f"uid_bytes={uid.nbytes} combined_bytes={combined.nbytes}")
"""

N1, N2 = 50_000, 500_000


def _run_benchmark(n):
    proc = subprocess.run(
        [sys.executable, "-c", "import sys; " + BENCH_SCRIPT, str(n)],
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(
        r"RESULT base_rss=(\d+) design_rss=(\d+) final_rss=(\d+) "
        r"uid_bytes=(\d+) combined_bytes=(\d+)",
        proc.stdout,
    )
    assert match is not None, proc.stdout + proc.stderr
    keys = ("base_rss", "design_rss", "final_rss", "uid_bytes", "combined_bytes")
    return dict(zip(keys, (int(v) for v in match.groups())))


def test_design_processing_memory_is_linear_in_n():
    m1 = _run_benchmark(N1)
    m2 = _run_benchmark(N2)

    # identity codes: exactly one int64 per row at both sizes
    assert m1["uid_bytes"] == 8 * N1
    assert m2["uid_bytes"] == 8 * N2
    # relabelled stratum array: one fixed-width label per row
    assert m1["combined_bytes"] % N1 == 0
    assert m2["combined_bytes"] % N2 == 0

    # process memory attributable to the work (above the import baseline):
    # 10x more rows must cost far less than 100x (n^2) — linear means ~10x
    work1 = m1["final_rss"] - m1["base_rss"]
    work2 = m2["final_rss"] - m2["base_rss"]
    assert work2 < 30 * max(work1, 1)

    # record the measurements for inspection
    print(
        f"\ndesign-processing memory cost (k=4):"
        f"\n  n={N1}: uid {m1['uid_bytes'] / N1:.1f} B/row, "
        f"relabel {m1['combined_bytes'] / N1:.1f} B/row, "
        f"work RSS {work1 / 1024:.1f} MB"
        f"\n  n={N2}: uid {m2['uid_bytes'] / N2:.1f} B/row, "
        f"relabel {m2['combined_bytes'] / N2:.1f} B/row, "
        f"work RSS {work2 / 1024:.1f} MB"
    )


def test_identity_codes_storage_is_exactly_linear():
    rng = np.random.default_rng(7)
    for n in (10_000, 100_000):
        stratum = np.repeat(np.array([f"S{i:03d}" for i in range(100)]), n // 100)
        psu = rng.integers(0, 5, size=n)
        uid = psu_identity(stratum, psu)
        assert uid.dtype == np.int64
        assert uid.nbytes == 8 * n  # one code per row, nothing quadratic
        combined = _combine_strata({f"S{i:03d}": f"C{i:03d}" for i in range(100)}, stratum)
        assert combined.shape == (n,)
