"""Calibrated replicate-weight containers and orchestration.

The calibration of replicate weights reuses the same bounded chi-square solver
(:mod:`samplics.weighting.calibration`) for the full sample and for every
replicate column.  Each replicate is solved independently against *its own*
base weights and its own structural-zero pattern: multiplying every replicate
by the full-sample adjustment factor is explicitly not what happens here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from samplics.weighting.calibration import CalibrationError, calibrate_bounded


__all__ = ["CalibratedReplicates", "calibrate_replicates"]

_CALIBRATION_KEYS = frozenset({"aux_vars", "control", "bounds", "tol", "max_iter"})
_REQUIRED_KEYS = frozenset({"aux_vars", "control"})


@dataclass
class CalibratedReplicates:
    """Result of generating replicate weights under calibration.

    Attributes:
        full_weights: length-n calibrated full-sample weights.
        rep_weights: n-by-R matrix of calibrated replicate weights; column
            ``r`` corresponds to ``replicate_ids[r]``.
        replicate_ids: length-R ordered labels inherited from the underlying
            replicate generator.
        method: replicate design (``RepMethod`` value).
        fay_coef: Fay coefficient for BRR-Fay, ``None`` otherwise.
        rep_coefs: length-R design coefficients used by the variance
            estimator (kept identical to the uncalibrated design).
        degree_of_freedom: design degrees of freedom.
        nb_strata: number of design strata (metadata used to validate the
            degrees of freedom downstream).
        conservative: whether the batch was produced for the conservative
            variance convention.
    """

    full_weights: np.ndarray
    rep_weights: np.ndarray
    replicate_ids: np.ndarray
    method: object
    fay_coef: Optional[float]
    rep_coefs: np.ndarray
    degree_of_freedom: int
    nb_strata: int
    conservative: bool = False

    @property
    def nb_reps(self) -> int:
        return int(self.rep_weights.shape[1])


def _parse_calibration(calibration: dict) -> dict:
    """Validate the calibration mapping and normalize its options."""

    if not isinstance(calibration, dict):
        raise ValueError("calibration must be a mapping with keys 'aux_vars' and 'control'.")
    keys = set(calibration.keys())
    unknown = keys - _CALIBRATION_KEYS
    if unknown:
        raise ValueError(
            f"Unknown calibration key(s): {sorted(unknown)}. "
            f"Allowed keys are: {sorted(_CALIBRATION_KEYS)}."
        )
    missing = _REQUIRED_KEYS - keys
    if missing:
        raise ValueError(f"calibration is missing required key(s): {sorted(missing)}.")

    aux_vars = np.asarray(calibration["aux_vars"], dtype=np.float64)
    control = np.asarray(calibration["control"], dtype=np.float64)
    if aux_vars.ndim == 1:
        aux_vars = aux_vars.reshape(-1, 1)
    options: dict = {
        "aux_vars": aux_vars,
        "control": control,
        "bounds": tuple(calibration.get("bounds", (0.2, 5.0))),
        "tol": float(calibration.get("tol", 1e-9)),
        "max_iter": int(calibration.get("max_iter", 500)),
    }
    return options


def calibrate_replicates(
    samp_weight: np.ndarray,
    raw_rep_weights: np.ndarray,
    replicate_ids: np.ndarray,
    calibration: dict,
    method: object,
    fay_coef: Optional[float],
    rep_coefs: Union[np.ndarray, list],
    degree_of_freedom: int,
    nb_strata: int,
    conservative: bool = False,
) -> CalibratedReplicates:
    """Calibrate the full sample first, then every replicate independently.

    Args:
        samp_weight: length-n base (full-sample) weights.
        raw_rep_weights: n-by-R uncalibrated replicate weights produced by the
            design generator; each column is its own base-weight vector and
            may contain its own structural zeros.
        replicate_ids: length-R column labels, in column order.
        calibration: validated calibration mapping (see
            :func:`_parse_calibration`).
        method, fay_coef, rep_coefs, degree_of_freedom: design metadata handed
            unchanged to the estimator.
        nb_strata: number of design strata.

    Returns:
        A :class:`CalibratedReplicates`.

    Raises:
        CalibrationError: if any single solve fails; the error identifies
            ``"full"`` or the failing replicate label and no partial result is
            returned.
    """

    options = _parse_calibration(calibration)
    aux_vars = options["aux_vars"]
    control = options["control"]
    common = dict(
        bounds=options["bounds"],
        tol=options["tol"],
        max_iter=options["max_iter"],
    )
    n = samp_weight.shape[0]
    if aux_vars.shape[0] != n:
        raise ValueError("aux_vars must have one row per sample unit.")
    if raw_rep_weights.ndim != 2 or raw_rep_weights.shape[0] != n:
        raise ValueError("rep_weights must be an n-by-R matrix aligned with samp_weight.")

    # 1. Full sample first.
    full_weights = calibrate_bounded(
        samp_weight, aux_vars, control, label="full", **common
    )

    # 2. Every replicate, solved against its own base weights and zeros.
    r_count = raw_rep_weights.shape[1]
    calibrated = np.zeros_like(raw_rep_weights, dtype=np.float64)
    for r in range(r_count):
        label = str(replicate_ids[r])
        try:
            calibrated[:, r] = calibrate_bounded(
                raw_rep_weights[:, r], aux_vars, control, label=label, **common
            )
        except CalibrationError as exc:
            # Re-raise with the replicate identity attached; nothing partial is
            # delivered because the output object is only built after this loop.
            exc.target = label if exc.target == "full" else exc.target
            raise

    coefs = np.asarray(rep_coefs, dtype=np.float64).reshape(-1)
    if coefs.shape[0] != r_count:
        raise ValueError("rep_coefs must have one entry per replicate column.")

    return CalibratedReplicates(
        full_weights=np.asarray(full_weights, dtype=np.float64),
        rep_weights=calibrated,
        replicate_ids=np.asarray(replicate_ids),
        method=method,
        fay_coef=fay_coef,
        rep_coefs=coefs,
        degree_of_freedom=int(degree_of_freedom),
        nb_strata=int(nb_strata),
        conservative=bool(conservative),
    )
