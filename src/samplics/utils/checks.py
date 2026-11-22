"""Runs checks on the data to help capture some errors faster and provides better error messages.

Functions:
    | *assert_probabilities()* ensures that probability values are between 0 and 1.
    | *assert_weights()* ensures that sample weights are not negatives.
    | *assert_not_unique()* return an assertion error if the array has non unique values.
    | *assert_response_status()* checks that the response values are in ("in", "rr", "nr", "uk").
    | *assert_brr_number_psus()* checks that the number of psus is a multiple of 2.
"""

from __future__ import annotations

from itertools import repeat
from typing import Any, Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import ArrayLike, NDArray

from samplics.utils import formats
from samplics.utils.types import Array, Number, StringNumber


def assert_probabilities(**kargs: Union[Number, Iterable]) -> None:
    err_msg = "Probabilities must be between 0 and 1, inclusively!"
    for k in kargs:
        if not assert_in_range(0, 1, x=kargs[k]):
            raise ValueError(err_msg)


def assert_proportions(**kargs: Union[Number, Iterable]) -> None:
    err_msg = "Proportions must be between 0 and 1, inclusively!"
    for k in kargs:
        if not assert_in_range(0, 1, x=kargs[k]):
            raise ValueError(err_msg)


def assert_in_range(low: Number, high: Number, x: Union[Number, Iterable]) -> bool:
    if isinstance(x, (int, float)):
        if x > high or x < low:
            return False
    elif isinstance(x, (np.ndarray, pd.Series)):
        if (x > high).any() or (x < low).any():
            return False
    elif isinstance(x, Iterable):
        for i in x:
            if isinstance(x, dict):
                if x[i] > high or x[i] < low:
                    return False
            elif i > high or i < low:
                return False

    return True


def assert_weights(weights: Array) -> None:
    weights = formats.numpy_array(weights)
    if (weights < 0).any():
        raise ValueError("Sample weights must be positive values")


def assert_not_unique(array_unique_values: Array) -> None:
    if np.unique(array_unique_values).size != len(array_unique_values):
        raise AssertionError(
            "The array must contains unique values identifying all the units in the sampling frame."
        )


def assert_response_status(
    response_status: Union[str, np.ndarray],
    response_dict: Optional[dict[str, StringNumber]],
) -> None:
    if response_status is None:
        raise AssertionError("response_status is not provided")
    elif not np.isin(response_status, ("in", "rr", "nr", "uk")).all() and response_dict is None:
        raise AssertionError(
            "The response status must only contains values in ('in', 'rr', 'nr', 'uk') or the mapping should be provided using response_dict parameter"
        )
    elif "rr" not in response_status and response_dict is None:
        raise AssertionError("The response status must at least contains rr!")
    elif isinstance(response_dict, dict):
        # resp_keys = list(response_dict.keys())
        resp_keys = [x.lower() for x in response_dict]
        if not np.isin(resp_keys, ("in", "rr", "nr", "uk")).all():
            raise AssertionError("Response mapping dictionnary has unexpected value(s)")


def assert_brr_number_psus(psu: np.ndarray) -> None:
    if psu.size % 2 != 0:
        raise AssertionError("For the BRR method, the number of PSUs must be a multiple of two.")


def _raise_singleton_error(single_psu_strata: Array) -> None:
    raise ValueError(f"Only one PSU in the following strata: {single_psu_strata}")


def _skip_singleton(single_psu_strata: Array, skipped_strata: Array) -> Array:
    skipped_str = np.isin(single_psu_strata, skipped_strata)
    if skipped_str.sum() > 0:
        return single_psu_strata[skipped_str]
    else:
        raise ValueError("{skipped_strata} does not contain singleton PSUs")


def _certainty_singleton(
    singletons: Array,
    _stratum: Array,
    _psu: Array,
    _ssu: Array,
) -> Array:
    # Make a writable copy (avoids 'assignment destination is read-only')
    psu = np.array(_psu, copy=True)

    # Normalize inputs
    singletons = np.atleast_1d(singletons)
    stratum = np.asarray(_stratum)

    # Case 1: per-record SSU array provided → direct copy for singleton strata
    if np.ndim(_ssu) == 1 and _ssu.shape[0] == stratum.shape[0]:
        ssu = np.asarray(_ssu)
        mask = np.isin(stratum, singletons)
        psu[mask] = ssu[mask]
        return psu

    # Case 2: scalar/0-d SSU → assign sequential PSU IDs within each singleton stratum
    if np.ndim(_ssu) == 0 or (_ssu.shape == () or _ssu.shape == (0,)):
        for s in singletons:
            m = stratum == s
            n = int(np.count_nonzero(m))
            if n > 0:
                # 1,2,...,n as integers (match psu dtype if it's integer)
                seq = np.arange(1, n + 1, dtype=int)
                # Cast to psu dtype if needed
                if psu.dtype.kind in ("i", "u"):
                    seq = seq.astype(psu.dtype, copy=False)
                psu[m] = seq
        return psu

    # Fallback: unexpected _ssu shape
    raise ValueError(
        f"Unexpected _ssu shape {np.shape(_ssu)}; expected scalar or per-record array."
    )


def _strata_label_kind(stratum: np.ndarray) -> str:
    """Returns 'int' or 'str' for homogeneous stratum labels, else raises ValueError."""
    if stratum.dtype.kind in ("i", "u"):
        return "int"
    if stratum.dtype.kind in ("U", "S"):
        return "str"
    if stratum.dtype.kind == "O":
        values = stratum.tolist()
        if all(isinstance(v, str) for v in values):
            return "str"
        if all(
            isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
            for v in values
        ):
            return "int"
    raise ValueError(
        "Stratum labels must be all integers or all strings; "
        f"got dtype {stratum.dtype}."
    )


def _check_strata_comb_label(label: Any, kind: str, role: str) -> None:
    if kind == "int":
        valid = isinstance(label, (int, np.integer)) and not isinstance(label, (bool, np.bool_))
    else:
        valid = isinstance(label, (str, np.str_))
    if not valid:
        raise ValueError(
            f"'strata_comb' {role} {label!r} is incompatible with the {kind} stratum labels; "
            "sources and targets must have the same type as the existing labels."
        )


def _combine_strata(
    comb_strata: Mapping[StringNumber, StringNumber],
    _stratum: ArrayLike,
) -> NDArray[np.generic]:
    """Relabels strata in one simultaneous pass, preserving PSU identity downstream.

    Each record is matched against its *original* stratum label only: with
    ``{"A": "B", "B": "C"}`` the records of ``A`` land in ``B`` and the records of
    ``B`` land in ``C``; the mapping is **not** applied transitively. Strata absent
    from the mapping are left untouched, and the target may be an existing stratum
    or a new label (possibly longer than the current labels). The caller's array is
    never modified.

    Args:
        comb_strata: non-empty mapping of source stratum labels to target labels.
            Sources must exist in ``_stratum``; sources and targets must share the
            type of the stratum labels (all integers or all strings).
        _stratum: per-record stratum labels.

    Returns:
        The relabelled stratum array (a new array).

    Raises:
        ValueError: if the mapping is empty, a source is missing, or the labels and
            the mapping are not of one consistent type.
    """
    if not isinstance(comb_strata, Mapping) or len(comb_strata) == 0:
        raise ValueError(
            "The parameter 'strata_comb' must be a non-empty mapping of stratum labels."
        )

    src = np.asarray(_stratum)
    if src.ndim != 1 or src.size == 0:
        raise ValueError("'stratum' must be a non-empty one-dimensional array.")

    kind = _strata_label_kind(src)
    for old, new in comb_strata.items():
        _check_strata_comb_label(old, kind, "source")
        _check_strata_comb_label(new, kind, "target")

    missing = [old for old in comb_strata if not np.any(src == old)]
    if missing:
        raise ValueError(f"'strata_comb' sources not found in 'stratum': {missing}")

    if kind == "str":
        # normalize to a unicode dtype wide enough for the longest target label
        out = np.array(src, dtype=str)
        width = max(
            [out.dtype.itemsize // 4] + [len(str(new)) for new in comb_strata.values()]
        )
        if width > out.dtype.itemsize // 4:
            out = out.astype(f"<U{width}")
    else:
        out = np.array(src, copy=True)

    # simultaneous relabel: masks are computed on the original labels only
    for old, new in comb_strata.items():
        out[src == old] = new

    return out


def _validate_fpc_value(value: Any) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("fpc values must be real numbers in [0, 1]; booleans are not valid.")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(
            f"fpc values must be real numbers in [0, 1]; got {type(value).__name__}."
        )
    if not np.isfinite(value) or value < 0 or value > 1:
        raise ValueError(f"fpc values must be finite numbers in [0, 1]; got {value}.")


def _validate_fpc(
    fpc: Union[dict, Array, Number],
    stratum: Array,
    combined: bool = False,
) -> Union[dict, Number]:
    """Validates the finite population correction against the *final* strata.

    When strata were combined, the strata that disappeared in the merge can no
    longer be referenced: a scalar applies to every final stratum and a dictionary
    must cover exactly the final strata (in any key order). Without combining, the
    historical semantics are kept (scalar broadcast, dictionary keyed by the
    observed strata, or one value per observation).

    Args:
        fpc: scalar, per-observation array, or dictionary keyed by stratum label.
        stratum: per-record *final* stratum labels (empty for unstratified designs).
        combined: whether strata were combined in this call.

    Returns:
        The validated fpc: a scalar for unstratified designs, otherwise a
        dictionary keyed by the final strata.

    Raises:
        ValueError: on missing or extra dictionary keys, values outside [0, 1],
            non-finite or non-numeric values (booleans included), or an array fpc
            used together with combined strata.
    """
    stratum = formats.numpy_array(stratum)

    if isinstance(fpc, dict):
        if stratum.shape in ((), (0,)):
            raise ValueError("A dictionary 'fpc' requires a stratified sample.")
        final_strata = np.unique(stratum).tolist()
        key_set = set(fpc.keys())
        strata_set = set(final_strata)
        if key_set != strata_set:
            missing = [s for s in final_strata if s not in key_set]
            extra = [k for k in fpc.keys() if k not in strata_set]
            raise ValueError(
                "'fpc' dictionary keys must be exactly the final strata; "
                f"missing: {missing}, unexpected: {extra}."
            )
        for value in fpc.values():
            _validate_fpc_value(value)
        # rebuild in a deterministic order so key insertion order plays no role
        return {s: fpc[s] for s in final_strata}
    elif isinstance(fpc, (np.ndarray, pd.Series, pl.Series, list, tuple)):
        if combined:
            raise ValueError(
                "An array 'fpc' cannot be mapped to combined strata; provide a scalar "
                "or a dictionary keyed by the final strata."
            )
        if stratum.shape in ((), (0,)):
            raise ValueError("An array 'fpc' requires a stratified sample.")
        fpc_arr = formats.numpy_array(fpc)
        if fpc_arr.shape[0] != stratum.shape[0]:
            raise ValueError("An array 'fpc' must provide one value per observation.")
        fpc_dict = dict(zip(stratum.tolist(), fpc_arr.tolist()))
        for value in fpc_dict.values():
            _validate_fpc_value(value)
        return fpc_dict
    else:
        _validate_fpc_value(fpc)
        if stratum.shape in ((), (0,)):
            return fpc
        return dict(zip(np.unique(stratum).tolist(), repeat(fpc)))
