import numpy as np
import pandas as pd
import pytest

from samplics.utils.checks import (assert_brr_number_psus, assert_in_range,
                                   assert_not_unique, assert_proportions,
                                   assert_response_status, assert_weights)


@pytest.mark.parametrize(
    "x1, x2, x3, x4",
    [(35, -35, 0.35, [0, 1, 39])],
)
def test_in_range_ints_successes(x1, x2, x3, x4):
    assert assert_in_range(low=10, high=39, x=x1)
    assert assert_in_range(low=-39, high=-10, x=x2)
    assert assert_in_range(low=0, high=1, x=x3)
    assert assert_in_range(low=-10, high=39, x=x4)


def test_in_range_for_ints_fails():
    assert not assert_in_range(low=10, high=39, x=5)
    assert not assert_in_range(low=-39, high=-10, x=-135)
    assert not assert_in_range(low=0, high=1, x=1.35)
    assert not assert_in_range(low=-10, high=39, x=-30)
    assert not assert_in_range(low=-10, high=39, x=1000)


def test_fin_range_for_floats_successes():
    assert assert_in_range(low=10.1, high=39.0, x=35.22)
    assert assert_in_range(low=-39.2, high=-10.1, x=-35.35)
    assert assert_in_range(low=0.0, high=1.0, x=0.35)
    assert assert_in_range(low=-10.0, high=39.0, x=0.0)
    assert assert_in_range(low=-10.0, high=39.0, x=1.9)
    assert assert_in_range(low=-10.0, high=39.0, x=0.039)


def test_in_range_for_floats_fails():
    assert not assert_in_range(low=10.0, high=39.0, x=5.5)
    assert not assert_in_range(low=-39.3, high=-10.1, x=-135.23)
    assert not assert_in_range(low=0.0, high=1.00, x=1.35)
    assert not assert_in_range(low=-10.0, high=39.33, x=100.01)
    assert not assert_in_range(low=-10.0, high=39.3, x=39.33)


@pytest.mark.parametrize(
    "x1",
    [np.array([20, 17, 11, 20, 23]), pd.Series([20, 17, 11, 20, 23])],
)
@pytest.mark.parametrize(
    "x2",
    [
        np.array([-2, 17, -11, 20, 23]),
        pd.Series([-10, 17, -11, 20, 23]),
        pd.Series([-10, 17, -11, 20, 23]),
        pd.Series([11, 15, 17]),
    ],
)
def test_in_range_np_pd_successes(x1, x2):
    assert assert_in_range(low=11, high=23, x=x1)
    assert assert_in_range(low=-11, high=23, x=x2)


def test_in_range_for_np_pd_fails():
    assert not assert_in_range(low=11, high=23, x=np.array([20, 17, 111, 20, 23]))
    assert not assert_in_range(low=11, high=23, x=pd.Series([20, -107, 11, 20, 23]))
    assert not assert_in_range(low=-101, high=0, x=np.array([-2, -17, -11, -20, 0.0023]))
    assert not assert_in_range(low=-11, high=23, x=pd.Series([-10, 17, -11.0001, 20, 23]))


def test_in_range_for_lists_successes():
    assert assert_in_range(low=11, high=23, x=[20, 17, 11, 20, 23])
    assert assert_in_range(low=-11, high=23, x=[-2, 17, -11, 20, 23])
    assert assert_in_range(low=-11, high=23, x=[-10, 17, -11, 20, 23])


def test_in_range_for_lists_fails():
    assert not assert_in_range(low=11, high=23, x=np.array([20, 17, 111, 20, 23]))
    assert not assert_in_range(low=11, high=23, x=pd.Series([20, -107, 11, 20, 23]))
    assert not assert_in_range(low=-101, high=0, x=np.array([-2, -17, -11, -20, 0.0023]))
    assert not assert_in_range(low=-11, high=23, x=pd.Series([-10, 17, -11.0001, 20, 23]))


@pytest.mark.parametrize(
    "x",
    [
        {"one": 11, "two": 23, "three": 17},
        {"one": -11, "two": 23, "three": 17, 4: 0.23},
        {4: 0.23},
    ],
)
def test_in_range_for_dicts_successes1(x):
    assert assert_in_range(low=-11, high=23, x=x)


@pytest.mark.parametrize(
    "x",
    [
        {"one": 101, "two": 23, "three": 17},
        {"one": -11.0001, "two": 23, "three": 17, 3: 0.23},
        {3: 0.23},
    ],
)
def test_in_range_for_dicts_successes2(x):
    assert not assert_in_range(low=11, high=23, x=x)


@pytest.mark.parametrize("x", [0.0, 0.00001, 0.3, 0.45, 0.85, 0.9999, 1.0])
def test_assert_proportions_for_numbers_successes(x):
    assert assert_proportions(x=x) is None


@pytest.mark.xfail(strict=True, reason="Number is not between 0 and 1")
@pytest.mark.parametrize("x", [-0.0, -0.00001, -1.000001, -1.1])
def test_assert_proportions_for_numbers_fails(x):
    assert assert_proportions(x=x)


@pytest.mark.xfail(strict=True, reason="At least one number in the lists is not between 0 and 1")
@pytest.mark.parametrize("x", [[-0.0, 0.1, 0, 3], [-0.00001], [-1.000001, -1.1]])
def test_assert_proportions_for_list_fails(x):
    assert assert_proportions(x=x)


@pytest.mark.xfail(
    strict=True, reason="At least one number in the numpy arrays is not between 0 and 1"
)
@pytest.mark.parametrize(
    "x",
    [np.array([-0.0, 0.1, 0, 3]), np.array([-0.00001]), np.array([-1.000001, -1.1])],
)
def test_assert_proportions_for_numpy_arrays_fails(x):
    assert assert_proportions(x=x)


@pytest.mark.xfail(
    strict=True,
    reason="At least one number in the pandas series is not between 0 and 1",
)
@pytest.mark.parametrize(
    "x",
    [pd.Series([-0.0, 0.1, 0, 3]), pd.Series([-0.00001]), pd.Series([-1.000001, -1.1])],
)
def test_assert_proportions_for_pandas_Series_fails(x):
    assert assert_proportions(x=x) is None


@pytest.mark.xfail(
    strict=True, reason="At least one number in the dictionaries is not between 0 and 1"
)
@pytest.mark.parametrize("x", [{"one": 0.1, "two": 1.1}, {"one": -1, 3: 0.5}, {1: -0.1, 2: 1}])
def test_assert_proportions_for_dicts_fails(x):
    assert assert_proportions(x=x) is None


def test_assert_weights():
    assert assert_weights([1]) is None
    assert assert_weights([1, 2, 3, 5]) is None
    assert assert_weights(np.array([1, 2, 3, 5])) is None
    assert assert_weights(pd.Series([1, 2, 3, 5])) is None


@pytest.mark.xfail(strict=True, reason="Negative weights")
def test_assert_weights2():
    assert assert_weights([-1]) is None
    assert assert_weights([1, 12, 3, 5]) is None
    assert assert_weights(np.array([1, 2, 3, -5])) is None
    assert assert_weights(pd.Series([-1, 2, 3, 5])) is None


def test_assert_not_unique():
    assert assert_not_unique([1]) is None
    assert assert_not_unique([1, 2, 3, 5]) is None
    assert assert_not_unique(np.array([1, 2, 3, 5])) is None
    assert assert_not_unique(pd.Series([1, 2, 3, 5])) is None


@pytest.mark.xfail(strict=True, reason="Duplicate values")
def test_assert_not_unique2():
    assert assert_not_unique([1, 3, 3, 5]) is None
    assert assert_not_unique(np.array([1, 2, 3, 5, 5])) is None
    assert assert_not_unique(pd.Series([1, 1, 2, 3, 5])) is None


def test_assert_response_status11():
    assert assert_response_status("rr", None) is None


@pytest.mark.xfail(strict=True, reason="response status missing")
def test_assert_response_status12():
    assert assert_response_status("in", None) is None
    assert assert_response_status("nr", None) is None
    assert assert_response_status("uk", None) is None


@pytest.mark.xfail(strict=True, reason="response status missing")
def test_assert_response_status0():
    assert assert_response_status(None, {"in": "inelgible"}) is None
    assert assert_response_status(None, {"nr": "inelgible"}) is None


@pytest.mark.xfail(strict=True, reason="Not in the standard dictionary")
def test_assert_response_status13():
    assert assert_response_status("inn", None) is None
    assert assert_response_status("rRR", None) is None
    assert assert_response_status("nonresponse", None) is None
    assert assert_response_status("uuk", None) is None


def test_assert_response_status21():
    assert assert_response_status("in", {"in": "ineligible"}) is None
    assert assert_response_status("in", {"iN": 1}) is None
    assert assert_response_status("nr", {"in": "ineligible"}) is None
    assert assert_response_status("nr", {"in": 1}) is None
    assert assert_response_status("nr", {"in": "ineligible", "nr": "nonresponse"}) is None


@pytest.mark.xfail(strict=True, reason="Not in the standard dictionary")
def test_assert_response_status22():
    assert assert_response_status("in", {"inn": "ineligible"}) is None
    assert assert_response_status("in", {"in2": 1}) is None
    assert assert_response_status("nr", {"ineligible": "ineligible"}) is None
    assert assert_response_status("nr", {"Nonresp": 1}) is None
    assert assert_response_status("nr", {"nr2": "ineligible", "nr": "nonresponse"}) is None


def test_assert_brr_number_psus_sucesses():
    assert assert_brr_number_psus(np.array([1, 2, 4, 3, 3, 4])) is None
    assert assert_brr_number_psus(np.array([1, 2, 3, 4])) is None
    assert assert_brr_number_psus(np.array([1, 4])) is None


@pytest.mark.xfail(strict=True, reason="Not a multiple of 2")
def test_assert_brr_number_psus_fails():
    assert assert_brr_number_psus(np.array([1, 2, 3])) is None


"""Tests for the stratum-combining and fpc-validation helpers."""

import numpy as np
import pytest

from samplics.utils.checks import _combine_strata, _validate_fpc


def test_combine_strata_simultaneous_relabel_not_transitive():
    stratum = np.array(["A", "A", "B", "B", "C", "C"])
    out = _combine_strata({"A": "B", "B": "C"}, stratum)
    # rows of A land in B, rows of B land in C: A's rows are NOT moved on to C
    assert (out == np.array(["B", "B", "C", "C", "C", "C"])).all()


def test_combine_strata_unlisted_strata_stay():
    stratum = np.array(["A", "B", "C", "D"])
    out = _combine_strata({"A": "C"}, stratum)
    assert (out == np.array(["C", "B", "C", "D"])).all()


def test_combine_strata_target_may_be_existing_stratum():
    stratum = np.array([1, 1, 2, 2, 3, 3])
    out = _combine_strata({3: 2}, stratum)
    assert (out == np.array([1, 1, 2, 2, 2, 2])).all()
    assert out.dtype == stratum.dtype


def test_combine_strata_long_target_not_truncated():
    stratum = np.array(["S1", "S1", "S2", "S2"])  # numpy width is 2 chars
    out = _combine_strata({"S1": "A_MUCH_LONGER_NAME", "S2": "A_MUCH_LONGER_NAME"}, stratum)
    assert (out == "A_MUCH_LONGER_NAME").all()


def test_combine_strata_does_not_mutate_caller_array():
    stratum = np.array(["A", "A", "B", "B"])
    snapshot = stratum.copy()
    mapping = {"A": "C", "B": "C"}
    _combine_strata(mapping, stratum)
    assert (stratum == snapshot).all()
    assert mapping == {"A": "C", "B": "C"}


def test_combine_strata_object_dtype_labels():
    stratum = np.array(["A", "A", "B", "B"], dtype=object)
    out = _combine_strata({"A": "LONG_TARGET", "B": "LONG_TARGET"}, stratum)
    assert (out == "LONG_TARGET").all()


def test_combine_strata_invalid_mappings():
    stratum = np.array(["A", "A", "B", "B"])
    with pytest.raises(ValueError):  # not a mapping
        _combine_strata(None, stratum)
    with pytest.raises(ValueError):  # empty mapping
        _combine_strata({}, stratum)
    with pytest.raises(ValueError):  # missing source
        _combine_strata({"A": "C", "ZZZ": "C"}, stratum)
    with pytest.raises(ValueError):  # int target for string labels
        _combine_strata({"A": 3}, stratum)
    with pytest.raises(ValueError):  # bool target
        _combine_strata({"A": True}, stratum)
    with pytest.raises(ValueError):  # stringified source for int labels
        _combine_strata({"1": "2"}, np.array([1, 1, 2, 2]))
    with pytest.raises(ValueError):  # stringified target for int labels
        _combine_strata({1: "2"}, np.array([1, 1, 2, 2]))
    with pytest.raises(ValueError):  # float labels are not supported
        _combine_strata({1.5: 2.5}, np.array([1.5, 1.5, 2.5, 2.5]))


def test_validate_fpc_scalar_broadcast_to_final_strata():
    stratum = np.array(["C", "C", "D", "D"])
    assert _validate_fpc(0.5, stratum, combined=True) == {"C": 0.5, "D": 0.5}


def test_validate_fpc_dict_key_order_irrelevant():
    stratum = np.array(["C", "C", "D", "D"])
    out1 = _validate_fpc({"C": 0.9, "D": 0.8}, stratum, combined=True)
    out2 = _validate_fpc({"D": 0.8, "C": 0.9}, stratum, combined=True)
    assert out1 == out2
    assert list(out1.keys()) == list(out2.keys())


def test_validate_fpc_dict_must_cover_exactly_final_strata():
    stratum = np.array(["C", "C", "D", "D"])
    with pytest.raises(ValueError):  # missing D
        _validate_fpc({"C": 0.5}, stratum, combined=True)
    with pytest.raises(ValueError):  # extra key (disappeared stratum)
        _validate_fpc({"C": 0.5, "D": 0.5, "A": 0.5}, stratum, combined=True)


def test_validate_fpc_values_must_be_finite_reals_in_unit_interval():
    stratum = np.array(["C", "C"])
    for bad in (True, False, "0.5", None, -0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            _validate_fpc(bad, stratum)
        with pytest.raises(ValueError):
            _validate_fpc({"C": bad}, stratum)
    # boundary values are valid
    assert _validate_fpc(0, stratum) == {"C": 0}
    assert _validate_fpc(1, stratum) == {"C": 1}


def test_validate_fpc_unstratified_scalar():
    assert _validate_fpc(0.7, np.array(())) == 0.7
    with pytest.raises(ValueError):
        _validate_fpc({"C": 0.5}, np.array(()))


def test_validate_fpc_array_rejected_when_combined():
    stratum = np.array(["C", "C", "D", "D"])
    with pytest.raises(ValueError):
        _validate_fpc(np.array([0.5, 0.5, 0.6, 0.6]), stratum, combined=True)
    # without combining, a per-observation array keeps its historical meaning
    out = _validate_fpc(np.array([0.5, 0.5, 0.6, 0.6]), stratum, combined=False)
    assert out == {"C": 0.5, "D": 0.6}
