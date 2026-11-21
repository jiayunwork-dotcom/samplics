"""Sampling selection module

The module has one main class called *SampleSelection* which provides a number of random selection
methods and associated probability of selection. All the samping techniques implemented in this
modules are discussed in the following reference book: Cochran, W.G. (1977) [#c1977]_,
Kish, L. (1965) [#k1965]_, and Lohr, S.L. (2010) [#l2010]_. Furthermore, Brewer, K.R.W. and
Hanif, M. (1983) [#bh1983]_ provides comprehensive and detailed descriptions of these complex
sampling algorithms.

.. [#c1977] Cochran, W.G. (1977), *Sampling Techniques, 3rd edn.*, Jonh Wiley & Sons, Inc.
.. [#k1965] Kish, L. (1965), *Survey Sampling*, Jonh Wiley & Sons, Inc.
.. [#l2010] Lohr, S.L. (2010), *Sampling: Design and Analysis, 2nd edn.*, Cengage Learning, Inc.
.. [#bh1983] Brewer, K.R.W. and Hanif, M. (1983), *Sampling With Unequal Probabilities*,
   Springer-Verlag New York, Inc
"""

from __future__ import annotations

import math
from dataclasses import InitVar, dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd

from samplics.sampling.conditional_poisson import ConditionalPoissonDesign
from samplics.utils.errors import (CertaintyError, MethodError, ProbError)
from samplics.utils.formats import (data_to_dict, numpy_array, remove_nans,
                                    sample_units)
from samplics.utils.types import (Array, DictStrBool, DictStrFloat, DictStrInt,
                                  DictStrNum, Number, SelectMethod)

_CONDITIONAL_POISSON_CONFLICTS = (
    "mos",
    "samp_rate",
    "probs",
)


def _is_integer_scalar(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _unique_labels(labels: np.ndarray) -> list:
    """Unique labels preserving first-appearance order, as Python scalars."""
    seen: list = []
    for value in labels:
        if value not in seen:
            seen.append(value)
    return seen


def _contains_boolean(values: object) -> bool:
    """Whether any element is a Python or NumPy boolean."""
    if isinstance(values, (bool, np.bool_)):
        return True
    if isinstance(values, np.ndarray):
        if values.dtype.kind == "b":
            return True
        if values.dtype.kind == "O" and values.ndim == 1:
            return any(isinstance(value, (bool, np.bool_)) for value in values)
        return False
    if isinstance(values, (list, tuple)):
        return any(isinstance(value, (bool, np.bool_)) for value in values)
    return False


@dataclass
class SampleSelection:
    # __slot__ = (
    #     "method",
    #     "strat",
    #     "wr",
    #     "pop_size",
    #     "samp_size",
    #     "samp_rate",
    #     "strata",
    #     "shuffle",
    #     "remove_nan",
    # )

    method: InitVar[SelectMethod] = field(init=True, default=False)
    strat: InitVar[bool] = field(init=True, default=False)
    wr: InitVar[bool] = field(init=True, default=False)

    pop_size: Union[DictStrNum, Number] = field(init=False, default=0)
    samp_size: Union[DictStrNum, Number] = field(init=False, default=0)
    samp_rate: Union[DictStrNum, Number] = field(init=False, default=0)
    strata: list = field(init=False, default=list)

    # sample: np.ndarray = field(init=False, default=None)
    # hits: np.ndarray = field(init=False, default=None)
    # probs: np.ndarray = field(init=False, default=None)zzzzzz

    shuffle: InitVar[bool] = field(init=True, default=False)
    remove_nan: InitVar[bool] = field(init=True, default=False)

    def __init__(
        self,
        method: SelectMethod,
        strat: bool = False,
        wr: bool = True,
    ) -> None:
        self.method: SelectMethod = method
        self.strat = strat
        self.fpc: Union[DictStrNum, Number] = {}
        self.wr = wr
        if method in (SelectMethod.srs_wr, SelectMethod.pps_wr):
            self.wr = True
        elif method in (
            SelectMethod.srs_wor,
            SelectMethod.pps_brewer,
            SelectMethod.pps_hv,
            SelectMethod.pps_murphy,
            SelectMethod.pps_rs,
            # SelectMethod.pps_sys,
        ):
            self.wr = False
        else:
            self.wr = wr

    @staticmethod
    def _to_dataframe(
        samp_unit: np.ndarray,
        stratum: np.ndarray,
        mos: np.ndarray,
        sample: np.ndarray,
        hits: np.ndarray,
        probs: np.ndarray,
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "_samp_unit": samp_unit,
                "_stratum": stratum,
                "_mos": mos,
                "_sample": sample,
                "_hits": hits,
                "_probs": probs,
            }
        )
        df.reset_index(drop=True, inplace=True)

        if stratum.shape in ((), (0,)):
            df.drop(columns=["_stratum"], inplace=True)
        if mos.shape in ((), (0,)):
            df.drop(columns=["_mos"], inplace=True)

        return df

    @staticmethod
    def _calculate_samp_size(
        strat: bool,
        pop_size: Union[DictStrInt, int],
        samp_rate: Union[DictStrInt, int],
    ) -> Union[DictStrInt, int]:
        if strat:
            samp_size = {}
            for s in samp_rate:
                samp_size[s] = math.ceil(samp_rate[s] * pop_size[s])
            return samp_size
        else:
            return math.ceil(samp_rate * pop_size)

    @staticmethod
    def _calculate_samp_rate(
        strat: bool,
        pop_size: Union[DictStrInt, int],
        samp_size: Union[DictStrInt, int],
    ):
        if strat:
            samp_rate = {}
            for s in samp_size:
                samp_rate[s] = samp_size[s] / pop_size[s]
            return samp_rate
        else:
            return samp_size / pop_size

    @staticmethod
    def _calculate_fpc(
        strat: bool,
        pop_size: Union[DictStrInt, int],
        samp_size: Union[DictStrInt, int],
    ) -> None:
        if strat:
            fpc = {}
            for s in samp_size:
                fpc[s] = np.sqrt((pop_size[s] - samp_size[s]) / (pop_size[s] - 1))
            return fpc
        else:
            return np.sqrt((pop_size - samp_size) / (pop_size - 1))

    @staticmethod
    def _grs_select(
        probs: np.ndarray,
        samp_unit: np.ndarray,
        samp_size: Union[DictStrInt, int],
        stratum: np.ndarray,
        wr: Union[DictStrBool, bool],
    ) -> tuple[np.ndarray, np.ndarray]:
        sample = np.zeros(samp_unit.size).astype(bool)
        hits = np.zeros(samp_unit.size).astype(int)

        if stratum.shape in ((), (0,)):
            sampled_indices = np.random.choice(
                a=samp_unit.size,
                size=samp_size,
                replace=wr,
                p=probs / np.sum(probs),
            )
        else:
            sampled_indices_list = []
            for s in samp_size:
                units_s = stratum == s
                sampled_indices_s = np.random.choice(
                    a=samp_unit[units_s],
                    size=samp_size[s],
                    replace=wr,
                    p=probs[units_s] / np.sum(probs[units_s]),
                )
                sampled_indices_list.append(sampled_indices_s)
            sampled_indices = np.array(
                [val for sublist in sampled_indices_list for val in sublist]
            ).flatten()

        indices_s, hits_s = np.unique(sampled_indices, return_counts=True)
        sample[indices_s] = True
        hits[indices_s] = hits_s

        return sample, hits

    @staticmethod
    def _anycertainty(
        samp_size: Union[DictStrInt, int],
        stratum: Optional[np.ndarray],
        mos: np.ndarray,
    ) -> bool:
        certainty = False
        if stratum.shape not in ((), (0,)) and isinstance(samp_size, dict):
            for s in np.unique(stratum):
                stratum_units = stratum == s
                mos_s = mos[stratum_units]
                if (samp_size[s] * mos_s / np.sum(mos_s) >= 1).any():
                    certainty = True
                    break
        elif isinstance(samp_size, (int, float)):
            if (samp_size * mos / np.sum(mos) >= 1).any():
                certainty = True

        return certainty

    # SRS methods
    def _srs_inclusion_probs(
        self,
        samp_unit: np.ndarray,
        samp_size: Union[DictStrInt, int],
        stratum: np.ndarray,
    ) -> np.ndarray:
        number_units = samp_unit.size
        if stratum.shape not in ((), (0,)) and isinstance(samp_size, dict):
            incl_probs = np.zeros(number_units) * np.nan
            for s in np.unique(stratum):
                number_units_s = samp_unit[stratum == s].size
                incl_probs[stratum == s] = samp_size[s] / number_units_s
        elif isinstance(samp_size, (int, float)):
            number_units = samp_unit.size
            incl_probs = np.ones(number_units) * samp_size / number_units
        else:
            raise TypeError("samp_size has the wrong type!")

        return incl_probs

    # PPS methods
    def _pps_inclusion_probs(
        self,
        samp_unit: np.ndarray,
        samp_size: Union[DictStrInt, int],
        mos: np.ndarray,
        stratum: np.ndarray,
    ) -> np.ndarray:
        if isinstance(samp_size, dict):
            number_units = samp_unit.size
            incl_probs = np.zeros(number_units) * np.nan
            for s in np.unique(stratum):
                stratum_units = stratum == s
                mos_s = mos[stratum_units]
                incl_probs[stratum_units] = samp_size[s] * mos_s / np.sum(mos_s)
        else:
            incl_probs = samp_size * mos / np.sum(mos)

        return incl_probs

    @staticmethod
    def _pps_sys_select_core(
        samp_unit: np.ndarray,
        samp_size: int,
        mos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cumsize = np.append(0, np.cumsum(mos))
        samp_interval = cumsize[-1] / samp_size
        random_start = np.random.random_sample() * samp_interval
        random_picks = random_start + samp_interval * np.linspace(0, samp_size - 1, samp_size)

        hits = np.zeros(samp_unit.size).astype(int)
        for k in range(cumsize.size - 1):
            for ll in range(random_picks.size):
                if cumsize[k] < random_picks[ll] <= cumsize[k + 1]:
                    hits[k] += 1

        return np.asarray(hits >= 1), hits

    def _pps_sys_select(
        self,
        samp_unit: np.ndarray,
        samp_size: Union[DictStrInt, int],
        stratum: np.ndarray,
        mos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        samp_unit = sample_units(samp_unit, unique=True)
        samp_size = data_to_dict(samp_size, self.strat, stratum)

        sample = hits = np.zeros(samp_unit.size).astype(int)

        if isinstance(samp_size, dict):
            for s in np.unique(stratum):
                stratum_units = stratum == s
                (
                    sample[stratum_units],
                    hits[stratum_units],
                ) = self._pps_sys_select_core(
                    samp_unit=samp_unit[stratum_units],
                    samp_size=samp_size[s],
                    mos=mos[stratum_units],
                )
        else:
            sample, hits = self._pps_sys_select_core(
                samp_unit=samp_unit, samp_size=samp_size, mos=mos
            )

        return sample, hits

    @staticmethod
    def _pps_hv_select_core(
        samp_unit: np.ndarray,
        samp_size: int,
        mos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pop_size = samp_unit.size
        all_indices = np.arange(pop_size)
        size_order = mos.argsort()
        all_indices_sorted = all_indices[size_order]
        mos_sorted = mos[size_order]
        probs_sorted = np.append(mos_sorted / np.sum(mos_sorted), 1 / samp_size)
        last_nplus1_probs = probs_sorted[range(-(samp_size + 1), 0)]
        diff_probs = np.ediff1d(last_nplus1_probs)
        s = np.sum(probs_sorted[0 : (pop_size - samp_size)])
        initial_probs_selection = (
            samp_size
            * diff_probs
            * (1 + np.linspace(1, samp_size, samp_size) * probs_sorted[pop_size - samp_size] / s)
        )
        probs_sorted = np.delete(probs_sorted, -1)
        selected_i = np.random.choice(np.arange(0, samp_size), size=1, p=initial_probs_selection)[
            0
        ]
        sampled_indices = all_indices_sorted[selected_i + 1 : samp_size]

        notsampled_indices = np.delete(all_indices_sorted, sampled_indices)
        notsampled_probs = np.delete(probs_sorted, sampled_indices)
        p_denominator = (
            s
            + np.linspace(
                1,
                pop_size - samp_size + selected_i + 1,
                pop_size - samp_size + selected_i + 1,
            )
            * probs_sorted[pop_size - samp_size]
        )
        p_starts = notsampled_probs / p_denominator
        range_part2 = range(pop_size - samp_size, pop_size - samp_size + selected_i)
        p_starts[range_part2] = probs_sorted[pop_size - samp_size] / p_denominator[range_part2]
        p_starts_sum = np.cumsum(np.flip(p_starts)[range(p_starts.size - 1)])
        p_starts_sum = np.append(np.flip(p_starts_sum), 1)
        p_double_starts = p_starts / p_starts_sum
        p_double_starts[-1] = 0

        start_j = 0
        end_j = pop_size - samp_size + 1
        for ll in np.arange(selected_i + 1):
            sampling_space = range(start_j, end_j)
            p_double_space = p_double_starts[range(start_j, end_j)]
            p_double_space = 1 - (selected_i + 1 - ll) * np.append(0, p_double_space)
            p_double_space = np.delete(p_double_space, -1)
            a_j = (
                (samp_size - ll + 1) * p_starts[range(start_j, end_j)] * np.cumprod(p_double_space)
            )
            indice_j = np.random.choice(sampling_space, size=1, p=a_j / np.sum(a_j))[0]
            selected_j = notsampled_indices[indice_j]
            sampled_indices = np.append(sampled_indices, selected_j)
            start_j = indice_j + 1
            end_j += 1
        sample = hits = np.zeros(samp_unit.size).astype(int)
        sample[sampled_indices] = True
        hits[sampled_indices] = 1

        return sample, hits

    def _pps_hv_select(
        self,
        samp_unit: np.ndarray,
        samp_size: Union[DictStrInt, int],
        stratum: np.ndarray,
        mos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        samp_unit = sample_units(samp_unit, unique=True)
        samp_size = data_to_dict(samp_size, self.strat, stratum)

        sample = hits = np.zeros(samp_unit.size).astype(int)

        if isinstance(samp_size, dict):
            for s in np.unique(stratum):
                stratum_units = stratum == s
                (
                    sample[stratum_units],
                    hits[stratum_units],
                ) = self._pps_hv_select_core(
                    samp_unit=samp_unit[stratum_units],
                    samp_size=samp_size[s],
                    mos=mos[stratum_units],
                )
        else:
            sample, hits = self._pps_hv_select_core(
                samp_unit=samp_unit, samp_size=samp_size, mos=mos
            )

        return sample, hits

    @staticmethod
    def _pps_brewer_select_core(
        samp_unit: np.ndarray,
        samp_size: int,
        mos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        all_indices = np.arange(samp_unit.size)
        all_probs = mos / np.sum(mos)
        working_probs = all_probs * (1 - all_probs) / (1 - samp_size * all_probs)
        working_probs = working_probs / np.sum(working_probs)
        sampled_indices = np.random.choice(all_indices, 1, p=working_probs)
        sample = hits = np.zeros(samp_unit.size).astype(int)
        for s in np.arange(1, samp_size):
            remaining_indices = np.delete(all_indices, sampled_indices)
            remaining_probs = np.delete(all_probs, sampled_indices)
            remaining_probs = (
                remaining_probs * (1 - remaining_probs) / (1 - (samp_size - s) * remaining_probs)
            )
            remaining_probs = remaining_probs / sum(remaining_probs)
            current_selection = np.random.choice(remaining_indices, 1, p=remaining_probs)
            sampled_indices = np.append(sampled_indices, current_selection)

        sample[sampled_indices] = True
        hits[sampled_indices] = 1

        return sample, hits

    def _pps_brewer_select(
        self,
        samp_unit: np.ndarray,
        samp_size: Union[DictStrInt, int],
        stratum: np.ndarray,
        mos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        samp_unit = sample_units(samp_unit, unique=True)
        samp_size = data_to_dict(samp_size, self.strat, stratum)

        sample = hits = np.zeros(samp_unit.size).astype(int)

        if isinstance(samp_size, dict):
            for s in np.unique(stratum):
                stratum_units = stratum == s
                (
                    sample[stratum_units],
                    hits[stratum_units],
                ) = self._pps_brewer_select_core(
                    samp_unit=samp_unit[stratum_units],
                    samp_size=samp_size[s],
                    mos=mos[stratum_units],
                )
        else:
            sample, hits = self._pps_brewer_select_core(
                samp_unit=samp_unit, samp_size=samp_size, mos=mos
            )

        return sample, hits

    @staticmethod
    def _pps_murphy_select_core(
        samp_unit: np.ndarray,
        samp_size: int,
        mos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if samp_size != 2:
            raise ValueError(
                "For the Murphy's selection algorithm, sample size must be equal to 2"
            )
        all_indices = np.arange(samp_unit.size)
        all_probs = mos / np.sum(mos)
        sampled_indices = np.random.choice(all_indices, 1, p=all_probs)
        remaining_indices = np.delete(all_indices, sampled_indices)
        remaining_probs = np.delete(all_probs, sampled_indices)
        remaining_probs = remaining_probs / (1 - all_probs[sampled_indices])
        current_selection = np.random.choice(remaining_indices, 1, p=remaining_probs)
        sampled_indices = np.append(sampled_indices, current_selection)

        sample = hits = np.zeros(samp_unit.size).astype(int)
        sample[sampled_indices] = True
        hits[sampled_indices] = 1

        return sample, hits

    def _pps_murphy_select(
        self,
        samp_unit: np.ndarray,
        samp_size: Union[DictStrInt, int],
        stratum: np.ndarray,
        mos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        samp_unit = sample_units(samp_unit, unique=True)
        samp_size = data_to_dict(samp_size, self.strat, stratum)

        sample = hits = np.zeros(samp_unit.size).astype(int)

        if isinstance(samp_size, dict):
            for s in np.unique(stratum):
                stratum_units = stratum == s
                (
                    sample[stratum_units],
                    hits[stratum_units],
                ) = self._pps_murphy_select_core(
                    samp_unit=samp_unit[stratum_units],
                    samp_size=samp_size[s],
                    mos=mos[stratum_units],
                )
        else:
            sample, hits = self._pps_murphy_select_core(
                samp_unit=samp_unit, samp_size=samp_size, mos=mos
            )

        return sample, hits

    @staticmethod
    def _pps_rs_select_core(
        samp_unit: np.ndarray,
        samp_size: int,
        mos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        all_indices = np.arange(samp_unit.shape[0])
        all_probs = mos / np.sum(mos)

        stop = False
        sample = hits = np.zeros(samp_unit.shape[0]).astype(int)
        sampled_indices = None
        while stop is not True:
            sampled_indices = np.random.choice(all_indices, 1, p=all_probs)
            remaining_indices = np.delete(all_indices, sampled_indices)
            remaining_probs = all_probs / (1 - samp_size * all_probs)
            remaining_probs = np.delete(remaining_probs, sampled_indices)
            remaining_probs = remaining_probs / np.sum(remaining_probs)
            remaining_sample = np.random.choice(
                remaining_indices, samp_size - 1, p=remaining_probs
            )
            sampled_indices, counts = np.unique(
                np.append(sampled_indices, remaining_sample), return_counts=True
            )
            if (counts == 1).all():
                stop = True

        sample[sampled_indices] = True
        hits[sampled_indices] = 1

        return sample, hits

    def _pps_rs_select(
        self,
        samp_unit: np.ndarray,
        samp_size: Union[DictStrInt, int],
        stratum: np.ndarray,
        mos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        samp_unit = sample_units(samp_unit, unique=True)
        samp_size = data_to_dict(samp_size, self.strat, stratum)

        sample = hits = np.zeros(samp_unit.size).astype(int)

        if isinstance(samp_size, dict):
            for s in np.unique(stratum):
                stratum_units = stratum == s
                (
                    sample[stratum_units],
                    hits[stratum_units],
                ) = self._pps_rs_select_core(
                    samp_unit=samp_unit[stratum_units],
                    samp_size=samp_size[s],
                    mos=mos[stratum_units],
                )
        else:
            sample, hits = self._pps_rs_select_core(
                samp_unit=samp_unit, samp_size=samp_size, mos=mos
            )

        return sample, hits

    def _sys_inclusion_probs(
        self,
        samp_unit: np.ndarray,
        samp_size: Union[DictStrInt, int],
        stratum: np.ndarray,
        samp_rate: Union[DictStrFloat, float],
    ) -> np.ndarray:
        pass

    @staticmethod
    def _sys_selection_method(
        samp_unit: np.ndarray,
        samp_size: Optional[int],
        samp_rate: Optional[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        if samp_size is not None and samp_rate is not None:
            raise AssertionError(
                """Both samp_size and samp_rate are provided.
                Only one of the two parameters should be specified."""
            )

        if samp_rate is None and samp_size is None:
            raise AssertionError("samp_size or samp_rate must be provided!")

        if samp_rate is not None:
            samp_size = round(samp_rate * samp_unit.size)
        samp_interval = int(samp_unit.size / samp_size)  # same as 1 / samp_rate

        random_start = np.random.choice(range(0, samp_interval))
        random_picks = random_start + samp_interval * np.linspace(
            0, samp_size - 1, samp_size
        ).astype(int)
        hits = np.zeros(samp_unit.size).astype(int)
        hits[random_picks] = 1

        return hits == 1, hits

    def _sys_select(
        self,
        samp_unit: np.ndarray,
        samp_size: Union[DictStrInt, int],
        stratum: np.ndarray,
        samp_rate: Union[DictStrFloat, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        sample = hits = np.zeros(samp_unit.size).astype(int)
        if self.strat and (isinstance(samp_size, dict) or isinstance(samp_rate, dict)):
            for s in np.unique(stratum):
                samp_size_s = None
                if samp_size is not None:
                    samp_size_s = samp_size[s]
                samp_rate_s = None
                if samp_rate is not None:
                    samp_rate_s = samp_rate[s]
                stratum_units = stratum == s
                (
                    sample[stratum_units],
                    hits[stratum_units],
                ) = self._sys_selection_method(samp_unit[stratum_units], samp_size_s, samp_rate_s)
        elif isinstance(samp_size, int) or isinstance(samp_rate, float):
            samp_size_n = None if samp_size is None else samp_size
            samp_rate_n = None if samp_rate is None else samp_rate
            sample, hits = self._sys_selection_method(samp_unit, samp_size_n, samp_rate_n)

        return sample, hits

    @staticmethod
    def _validate_frame_and_parameters(
        samp_unit: Array,
        probabilities: Array,
        stratum: Optional[Array],
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], bool]:
        """Validate and copy the design inputs. Raises ValueError on failure."""
        units = numpy_array(samp_unit)
        units = np.asarray(units)
        if units.ndim != 1:
            raise ValueError("samp_unit must be a one-dimensional array of unit identifiers.")
        if units.size == 0:
            raise ValueError("samp_unit must contain at least one unit.")
        if units.dtype.kind not in ("U", "S", "O", "i", "u"):
            raise ValueError("samp_unit must contain string or integer identifiers.")
        if units.dtype.kind == "O":
            normalized = []
            for value in units:
                value = value.item() if hasattr(value, "item") else value
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, int)):
                    raise ValueError(
                        "samp_unit must contain string or integer identifiers."
                    )
                normalized.append(value)
            units = np.asarray(normalized, dtype=object)
        elif units.dtype.kind == "S":
            units = units.astype(str)
        units = np.array(units, copy=True)
        for value in units:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError("Boolean unit identifiers are not accepted.")
        if len(set(units.tolist())) != units.size:
            raise ValueError("samp_unit must contain unique unit identifiers.")

        if _contains_boolean(probabilities):
            raise ValueError("probabilities must be real numbers, not booleans.")
        probs = np.asarray(probabilities)
        if probs.dtype.kind == "b":
            raise ValueError("probabilities must be real numbers, not booleans.")
        if probs.dtype.kind == "O":
            for value in probs.ravel():
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    raise ValueError("probabilities must be real numbers.")
            probs = probs.astype(float)
        elif probs.dtype.kind not in ("i", "u", "f"):
            raise ValueError("probabilities must be real numbers.")
        else:
            probs = np.asarray(probabilities, dtype=float)
        if probs.ndim != 1 or probs.size != units.size:
            raise ValueError("probabilities must be a one-dimensional array aligned with samp_unit.")
        if not np.all(np.isfinite(probs)):
            raise ValueError("probabilities must be finite.")
        if (probs <= 0.0).any() or (probs > 1.0).any():
            raise ValueError("probabilities must lie in the interval (0, 1].")
        probs = np.array(probs, dtype=float, copy=True)

        if stratum is None:
            return units, probs, None, False

        labels = np.asarray(stratum)
        if labels.ndim != 1 or labels.size != units.size:
            raise ValueError("stratum must be a one-dimensional array aligned with samp_unit.")
        if labels.dtype.kind not in ("U", "S", "O", "i", "u", "f"):
            raise ValueError("stratum must contain string or integer labels.")
        if labels.dtype.kind == "f" and not np.all(np.isfinite(labels)):
            raise ValueError("stratum must not contain missing labels.")
        normalized_labels = np.empty(labels.size, dtype=object)
        for index, value in enumerate(labels):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError("stratum must contain string or integer labels.")
            if isinstance(value, str):
                if value in ("", "nan", "None"):
                    raise ValueError("stratum must not contain missing labels.")
                normalized_labels[index] = value
            elif isinstance(value, (int, np.integer)):
                normalized_labels[index] = int(value)
            elif isinstance(value, (float, np.floating)):
                if not np.isfinite(value):
                    raise ValueError("stratum must not contain missing labels.")
                normalized_labels[index] = float(value)
            else:
                raise ValueError("stratum must contain string or integer labels.")
        return units, probs, normalized_labels, True

    @staticmethod
    def _validate_sample_size(
        samp_size: Union[DictStrInt, int],
        stratum: Optional[np.ndarray],
        units: np.ndarray,
        probabilities: np.ndarray,
    ) -> Union[int, dict]:
        if stratum is None:
            if not _is_integer_scalar(samp_size):
                raise ValueError("samp_size must be a non-boolean integer when not stratified.")
            size = int(samp_size)
            labels = np.zeros(units.size, dtype=np.intp)
            mapping = {0: size}
        else:
            if not isinstance(samp_size, dict) or isinstance(samp_size, bool):
                raise ValueError(
                    "samp_size must be a dictionary mapping every stratum to an integer."
                )
            labels = stratum
            observed = _unique_labels(labels)
            provided = list(samp_size.keys())
            if len(provided) != len(observed) or any(
                key not in observed for key in provided
            ) or any(key not in provided for key in observed):
                raise ValueError(
                    "samp_size must cover exactly the strata present in stratum."
                )
            mapping = {}
            for key, value in samp_size.items():
                if not _is_integer_scalar(value):
                    raise ValueError("samp_size values must be non-boolean integers.")
                mapping[key] = int(value)

        for label in _unique_labels(labels):
            mask = labels == label
            population = int(mask.sum())
            size = mapping[label]
            if size < 1 or size > population:
                raise ValueError(
                    f"Sample size must be between 1 and the stratum population size "
                    f"(stratum {label!r})."
                )
            probability = probabilities[mask]
            nb_mandatory = int((probability == 1.0).sum())
            remaining = size - nb_mandatory
            nb_optional = population - nb_mandatory
            if remaining < 0 or remaining > nb_optional:
                raise ValueError(
                    "The number of certainty (probability 1) units is incompatible "
                    f"with the sample size of stratum {label!r}."
                )
            if nb_optional > 0 and remaining == 0:
                raise ValueError(
                    "Zero remaining selections with non-certainty units present would "
                    f"assign zero inclusion probability (stratum {label!r})."
                )
        return int(samp_size) if stratum is None else mapping

    @staticmethod
    def _check_conditional_poisson_conflicts(**options: object) -> None:
        for name in _CONDITIONAL_POISSON_CONFLICTS:
            if options.get(name) is not None:
                raise ValueError(
                    f"The option '{name}' is not supported with conditional Poisson "
                    "sampling; provide probabilities instead."
                )
        if options.get("remove_nan", False):
            raise ValueError(
                "remove_nan=True is not supported with conditional Poisson sampling."
            )

    def prepare_design(
        self,
        samp_unit: Array,
        samp_size: Union[DictStrInt, int],
        *,
        probabilities: Array,
        stratum: Optional[Array] = None,
        mos: Optional[Array] = None,
        samp_rate: Optional[Union[DictStrFloat, float]] = None,
        probs: Optional[Array] = None,
        remove_nan: bool = False,
    ) -> ConditionalPoissonDesign:
        """Prepare a conditional Poisson design without sampling.

        Args:
            samp_unit: non-empty, unique one-dimensional unit identifiers.
            samp_size: integer sample size, or a mapping stratum -> integer when
                stratified.  Must exactly cover the observed strata.
            probabilities: per-row Bernoulli parameters in (0, 1], aligned with
                ``samp_unit``.  These are the pre-conditioning parameters, not
                the final inclusion probabilities.
            stratum: optional equal-length stratum labels without missing values.
            mos, samp_rate, probs: not supported; passing any raises ValueError.
            remove_nan: not supported; ``True`` raises ValueError.

        Returns:
            ConditionalPoissonDesign: reusable exact design.  No random numbers
            are consumed and the NumPy global random state is left untouched.
        """
        if self.method != SelectMethod.conditional_poisson:
            raise MethodError(
                "prepare_design() requires SelectMethod.conditional_poisson."
            )
        self._check_conditional_poisson_conflicts(
            mos=mos, samp_rate=samp_rate, probs=probs, remove_nan=remove_nan
        )
        units, probs_validated, labels, stratified = self._validate_frame_and_parameters(
            samp_unit, probabilities, stratum
        )
        size = self._validate_sample_size(samp_size, labels, units, probs_validated)
        return ConditionalPoissonDesign(
            samp_unit=units,
            probabilities=probs_validated,
            sample_size=size,
            stratum=labels,
            stratified=stratified,
        )

    def inclusion_probs(
        self,
        samp_unit: np.ndarray,
        samp_size: Optional[Union[DictStrInt, int]] = None,
        stratum: Optional[np.ndarray] = None,
        mos: Optional[np.ndarray] = None,
        samp_rate: Optional[Union[DictStrFloat, float]] = None,
        *,
        probabilities: Optional[Array] = None,
    ) -> np.ndarray:
        if probabilities is not None:
            return self._conditional_inclusion_probs(
                samp_unit=samp_unit,
                samp_size=samp_size,
                stratum=stratum,
                mos=mos,
                samp_rate=samp_rate,
                probabilities=probabilities,
            )
        if self.method == SelectMethod.conditional_poisson:
            raise MethodError(
                "Conditional Poisson inclusion probabilities require the "
                "'probabilities' argument."
            )

        samp_unit = sample_units(samp_unit, unique=True)

        samp_size_temp: Union[DictStrInt, int]
        if stratum is not None:
            stratum = numpy_array(stratum)
            if isinstance(samp_size, (int, float)):
                strata = np.unique(stratum)
                samp_size_temp = dict(zip(strata, np.repeat(int(samp_size), strata.shape[0])))
            elif isinstance(samp_size, dict):
                samp_size_temp = samp_size.copy()
            else:
                raise TypeError("samp_size or samp_rate has the wrong type")
        else:
            if isinstance(samp_size, (int, float)):
                samp_size_temp = int(samp_size)
            else:
                raise TypeError("samp_size or samp_rate has the wrong type")

        mos = numpy_array(mos) if mos is not None else np.ones(samp_unit.shape[0])

        samp_size_temp = data_to_dict(samp_size_temp, self.strat, stratum)

        if self.method in (SelectMethod.srs_wr, SelectMethod.srs_wor):
            incl_probs = self._srs_inclusion_probs(samp_unit, samp_size, stratum)
        elif self.method in (
            SelectMethod.pps_sys,
            SelectMethod.pps_hv,
            SelectMethod.pps_brewer,
            SelectMethod.pps_murphy,
            SelectMethod.pps_rs,
        ):
            if self._anycertainty(samp_size_temp, stratum, mos):
                raise AssertionError("Some clusters are certainties.")
            incl_probs = self._pps_inclusion_probs(samp_unit, samp_size_temp, mos, stratum)
        elif self.method == SelectMethod.sys:
            incl_probs = self._sys_inclusion_probs(samp_unit, samp_size_temp, stratum, samp_rate)
        else:
            raise ValueError("method not valid!")

        return incl_probs

    def joint_inclusion_probs(self) -> None:
        pass

    def _design_from_arguments(
        self,
        samp_unit: Array,
        samp_size: Optional[Union[DictStrInt, int]],
        stratum: Optional[Array],
        probabilities: Array,
    ) -> ConditionalPoissonDesign:
        if samp_size is None:
            raise ValueError("samp_size must be provided for conditional Poisson sampling.")
        return self.prepare_design(
            samp_unit=samp_unit,
            samp_size=samp_size,
            probabilities=probabilities,
            stratum=stratum,
        )

    def _conditional_inclusion_probs(
        self,
        samp_unit: Array,
        samp_size: Optional[Union[DictStrInt, int]],
        stratum: Optional[Array],
        mos: Optional[Array],
        samp_rate: Optional[Union[DictStrFloat, float]],
        probabilities: Array,
    ) -> np.ndarray:
        if self.method != SelectMethod.conditional_poisson:
            raise MethodError(
                "The 'probabilities' argument is only supported for "
                "SelectMethod.conditional_poisson."
            )
        self._check_conditional_poisson_conflicts(
            mos=mos, samp_rate=samp_rate, probs=None, remove_nan=False
        )
        design = self._design_from_arguments(samp_unit, samp_size, stratum, probabilities)
        return design.inclusion_probabilities

    def _conditional_select(
        self,
        samp_unit: Array,
        samp_size: Optional[Union[DictStrInt, int]],
        stratum: Optional[Array],
        mos: Optional[Array],
        samp_rate: Optional[Union[DictStrFloat, float]],
        probs: Optional[Array],
        probabilities: Array,
        random_state: Optional[Union[int, np.random.Generator]],
        to_dataframe: bool,
        sample_only: bool,
        shuffle: bool,
    ) -> Union[tuple[np.ndarray, np.ndarray, np.ndarray], pd.DataFrame]:
        self._check_conditional_poisson_conflicts(
            mos=mos, samp_rate=samp_rate, probs=probs, remove_nan=False
        )
        if shuffle:
            raise ValueError("shuffle is not supported with conditional Poisson sampling.")
        design = self._design_from_arguments(samp_unit, samp_size, stratum, probabilities)
        result = design.draw(random_state=random_state)

        hits = np.zeros(design.frame_size, dtype=int)
        hits[result.indices] = 1
        sample = hits.astype(bool)
        final_probs = design.inclusion_probabilities

        if to_dataframe:
            frame = self._to_dataframe(
                samp_unit=design.samp_unit,
                stratum=np.array(None) if design.stratum is None else design.stratum,
                mos=np.array(None),
                sample=sample,
                hits=hits,
                probs=final_probs,
            )
            if sample_only:
                return frame.loc[frame["_sample"] == 1].reset_index(drop=True)
            return frame
        if sample_only:
            return sample[sample], hits[sample], final_probs[sample]
        return sample, hits, final_probs

    def select(
        self,
        samp_unit: Array,
        samp_size: Optional[Union[DictStrInt, int]] = None,
        stratum: Optional[Array] = None,
        mos: Optional[Array] = None,
        samp_rate: Optional[Union[DictStrFloat, float]] = None,
        probs: Optional[Array] = None,
        shuffle: bool = False,
        to_dataframe: bool = False,
        sample_only: bool = False,
        remove_nan: bool = False,
        *,
        probabilities: Optional[Array] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> Union[tuple[np.ndarray, np.ndarray, np.ndarray], pd.DataFrame]:
        if probabilities is not None:
            if remove_nan:
                raise ValueError(
                    "remove_nan=True is not supported with conditional Poisson sampling."
                )
            return self._conditional_select(
                samp_unit=samp_unit,
                samp_size=samp_size,
                stratum=stratum,
                mos=mos,
                samp_rate=samp_rate,
                probs=probs,
                probabilities=probabilities,
                random_state=random_state,
                to_dataframe=to_dataframe,
                sample_only=sample_only,
                shuffle=shuffle,
            )
        if self.method == SelectMethod.conditional_poisson:
            raise MethodError(
                "Conditional Poisson sampling requires the 'probabilities' argument."
            )
        if random_state is not None:
            raise ValueError(
                "random_state is only supported when 'probabilities' is provided."
            )
        if samp_size is not None and samp_rate is not None:
            raise AssertionError(
                """Both samp_size and samp_rate are provided.
                Only one of the two parameters should be specified."""
            )

        if self.strat and stratum is None:
            raise AssertionError("Stratum must be provided for stratified samples!")

        _samp_unit = numpy_array(samp_unit)
        _stratum = numpy_array(stratum)  # if stratum is not None else None
        _samp_unit = sample_units(_samp_unit, unique=True)
        if self.strat:
            self.strata = np.unique(_stratum).tolist()
            strata, nobs = np.unique(_stratum, return_counts=True)
            self.pop_size = dict(zip(strata, nobs))
        else:
            self.strata = []
            self.pop_size = _samp_unit.shape[0]
        if samp_rate is None:
            self.samp_size = data_to_dict(data=samp_size, strat=self.strat, stratum=_stratum)
            self.samp_rate = self._calculate_samp_rate(
                strat=self.strat, pop_size=self.pop_size, samp_size=self.samp_size
            )
        else:
            self.samp_rate = data_to_dict(data=samp_rate, strat=self.strat, stratum=_stratum)
            self.samp_size = self._calculate_samp_size(
                strat=self.strat, pop_size=self.pop_size, samp_rate=self.samp_rate
            )

        self.fpc = self._calculate_fpc(
            strat=self.strat, pop_size=self.pop_size, samp_size=self.samp_size
        )

        if self.method == SelectMethod.grs:
            _probs = numpy_array(probs)
            if _probs.shape in ((), (0,)) or (_probs < 0).any() or (_probs > 1).any():
                raise ProbError("Probabilities must be between 0 and 1!")
        else:
            _probs = np.array(None)

        _mos = numpy_array(mos)
        if _mos.shape in ((), (0,)) and self.method in (
            SelectMethod.pps_brewer,
            SelectMethod.pps_hv,
            SelectMethod.pps_murphy,
            SelectMethod.pps_rs,
            SelectMethod.pps_sys,
        ):
            raise MethodError("Measure of size (MOS) not provided!")
        elif _mos.shape not in ((), (0,)) and self.method in (
            SelectMethod.pps_brewer,
            SelectMethod.pps_hv,
            SelectMethod.pps_murphy,
            SelectMethod.pps_rs,
            SelectMethod.pps_sys,
        ):
            if self._anycertainty(samp_size=self.samp_size, stratum=_stratum, mos=_mos):
                raise CertaintyError("Some clusters are certainties.")

        _samp_ids = np.linspace(
            start=0, stop=_samp_unit.shape[0] - 1, num=_samp_unit.shape[0], dtype="int"
        )

        if remove_nan:
            items_to_keep = remove_nans(_samp_ids.shape[0], _samp_ids, _stratum, _mos, _probs)
            _samp_ids = _samp_ids[items_to_keep]
            _stratum = _stratum[items_to_keep]
            _mos = _mos[items_to_keep]
            _probs = _probs[items_to_keep]

        suffled_order = None
        if shuffle and self.method in (SelectMethod.sys, SelectMethod.pps_sys):
            suffled_order = np.linspace(
                start=0, stop=self.pop_size - 1, num=self.pop_size, dtype="int"
            )
            np.random.shuffle(suffled_order)
            _samp_unit = _samp_unit[suffled_order]
            if _stratum.shape not in ((), (0,)):
                _stratum = _stratum[suffled_order]
            if self.method == SelectMethod.pps_sys and _mos.shape not in ((), (0,)):
                _mos = _mos[suffled_order]

        if self.method in (SelectMethod.srs_wr, SelectMethod.srs_wor):
            _probs = self._srs_inclusion_probs(
                samp_unit=_samp_ids, samp_size=self.samp_size, stratum=_stratum
            )
            sample, hits = self._grs_select(
                probs=_probs,
                samp_unit=_samp_ids,
                samp_size=self.samp_size,
                stratum=_stratum,
                wr=self.wr,
            )
        elif self.method == SelectMethod.sys:
            sample, hits = self._sys_select(
                samp_unit=_samp_ids,
                samp_size=None,
                samp_rate=self.samp_rate,
                stratum=_stratum,
            )
        elif self.method == SelectMethod.pps_wr:
            _probs = self._pps_inclusion_probs(
                samp_unit=_samp_ids,
                samp_size=self.samp_size,
                samp_rate=self.samp_rate,
                mos=_mos,
                stratum=_stratum,
            )
            sample, hits = self._grs_select(
                probs=_probs / np.sum(probs),
                samp_unit=_samp_ids,
                samp_size=self.samp_size,
                stratum=_stratum,
                wr=self.wr,
            )
        elif self.method == SelectMethod.pps_sys:
            _probs = self.inclusion_probs(
                samp_unit=_samp_ids,
                samp_size=self.samp_size,
                samp_rate=self.samp_rate,
                stratum=_stratum,
                mos=_mos,
            )
            sample, hits = self._pps_sys_select(
                samp_unit=_samp_ids,
                samp_size=self.samp_size,
                stratum=_stratum,
                mos=_mos,
            )
        elif self.method == SelectMethod.pps_brewer:
            _probs = self.inclusion_probs(
                samp_unit=_samp_ids,
                samp_size=self.samp_size,
                samp_rate=self.samp_rate,
                stratum=_stratum,
                mos=_mos,
            )
            sample, hits = self._pps_brewer_select(
                samp_unit=_samp_ids,
                samp_size=self.samp_size,
                stratum=_stratum,
                mos=_mos,
            )
        elif self.method == SelectMethod.pps_hv:
            _probs = self.inclusion_probs(
                samp_unit=_samp_ids,
                samp_size=self.samp_size,
                samp_rate=self.samp_rate,
                stratum=_stratum,
                mos=_mos,
            )
            sample, hits = self._pps_hv_select(
                samp_unit=_samp_ids,
                samp_size=self.samp_size,
                stratum=_stratum,
                mos=_mos,
            )
        elif self.method == SelectMethod.pps_murphy:
            _probs = self.inclusion_probs(
                samp_unit=_samp_ids,
                samp_size=self.samp_size,
                samp_rate=self.samp_rate,
                stratum=_stratum,
                mos=_mos,
            )
            sample, hits = self._pps_murphy_select(
                samp_unit=_samp_ids,
                samp_size=self.samp_size,
                stratum=_stratum,
                mos=_mos,
            )
        elif self.method == SelectMethod.pps_rs:
            _probs = self.inclusion_probs(
                samp_unit=_samp_ids,
                samp_size=self.samp_size,
                samp_rate=self.samp_rate,
                stratum=_stratum,
                mos=_mos,
            )
            sample, hits = self._pps_rs_select(
                samp_unit=_samp_ids,
                samp_size=self.samp_size,
                stratum=_stratum,
                mos=_mos,
            )
        elif self.method == SelectMethod.grs:
            sample, hits = self._grs_select(
                probs=_probs,
                samp_unit=_samp_ids,
                samp_size=self.samp_size,
                stratum=_stratum,
                wr=self.wr,
            )
        else:
            raise ValueError("sampling method not implemented!")

        if shuffle:
            sample = sample[suffled_order]
            hits = hits[suffled_order]

        if to_dataframe and sample_only:
            frame = self._to_dataframe(
                samp_unit=_samp_unit,
                stratum=_stratum,
                mos=_mos,
                sample=sample,
                hits=hits,
                probs=_probs,
            )
            return frame.loc[frame["_sample"] == 1].reset_index(drop=True)
        elif to_dataframe and not sample_only:
            frame = self._to_dataframe(
                samp_unit=_samp_unit,
                stratum=_stratum,
                mos=_mos,
                sample=sample,
                hits=hits,
                probs=_probs,
            )
            return frame
        elif not to_dataframe and sample_only and probs is not None:
            return sample[sample], hits[sample], probs[sample]
        else:
            return sample, hits, _probs
