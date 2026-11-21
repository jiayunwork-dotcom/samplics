import math

from typing import Any, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

from scipy.stats import t as student

from samplics.estimation.expansion import _SurveyEstimator
from samplics.utils.formats import numpy_array, remove_nans
from samplics.utils.types import Array, Number, PopParam, RepMethod


TypeRepEst = TypeVar("TypeRepEst", bound="ReplicateEstimator")


def _method_equal(method_a: object, method_b: object) -> bool:
    """Compare method labels whether given as RepMethod or raw strings."""

    def value_of(m: object) -> str:
        return m.value if isinstance(m, RepMethod) else str(m)

    return value_of(method_a) == value_of(method_b)


def _finite_vector(name: str, arr: object, n: int) -> np.ndarray:
    """Validate a one-dimensional finite real input of length n."""

    if isinstance(arr, bool):
        raise ValueError(f"{name} must not be a boolean.")
    a = numpy_array(arr)
    if a.dtype.kind in ("b", "c", "O", "U", "S"):
        raise ValueError(f"{name} must be a numeric array of finite reals.")
    a = a.astype(np.float64, copy=False)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if a.shape[0] != n:
        raise ValueError(f"{name} must have length {n} to match the calibrated batch.")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain finite values only.")
    return a


def _assert_positive_denominators(
    weights: np.ndarray,
    n: int,
    param: PopParam,
    x: Optional[np.ndarray],
    domain: np.ndarray,
    where: str,
) -> None:
    """Reject mean/ratio runs whose denominator is zero on any reported unit.

    The check covers the full sample plus every replicate and, when domains are
    requested, every domain. Undefined variances are never replaced by zero.
    """

    if param == PopParam.total:
        return
    if param == PopParam.ratio:
        if x is None:
            raise ValueError("x must be provided for ratio estimation.")
        weighted = weights * x[:, None] if weights.ndim == 2 else weights * x
        totals = weighted.sum(axis=0)
    else:  # mean (and any denominator-based parameter)
        totals = weights.sum(axis=0)
    if np.any(totals == 0.0) or np.any(~np.isfinite(totals)):
        raise ZeroDivisionError(
            f"Zero or non-finite denominator for {param.value} in {where}; "
            "the estimate is undefined and no variance was computed."
        )
    if domain.shape not in ((), (0,)):
        for d_label in np.unique(domain):
            mask = (domain == d_label)
            sub = weights[mask, :] if weights.ndim == 2 else weights[mask]
            if param == PopParam.ratio:
                sub_num = (sub * x[mask, None]) if weights.ndim == 2 else sub * x[mask]
                d_totals = sub_num.sum(axis=0)
            else:
                d_totals = sub.sum(axis=0)
            if np.any(d_totals == 0.0) or np.any(~np.isfinite(d_totals)):
                raise ZeroDivisionError(
                    f"Zero or non-finite denominator for {param.value} in domain "
                    f"{d_label!r} ({where}); the domain estimate is undefined."
                )


class ReplicateEstimator(_SurveyEstimator):
    """*ReplicateEstimator* implements replicate-based variance approximations.


    Attributes
        | point_est (dict): point estimate of the parameter of interest.
        | variance (dict): variance estimate of the parameter of interest.
        | stderror (dict): standard error of the parameter of interest.
        | coef_var (dict): estimate of the coefficient of variation.
        | deff (dict): estimate of the design effect due to weighting.
        | lower_ci (dict): estimate of the lower bound of the confidence interval.
        | upper_ci (dict): estimate of the upper bound of the confidence interval.
        | degree_of_freedom (int): degree of freedom for the confidence interval.
        | alpha (float): significant level for the confidence interval
        | strata (list): list of the strata in the sample.
        | domains (list): list of the domains in the sample.
        | method (str): variance estimation method.
        | param (str): the parameter of the population to estimate e.g. total.
        | nb_strata (int): number of strata.
        | nb_psus (int): number of primary sampling units (psus).
        | conservative (bool): indicate whether to produce conservative variance estimates.
        | nb_reps (int): number of replicate weights.
        | rep_coefs (array): coefficients associated to the replicate weights.
        | fay_coef (float): Fay coefficient for the the BRR-Fay algorithm.
        | rand_seed (int): random seed for reproducibility.

    Methods
        | estimate(): produces the point estimate of the parameter of interest with the associated
        |   measures of precision.
    """

    def __init__(
        self,
        method: str,
        param: str,
        rep_weight_cls: Optional[Any] = None,
        fay_coef: Optional[Number] = None,
        alpha: float = 0.05,
        rand_seed: Optional[int] = None,
    ) -> None:
        if method not in (RepMethod.bootstrap, RepMethod.brr, RepMethod.jackknife):
            raise ValueError("Method must be 'bootstrap', 'brr', or 'jackknife'!")

        super().__init__(param=param, alpha=alpha, rand_seed=rand_seed)
        self.method = method
        self.conservative = False
        self.degree_of_freedom: Optional[int] = None
        self.nb_reps: Optional[int] = None
        self.rep_coefs: Optional[Union[np.ndarray, Number]] = None
        if method == RepMethod.brr and fay_coef is not None:
            self.fay_coef = fay_coef
        elif method == RepMethod.brr:
            self.fay_coef = 0
        if rep_weight_cls is not None:
            self.nb_reps = rep_weight_cls.nb_reps
            self.rep_coefs = rep_weight_cls.rep_coefs
            self.degree_of_freedom = rep_weight_cls.degree_of_freedom
            self.fay_coef = rep_weight_cls.fay_coef if self.method == RepMethod.brr else None

    def _rep_point(
        self, y: np.ndarray, rep_weights: np.ndarray, x: Optional[np.ndarray]
    ) -> np.ndarray:
        if self.param in (PopParam.prop, PopParam.mean):
            return np.asarray(
                np.sum(rep_weights * y[:, None], axis=0) / np.sum(rep_weights, axis=0)
            )
        elif self.param == PopParam.total:
            return np.asarray(np.sum(rep_weights * y[:, None], axis=0))
        elif self.param == PopParam.ratio and x is not None:
            return np.asarray(
                np.sum(rep_weights * y[:, None], axis=0) / np.sum(rep_weights * x[:, None], axis=0)
            )
        else:
            raise AssertionError("Parameter not valid!")

    def _bias(
        self,
        y: np.ndarray,
        samp_weight: np.ndarray,
        rep_weights: np.ndarray,
        x: Optional[np.ndarray],
    ) -> Number:
        estimate = self._get_point(y, samp_weight, x)
        if isinstance(estimate, (int, float)):
            rep_estimates = self._rep_point(y, rep_weights, x)
            return float(np.sum(np.mean(rep_estimates) - estimate))
        else:
            raise AssertionError

    def _rep_coefs(self, rep_coefs: Optional[Union[np.ndarray, Number]] = None) -> None:
        if rep_coefs is not None:
            if isinstance(rep_coefs, np.ndarray):
                self.rep_coefs = rep_coefs
            elif isinstance(rep_coefs, (int, float)):
                self.rep_coefs = rep_coefs
        elif self.nb_reps is not None and self.method == RepMethod.bootstrap:
            self.rep_coefs = (1 / self.nb_reps) * np.ones(self.nb_reps)
        elif self.nb_reps is not None and self.method == RepMethod.brr:
            self.rep_coefs = (1 / (self.nb_reps * pow(1 - self.fay_coef, 2))) * np.ones(
                self.nb_reps
            )
        elif self.nb_reps is not None and self.method == RepMethod.jackknife:
            self.rep_coefs = ((self.nb_reps - 1) / self.nb_reps) * np.ones(self.nb_reps)

    def _variance(
        self,
        y: np.ndarray,
        rep_weights: np.ndarray,
        rep_coefs: np.ndarray,
        x: np.ndarray,
        estimate: float,
        conservative: bool = False,
    ) -> Number:
        variance = 0.0
        rep_estimates = self._rep_point(y, rep_weights, x)
        if self.method == RepMethod.jackknife:  # page 155 (4.2.3 and 4.2.5) - Wolter(2003)
            jk_factor = np.array(1 / (1 - rep_coefs))
            pseudo_estimates = jk_factor * estimate - (jk_factor - 1) * rep_estimates
            if conservative:
                variance = float(
                    np.sum(rep_coefs * pow((pseudo_estimates - estimate) / (jk_factor - 1), 2))
                )
            elif not conservative:
                variance = float(
                    np.sum(
                        rep_coefs
                        * pow(
                            (pseudo_estimates - np.mean(pseudo_estimates)) / (jk_factor - 1),
                            2,
                        )
                    )
                )
        else:
            if conservative:
                variance = float(np.sum(rep_coefs * pow(rep_estimates - estimate, 2)))
            elif not conservative:
                variance = float(
                    np.sum(rep_coefs * pow(rep_estimates - np.mean(rep_estimates), 2))
                )
        return variance

    def _get_bias(
        self,
        y: np.ndarray,
        samp_weight: np.ndarray,
        rep_weights: np.ndarray,
        x: Optional[np.ndarray] = None,
        domain: Optional[np.ndarray] = None,
        remove_nan: bool = False,
    ) -> Any:
        if remove_nan:
            if self.param == PopParam.ratio and x is not None:
                excluded_units = np.isnan(y) | np.isnan(x)
            else:
                excluded_units = np.isnan(y)
            y, samp_weight, x, stratum, domain, _, _ = remove_nans(
                excluded_units, y, samp_weight, x, None, domain, None, None
            )
            rep_weights = rep_weights[~excluded_units, :]

        if self.param == PopParam.prop:
            y_dummies = pd.get_dummies(y)
            categories = y_dummies.columns
            y_dummies = y_dummies.values

        bias: Any
        if domain is None:
            if self.param == PopParam.prop and categories is not None:
                cat_dict = dict()
                for k in range(categories.size):
                    cat_dict_k = dict(
                        {
                            categories[k]: self._bias(
                                y_dummies[:, k], samp_weight, rep_weights, None
                            )
                        }
                    )
                    cat_dict.update(cat_dict_k)
                bias = cat_dict
            else:
                bias = self._bias(y, samp_weight, rep_weights, x)
        else:
            for d in np.unique(domain):
                samp_weight_d = samp_weight * (domain == d)
                rep_weights_d = rep_weights * (domain == d)[:, None]
                if self.param == PopParam.ratio:
                    x_d = x * (domain == d)
                else:
                    x_d = x
                if self.param == PopParam.prop:
                    y_dummies_d = y_dummies * (domain == d)[:, None]
                    cat_dict = dict()
                    for k in range(categories.size):
                        cat_dict_d_k = dict(
                            {
                                categories[k]: self._bias(
                                    y_dummies_d[:, k],
                                    samp_weight_d,
                                    rep_weights_d,
                                    None,
                                )
                            }
                        )
                        cat_dict.update(cat_dict_d_k)
                    bias[d] = cat_dict
                else:
                    y_d = y * (domain == d)
                    bias[d] = self._bias(y_d, samp_weight_d, rep_weights, x_d)

        return bias

    def _get_variance(
        self,
        y: np.ndarray,
        samp_weight: np.ndarray,
        rep_weights: np.ndarray,
        rep_coefs: np.ndarray,
        x: Optional[np.ndarray] = None,
        domain: Optional[np.ndarray] = None,
        conservative: bool = False,
        remove_nan: bool = False,
    ) -> Any:
        if self.param == PopParam.prop:
            y_dummies = pd.get_dummies(y)
            categories = y_dummies.columns
            y_dummies = y_dummies.values

        if domain.shape in ((), (0,)):
            if self.param == PopParam.prop:
                estimate_k: dict
                cat_dict = dict()
                for k in range(categories.size):
                    estimate_k = self._get_point(
                        y=y_dummies[:, k],
                        samp_weight=samp_weight,
                        x=x,
                        domain=np.array(None),
                    )
                    cat_dict_k = dict(
                        {
                            categories[k]: self._variance(
                                y_dummies[:, k],
                                rep_weights,
                                rep_coefs,
                                x,
                                estimate_k[1],
                                conservative,
                            )
                        }
                    )
                    cat_dict.update(cat_dict_k)
                return cat_dict
            else:
                estimate = self._get_point(y=y, samp_weight=samp_weight, x=x, domain=domain)
                return self._variance(y, rep_weights, rep_coefs, x, estimate, conservative)
        else:
            variance_else1 = {}
            variance_else2 = {}
            for d in np.unique(domain):
                samp_weight_d = samp_weight * (domain == d)
                rep_weights_d = rep_weights * (domain == d)[:, None]
                if self.param == PopParam.ratio:
                    x_d = x * (domain == d)
                else:
                    x_d = x
                if self.param == PopParam.prop and categories is not None:
                    y_dummies_d = np.asarray(y_dummies * (domain == d)[:, None])
                    cat_dict = dict()
                    for k in range(categories.size):
                        estimate_d_k = self._get_point(
                            y=y_dummies_d[:, k],
                            samp_weight=samp_weight_d,
                            x=x_d,
                            domain=np.array(None),
                        )[1]
                        cat_dict_d_k = dict(
                            {
                                categories[k]: self._variance(
                                    y=y_dummies_d[:, k],
                                    rep_weights=rep_weights_d,
                                    rep_coefs=rep_coefs,
                                    x=x_d,
                                    estimate=estimate_d_k,
                                    conservative=conservative,
                                )
                            }
                        )
                        cat_dict.update(cat_dict_d_k)
                    variance_else1[d] = cat_dict
                else:
                    y_d = y * (domain == d)
                    estimate_d = self._get_point(
                        y=y_d, samp_weight=samp_weight_d, x=x_d, domain=np.array(None)
                    )
                    variance_else2[d] = self._variance(
                        y=y_d,
                        rep_weights=rep_weights_d,
                        rep_coefs=rep_coefs,
                        x=x_d,
                        estimate=estimate_d,
                        conservative=conservative,
                    )
            if self.param == PopParam.prop:
                return variance_else1
            else:
                return variance_else2

    def _get_confint(
        self,
        param: str,
        estimate: Any,
        variance: Any,
        quantile: float,
    ) -> Tuple[Any, Any]:
        lower_ci = {}
        upper_ci = {}
        if self.domains.shape in ((), (0,)):
            if param == PopParam.prop:
                for level in variance:
                    point_est = estimate[level]
                    std_est = pow(variance[level], 0.5)
                    location_ci = math.log(point_est / (1 - point_est))
                    scale_ci = std_est / (point_est * (1 - point_est))
                    ll = location_ci - quantile * scale_ci
                    lower_ci[level] = math.exp(ll) / (1 + math.exp(ll))
                    uu = location_ci + quantile * scale_ci
                    upper_ci[level] = math.exp(uu) / (1 + math.exp(uu))
                return lower_ci, upper_ci
            else:
                return estimate - quantile * pow(variance, 0.5), estimate + quantile * pow(
                    variance, 0.5
                )

        else:
            lower_ci_else1 = {}
            upper_ci_else1 = {}
            lower_ci_else2 = {}
            upper_ci_else2 = {}
            for key in variance:
                if param == PopParam.prop:
                    lower_ci_k = {}
                    upper_ci_k = {}
                    for level in variance[key]:
                        point_est = estimate[key][level]
                        std_est = pow(variance[key][level], 0.5)
                        location_ci = math.log(point_est / (1 - point_est))
                        scale_ci = std_est / (point_est * (1 - point_est))
                        ll = location_ci - quantile * scale_ci
                        lower_ci_k[level] = math.exp(ll) / (1 + math.exp(ll))
                        uu = location_ci + quantile * scale_ci
                        upper_ci_k[level] = math.exp(uu) / (1 + math.exp(uu))
                    lower_ci_else1[key] = lower_ci_k
                    upper_ci_else1[key] = upper_ci_k
                else:
                    lower_ci_else2[key] = estimate[key] - quantile * pow(variance[key], 0.5)
                    upper_ci_else2[key] = estimate[key] + quantile * pow(variance[key], 0.5)

            if self.param == PopParam.prop:
                return lower_ci_else1, upper_ci_else1
            else:
                return lower_ci_else2, upper_ci_else2

    def _get_coefvar(
        self,
        param: str,
        estimate: Any,  # Any := Union[dict[StringNumber, DictStrNum], DictStrNum, Number]
        variance: Any,
    ) -> Any:  # Any := Union[dict[StringNumber, DictStrNum], DictStrNum, Number]
        if self.domains.shape in ((), (0,)):
            if param == PopParam.prop:
                coef_var = {}
                for level in variance:
                    coef_var[level] = pow(variance[level], 0.5) / estimate[level]
            else:
                coef_var = pow(variance, 0.5) / estimate
        else:
            coef_var = {}
            for key in variance:
                if param == PopParam.prop:
                    coef_var_k = {}
                    for level in variance[key]:
                        coef_var_k[level] = pow(variance[key][level], 0.5) / estimate[key][level]
                    coef_var[key] = coef_var_k
                else:
                    coef_var[key] = pow(variance[key], 0.5) / estimate[key]
        return coef_var

    def estimate(
        self: TypeRepEst,
        y: Array,
        samp_weight: Array,
        rep_weights: Union[np.ndarray, pd.DataFrame],
        x: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        rep_coefs: Optional[Union[float, np.ndarray]] = None,
        domain: Optional[np.ndarray] = None,
        conservative: bool = False,
        deff: bool = False,  # Todo
        remove_nan: bool = False,
    ) -> TypeRepEst:
        """[summary]

        Args:
            self (TypeRepEst): [description]
            y (Array): [description]
            samp_weight (Array): [description]
            rep_weights (Union[np.ndarray, pd.DataFrame]): [description]
            x (Union[np.ndarray, pd.DataFrame, None], optional): [description]. Defaults to None.
            rep_coefs (Union[float, np.ndarray, None], optional): [description]. Defaults to None.
            domain (Optional[np.ndarray], optional): [description]. Defaults to None.
            conservative (bool, optional): [description]. Defaults to False.
            deff (bool, optional): [description]. Defaults to False.

        Raises:
            AssertionError: [description]

        Returns:
            TypeRepEst: [description]
        """

        if self.param == PopParam.ratio and x is None:
            raise AssertionError("x must be provided for ratio estimation.")

        _y = numpy_array(y)
        _x = numpy_array(x)
        _samp_weight = numpy_array(samp_weight)
        _rep_weights = numpy_array(rep_weights)
        _rep_coefs = numpy_array(rep_coefs)
        _domain = numpy_array(domain)

        if remove_nan:
            to_keep = remove_nans(_y.shape[0], _y)

            _y = _y[to_keep]
            _x = _x[to_keep] if _x.shape not in ((), (0,)) else _x
            _samp_weight = _samp_weight[to_keep]
            _rep_coefs = _rep_coefs[to_keep] if _rep_coefs.shape not in ((), (0,)) else _rep_coefs
            _rep_weights = (
                _rep_weights[to_keep] if _rep_weights.shape not in ((), (0,)) else _rep_weights
            )
            _domain = _domain[to_keep] if _domain.shape not in ((), (0,)) else _domain

        self.conservative = conservative

        if self.nb_reps is None:
            self.nb_reps = rep_weights.shape[1]

        self._rep_coefs(rep_coefs)

        self.domains = np.unique(_domain) if _domain.shape not in ((), (0,)) else _domain

        self.point_est = self._get_point(y=_y, samp_weight=_samp_weight, x=_x, domain=_domain)
        self.variance = self._get_variance(
            y=_y,
            samp_weight=_samp_weight,
            rep_weights=_rep_weights,
            rep_coefs=np.array(self.rep_coefs),
            x=_x,
            domain=_domain,
            conservative=conservative,
            remove_nan=remove_nan,
        )

        if self.method == RepMethod.brr and self.degree_of_freedom is None:
            self.degree_of_freedom = int(self.nb_reps / 2)
        elif self.degree_of_freedom is None:
            self.degree_of_freedom = int(self.nb_reps) - 1

        t_quantile = student.ppf(1 - self.alpha / 2, df=self.degree_of_freedom)

        self.lower_ci, self.upper_ci = self._get_confint(
            self.param, self.point_est, self.variance, t_quantile
        )
        self.coef_var = self._get_coefvar(self.param, self.point_est, self.variance)

        if self.domains.shape in ((), (0,)):
            if self.param == PopParam.prop:
                for level in self.variance:
                    self.stderror[level] = pow(self.variance[level], 0.5)
            elif isinstance(self.variance, (int, float)):
                self.stderror = pow(self.variance, 0.5)
        else:
            for key in self.variance:
                if self.param == PopParam.prop:
                    stderror = {}
                    for level in self.variance[key]:
                        stderror[level] = pow(self.variance[key][level], 0.5)
                    self.stderror[key] = stderror
                else:
                    self.stderror[key] = pow(self.variance[key], 0.5)

        return self

    def estimate_calibrated(
        self: TypeRepEst,
        y: Array,
        calibrated: Any,
        *,
        x: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        domain: Optional[np.ndarray] = None,
        conservative: bool = False,
    ) -> TypeRepEst:
        """Estimate total/mean/ratio from a :class:`CalibratedReplicates` batch.

        The point estimate uses the calibrated full-sample weights; the
        variance uses every calibrated replicate column. The design metadata
        carried by the batch (replicate coefficients, Fay correction,
        conservative convention, degrees of freedom) drive the variance
        formulas and are validated against this estimator's construction
        instead of being recomputed from the surviving non-zero rows.

        Args:
            y: length-n one-dimensional finite real outcome.
            calibrated: a CalibratedReplicates produced by
                :meth:`ReplicateWeight.replicate` with ``calibration`` set.
            x: length-n auxiliary for ratio estimation (required then).
            domain: length-n domain labels without missing values; the same
                calibrated weights are masked per domain - domains are never
                re-calibrated separately.
            conservative: must match the batch's recorded convention.

        Raises:
            ValueError: method mismatch, wrong lengths, bad options or
                metadata inconsistencies.
            ZeroDivisionError: a mean/ratio denominator vanishes for the full
                sample, any replicate or any reported domain. On any failure
                the previous successful results on this estimator are left
                untouched.

        Returns:
            The estimator instance, with the same result attributes and
            :meth:`to_dataframe` support as :meth:`estimate`.
        """

        from samplics.weighting.calibrated import CalibratedReplicates

        if not isinstance(calibrated, CalibratedReplicates):
            raise ValueError(
                "calibrated must be a CalibratedReplicates instance; use estimate() "
                "for raw replicate arrays."
            )
        if not isinstance(conservative, bool):
            raise ValueError("conservative must be a boolean.")
        if conservative != bool(getattr(calibrated, "conservative", False)):
            raise ValueError(
                "conservative must match the convention recorded with the calibrated "
                "batch."
            )
        if not _method_equal(self.method, calibrated.method):
            raise ValueError(
                f"Estimator method {self.method!r} does not match the batch method "
                f"{calibrated.method!r}."
            )
        if self.param == PopParam.ratio and x is None:
            raise ValueError("x must be provided for ratio estimation.")

        full_weights = np.asarray(calibrated.full_weights, dtype=np.float64)
        rep_weights = np.asarray(calibrated.rep_weights, dtype=np.float64)
        n = full_weights.shape[0]
        if rep_weights.ndim != 2 or rep_weights.shape[0] != n:
            raise ValueError("calibrated.rep_weights must be an n-by-R matrix.")
        nb_reps = rep_weights.shape[1]

        _y = _finite_vector("y", y, n)
        _x = numpy_array(x)
        if _x.shape not in ((), (0,)):
            _x = _finite_vector("x", _x, n)
        _domain = numpy_array(domain)
        if _domain.shape not in ((), (0,)):
            if _domain.shape[0] != n:
                raise ValueError("domain must have one label per sample unit.")
            if np.issubdtype(_domain.dtype, np.floating) and np.any(np.isnan(_domain)):
                raise ValueError("domain labels must not contain missing values.")
            if _domain.dtype.kind == "O":
                if any(label is None or (isinstance(label, float) and np.isnan(label))
                       for label in _domain):
                    raise ValueError("domain labels must not contain missing values.")

        # Design metadata checks against the estimator's construction.
        batch_coefs = np.asarray(calibrated.rep_coefs, dtype=np.float64).reshape(-1)
        if batch_coefs.shape[0] != nb_reps:
            raise ValueError("The batch rep_coefs must have one entry per replicate.")
        if self.method == RepMethod.brr:
            fay_batch = calibrated.fay_coef
            if fay_batch is None:
                raise ValueError("BRR batch must carry a Fay coefficient (0 for plain BRR).")
            expected_coef = 1.0 / (nb_reps * (1.0 - fay_batch) ** 2)
            if not np.allclose(batch_coefs, expected_coef, rtol=1e-9, atol=1e-12):
                raise ValueError(
                    "BRR replicate coefficients are inconsistent with the Fay coefficient."
                )
            if int(calibrated.degree_of_freedom) != int(calibrated.nb_strata):
                raise ValueError(
                    "BRR degrees of freedom must equal the number of strata."
                )
            if self.fay_coef is not None and not np.isclose(
                float(self.fay_coef), float(fay_batch), rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    "Estimator Fay coefficient does not match the calibrated batch."
                )
        elif self.method == RepMethod.jackknife:
            if calibrated.fay_coef is not None:
                raise ValueError("Jackknife batches must have fay_coef set to None.")
            if not np.all(batch_coefs > 0.0) or not np.all(batch_coefs < 1.0):
                raise ValueError("Jackknife replicate coefficients must lie in (0, 1).")
            if int(calibrated.degree_of_freedom) <= 0:
                raise ValueError("Jackknife degrees of freedom must be positive.")
            if int(calibrated.degree_of_freedom) > nb_reps:
                raise ValueError(
                    "Jackknife degrees of freedom cannot exceed the number of replicates."
                )

        # Zero-denominator rejection before any state is mutated.
        _assert_positive_denominators(
            full_weights, n, self.param, _x, _domain, "the full-sample weights"
        )
        _assert_positive_denominators(
            rep_weights, n, self.param, _x, _domain, "the calibrated replicate weights"
        )

        # From here on the call is valid: mirror the legacy estimate() flow so
        # point estimates, variance formulas, CIs and to_dataframe are reused.
        previous = {
            "point_est": self.point_est,
            "variance": self.variance,
            "stderror": self.stderror,
            "lower_ci": self.lower_ci,
            "upper_ci": self.upper_ci,
            "coef_var": self.coef_var,
            "domains": self.domains,
            "degree_of_freedom": self.degree_of_freedom,
            "nb_reps": self.nb_reps,
            "rep_coefs": self.rep_coefs,
            "conservative": self.conservative,
        }
        try:
            self.conservative = conservative
            self.nb_reps = nb_reps
            self.rep_coefs = batch_coefs
            self.degree_of_freedom = int(calibrated.degree_of_freedom)
            self.domains = (
                np.unique(_domain) if _domain.shape not in ((), (0,)) else _domain
            )

            self.point_est = self._get_point(
                y=_y, samp_weight=full_weights, x=_x, domain=_domain
            )
            self.variance = self._get_variance(
                y=_y,
                samp_weight=full_weights,
                rep_weights=rep_weights,
                rep_coefs=batch_coefs,
                x=_x,
                domain=_domain,
                conservative=conservative,
            )

            t_quantile = student.ppf(1 - self.alpha / 2, df=self.degree_of_freedom)
            self.lower_ci, self.upper_ci = self._get_confint(
                self.param, self.point_est, self.variance, t_quantile
            )
            self.coef_var = self._get_coefvar(
                self.param, self.point_est, self.variance
            )

            if self.domains.shape in ((), (0,)):
                if self.param == PopParam.prop:
                    for level in self.variance:
                        self.stderror[level] = pow(self.variance[level], 0.5)
                elif isinstance(self.variance, (int, float)):
                    self.stderror = pow(self.variance, 0.5)
            else:
                for key in self.variance:
                    if self.param == PopParam.prop:
                        stderror = {}
                        for level in self.variance[key]:
                            stderror[level] = pow(self.variance[key][level], 0.5)
                        self.stderror[key] = stderror
                    else:
                        self.stderror[key] = pow(self.variance[key], 0.5)
        except Exception:
            # Keep the last successful result intact after a failed call.
            for attr, value in previous.items():
                setattr(self, attr, value)
            raise

        return self

    def to_dataframe(self, col_names: Optional[list] = None) -> pd.DataFrame:
        """Return the latest estimates as a dataframe.

        Mirrors :meth:`TaylorEstimator.to_dataframe` using the same column
        layout and :func:`~samplics.utils.formats.dict_to_dataframe` helper, so
        calibrated and uncalibrated replicate runs produce identical output
        shapes.
        """

        from samplics.utils.formats import dict_to_dataframe

        if self.point_est in (None, {}):
            raise AssertionError("No estimates yet. Must first run an estimate method.")

        if col_names is None:
            col_names = [
                "_param",
                "_domain",
                "_estimate",
                "_stderror",
                "_lci",
                "_uci",
                "_cv",
            ]
            if self.domains is None or self.domains.shape in ((), (0,)):
                col_names.pop(1)
        elif len(col_names) != (6 if self.domains is None or self.domains.shape in ((), (0,)) else 7):
            raise AssertionError("col_names has the wrong number of columns for this run.")

        est_df = dict_to_dataframe(
            col_names,
            self.point_est,
            self.stderror,
            self.lower_ci,
            self.upper_ci,
            self.coef_var,
        )
        est_df.iloc[:, 0] = self.param.value if hasattr(self.param, "value") else str(self.param)
        return est_df
