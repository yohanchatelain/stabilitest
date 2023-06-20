import multiprocessing
import warnings
from itertools import chain
from typing import Dict

import numpy as np
import scipy.stats as sps
import significantdigits
import sklearn.mixture
import tqdm
from icecream import ic


def sequential_fit_normal(x, fit):
    _iterable = tqdm.tqdm(iterable=range(x.shape[-1]), total=x.shape[-1])
    _parameters = np.fromiter(
        chain.from_iterable(fit(x[..., i]) for i in _iterable), dtype=np.float64
    )
    parameters = dict(
        a=_parameters[..., 0], loc=_parameters[..., 1], scale=_parameters[..., 2]
    )
    return parameters


def parallel_fit_normal(x, fit):
    _iterable = tqdm.tqdm(np.swapaxes(x, 0, 1))

    with multiprocessing.Pool() as pool:
        _parameters = np.fromiter(
            chain.from_iterable(pool.map(fit, _iterable, chunksize=500)),
            dtype=np.float64,
        )

        parameters = dict(
            beta=_parameters[..., 0], loc=_parameters[..., 1], scale=_parameters[..., 2]
        )
        return parameters

    return None


class Distribution:
    def __init__(self, name: str, parameters: Dict = {}):
        self.name = name
        self.parameters = parameters

    def get_name(self):
        return self.name

    def get_loc(self):
        return self.parameters["loc"]

    def get_scale(self):
        return self.parameters["scale"]

    def p_value(self, x):
        raise NotImplementedError


class Student(Distribution):
    def __init__(self, parameters: Dict = {}):
        super().__init__("student", parameters)

    def t_score(self, x):
        return np.abs(x - self.get_loc()) / self.get_scale()

    def p_value(self, x, df=None):
        df = df if df else len(x)
        t = self.t_score(x)
        return sps.t.sf(np.abs(t), df=df - 1) * 2

    def fit(self, x):
        self.parameters["loc"] = np.mean(x, axis=0)
        self.parameters["scale"] = sps.sem(x, axis=0, ddof=1)


class Gaussian(Distribution):
    def __init__(self, parameters: Dict = {}):
        super().__init__("norm", parameters)

    def z_score(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return (x - self.get_loc()) / self.get_scale()

    def p_value(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean = self.get_loc()
            std = self.get_scale()
            low = sps.norm.cdf(x, loc=mean, scale=std)
            up = sps.norm.sf(x, loc=mean, scale=std)
            return 2 * np.min((low, up), axis=0)

    def fit(self, x):
        self.parameters["loc"] = np.mean(x, axis=0)
        self.parameters["scale"] = np.std(x, axis=0, dtype=np.float64)


class GaussianSkew(Distribution):
    def __init__(self, parameters: Dict = {}):
        super().__init__("norm-skew", parameters)

    def z_score(self, x):
        return (x - self.get_loc()) / self.get_scale()

    def p_value(self, x):
        mean = self.get_loc()
        std = self.get_scale()
        a = self.parameters["a"]
        low = sps.skewnorm.cdf(x, a=a, loc=mean, scale=std)
        up = sps.skewnorm.sf(x, a=a, loc=mean, scale=std)
        return 2 * np.min((low, up), axis=0)

    def fit(self, x):
        self.parameters = chain.from_iterable(sps.skewnorm.fit(x))


class GaussianGeneral(Distribution):
    def __init__(self, parameters: Dict = {}):
        super().__init__("norm-general", parameters)

    def z_score(self, x):
        return (x - self.get_loc()) / self.get_scale()

    def p_value(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean = self.get_loc()
            std = self.get_scale()
            beta = self.parameters["beta"]
            low = sps.gennorm.cdf(x, beta=beta, loc=mean, scale=std)
            up = sps.gennorm.sf(x, beta=beta, loc=mean, scale=std)
            return 2 * np.min((low, up), axis=0)

    def fit(self, x):
        self.parameters = chain.from_iterable(sps.gennorm.fit(x))


class GaussianMixture(Distribution):
    def __init__(self, parameters: Dict = {}):
        super().__init__("gaussian-mixture", parameters)

    def cdf(self, x):
        mean = self.get_loc()
        std = self.get_scale()
        weights = self.parameters["weights"]
        return sum(
            lambda w, m, s: w * sps.norm.cdf(x, loc=m, scale=s),
            zip(weights, mean, std),
        )

    def p_value(self, x):
        cdf = self.cdf(x)
        return 2 * np.min((cdf, 1 - cdf), axis=0)

    def fit(self, x):
        reg_covar = 10**-6
        while reg_covar < 1:
            gmm = sklearn.mixture.GaussianMixture(
                n_components=2, covariance_type="diag", verbose=2, reg_covar=reg_covar
            )
            try:
                _fitted_model = gmm.fit(x)
                self.parameters["loc"] = _fitted_model.means_
                self.parameters["scale"] = _fitted_model.covariances_
                self.parameters["weights"] = _fitted_model.weights_
            except Exception:
                reg_covar *= 10
        if reg_covar == 1:
            raise Exception("GMMFitNonConvergence")


class SignificantDigits(Distribution):
    def __init__(self, parameters: Dict = {}):
        super().__init__("significant-digits", parameters)

    def fit(self, x):
        self.parameters["loc"] = significantdigits.significant_digits(
            np.mean(x, axis=0),
            reference=x,
            axis=0,
            error=significantdigits.Error.Relative,
            method=significantdigits.Method.CNH,
        )

    def p_value(self, x):
        test = significantdigits.significant_digits(
            self.get_loc(),
            reference=x,
            axis=0,
            error=significantdigits.Error.Relative,
            method=significantdigits.Method.CNH,
        )
        return test < self.parameters["loc"]


def get_distribution(args):
    if args.distribution == "normal":
        return Gaussian()
    if args.distribution == "student":
        return Student()
    if args.distribution == "normal-skew":
        return GaussianSkew()
    if args.distribution == "normal-general":
        return GaussianGeneral()
    if args.distribution == "significant-digit":
        return SignificantDigits()
