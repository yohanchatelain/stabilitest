import numpy as np
import significantdigits
from significantdigits import Error, Method
import itertools
from joblib import Parallel, delayed

import os
from stabilitest.model import load_configuration

def build_hyperparameters(config):
    hyper_parameters = config.get("hyperparameters", None)
    if hyper_parameters is not None:
        return list(hyper_parameters.keys()), itertools.product(
            *hyper_parameters.values()
        )
    else:
        return [], []

def run_test_with_hyperparameters(config, sample_module, collector, hpn, hpv):
    hyperparameters = dict(zip(hpn, hpv))
    return _compute_stats(config, sample_module, collector, hyperparameters)


def run_test_over_hyperparameters(args, sample_module, collector, run_test):
    config = load_configuration(args.configuration_file)
    hyperparameters_names, hyperparameters_values = build_hyperparameters(config)
    fvr_map = {}
    for hpv in hyperparameters_values:
        fvr_map[hpv] = run_test_with_hyperparameters(config, sample_module, collector, hyperparameters_names, hpv)
    return hyperparameters_names, fvr_map


def compute_stats(args, sample_module, collector):
    config = load_configuration(args.configuration_file)
    hyperparameters_names, hyperparameters_values = build_hyperparameters(config)
    if config.get("cpus", 1) > 1:
        results = Parallel(n_jobs=-1)(delayed(run_test_with_hyperparameters)(config, sample_module, collector, hyperparameters_names, hpv) for hpv in hyperparameters_values)
        return results
    else:
        return run_test_over_hyperparameters(args, sample_module, collector, _compute_stats)


def _compute_stats(config, sample_module, collector, hyperparameters):
    print("\nRun stats")
    print(f"Hyperparameters: {hyperparameters}")
    reference_sample = sample_module.get_reference_sample(config, hyperparameters)
    reference_sample.load()
    sample_module.preprocess(reference_sample)
    data = reference_sample.get_subsample()

    _max = np.max(data, axis=0)
    _min = np.min(data, axis=0)
    _median = np.median(data, axis=0)
    _mean = np.mean(data, axis=0)
    _std = np.std(data, axis=0)

    _sig = significantdigits.significant_digits(
        array=data,
        reference=_mean,
        basis=2,
        axis=0,
        error=Error.Relative,
        method=Method.CNH,
    )

    info = {}
    for name, stat in {
        "max": _max,
        "min": _min,
        "median": _median,
        "mean": _mean,
        "std": _std,
    }.items():
        info[f"{name}_max"] = stat.max()
        info[f"{name}_min"] = stat.min()
        info[f"{name}_median"] = np.median(stat)
        info[f"{name}_mean"] = stat.mean()
        info[f"{name}_std"] = stat.std()

    collector.append(**info)

    filename = config['output']
    fwhm = hyperparameters.get("smooth-kernel", None)


    def save(x, name):
        print(f"Save NumPy {name}")
        np_filename = f"{filename}_{name}_FWHM-{fwhm}mm"
        if not os.path.exists(np_filename):        
            np.save(f"{filename}_{name}_FWHM-{fwhm}mm", x)
        nii_filename = f"{filename}_{name}_FWHM-{fwhm}mm.nii.gz"
        if not os.path.exists(nii_filename):
            reference_sample.dump(x, nii_filename)

    save(data, "data")
    save(_max, "max")
    save(_min, "min")
    save(_median, "median")
    save(_mean, "mean")
    save(_std, "std")
    save(_sig, "sig")
