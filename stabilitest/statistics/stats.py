import numpy as np
import significantdigits
from significantdigits import Error, Method


def compute_stats(args, sample_module, collector):
    reference_sample = sample_module.get_reference_sample(args)
    reference_sample.load()
    sample_module.preprocess(reference_sample, ...)
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

    filename = args.output

    def save(x, name):
        print(f"Save NumPy {name}")
        _filename = f"{filename}_{name}"
        np.save(filename, x)
        reference_sample.dump(x, _filename)

    save(_max, "max")
    save(_min, "min")
    save(_median, "median")
    save(_mean, "mean")
    save(_std, "std")
    save(_sig, "sig")
