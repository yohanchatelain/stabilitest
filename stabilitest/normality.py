import numpy as np
from joblib import Parallel, delayed
from scipy import stats
import itertools
import os

import stabilitest.statistics.multiple_testing as mt
from statsmodels.stats.multitest import multipletests

from stabilitest.model import load_configuration


def perform_test_on_voxel(voxel_values):
    _, p = stats.normaltest(voxel_values)
    return p

def build_hyperparameters(config):
    hyper_parameters = config.get("hyperparameters", None)
    if hyper_parameters is not None:
        return list(hyper_parameters.keys()), itertools.product(
            *hyper_parameters.values()
        )
    else:
        return [], []




def _run_normality_test(config, sample_module, collector, hyperparameters):
    print("\nRun normality test")
    print(f"Hyperparameters: {hyperparameters}")

    confidences = config["confidence"]
    subject = config["reference"]["subject"]
    perturbation = config["reference"]["perturbation"]
    fhwm = hyperparameters["smooth-kernel"]
    
    filename = f"{subject}_{perturbation}.pkl_data_FWHM-{fhwm}mm.npy"
    print(filename)
    if os.path.exists(filename):
        print("Loading data from cached file")
        data = np.load(filename)
        sample_size = data.shape[0]
        print("Data loaded")
        info = {
            "reference_version": config["reference"]["version"],
            "reference_architecture": config["reference"]["architecture"],
            "reference_perturbation": config["reference"]["perturbation"],
            "reference_prefix": config["reference"]["prefix"],
            "reference_dataset": config["reference"]["dataset"],
            "reference_subject": config["reference"]["subject"],
            "reference_template": config["reference"]["template"],
            "reference_sample_size": sample_size,
            "reference_fwhm": hyperparameters['smooth-kernel'],
            "reference_mask": hyperparameters['mask-combination'],
        }
    else:
        sample = sample_module.get_reference_sample(config, hyperparameters)
        sample.load()
        sample_module.preprocess(sample)
        data = sample.data
        sample_size = sample.get_observation_shape()
        info = sample.get_info()

    data_size = np.prod(data.shape[1:])
        
    # Flatten the 3D voxels in each image into 1D,
    # so that voxels at the same location in different images line up
    flattened_data = data.reshape(sample_size, -1)

    p_values = Parallel(n_jobs=-1, verbose=10, batch_size=1024)(
        delayed(perform_test_on_voxel)(flattened_data[:, i])
        for i in range(flattened_data.shape[1])
    )

    # Reshape the p-values back into the shape of a single 3D voxel grid
    # p_values = np.array(p_values).reshape(subsample.shape[1:])
    p_values = np.array(p_values)
    p_values.sort()

    methods = [mt.pce, mt.fwe_bonferroni, mt.fdr_BY]

    for confidence in confidences:
        alpha = 1 - confidence

        for method in methods:
            nb_reject, size, _ = method("", alpha, p_values)
            ratio = nb_reject / size

            print(f"Method {method.__name__}")
            print(f"Card(Data not normal)  = {nb_reject}")
            print(f"Card(Data)             = {data_size}")
            print(f"non-normal data ratio  = {ratio:.2e} [{ratio*100:f}%]")

            collector.append(**info)



def run_normality_test(args, sample_module, collector):
    """
    Run the non-normality test for the given sample
    """

    config = load_configuration(args.configuration_file)
    hyperparameters_names, hyperparameters_values = build_hyperparameters(config)

    results =  Parallel(n_jobs=-1, verbose=10)(
        delayed(_run_normality_test)(
            config,
            sample_module,
            collector,
            dict(zip(hyperparameters_names, hpv)),
        )
        for hpv in hyperparameters_values
    )

    fvr_map = dict(zip(hyperparameters_values, results))
    return hyperparameters_names, fvr_map
