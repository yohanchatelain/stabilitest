import numpy as np
from joblib import Parallel, delayed
from scipy import stats

import stabilitest.statistics.multiple_testing as mt


def perform_test_on_voxel(voxel_values):
    _, p = stats.normaltest(voxel_values)
    return p


def test_normality(args, sample, collector):
    confidences = args.confidence
    subsample = sample.get_subsample()
    data_size = subsample.shape[1:]

    # Flatten the 3D voxels in each image into 1D,
    # so that voxels at the same location in different images line up
    flattened_data = subsample.reshape(subsample.shape[0], -1)

    p_values = Parallel(n_jobs=-1)(
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

            info = sample.get_info()
            collector.append(**info)


def run_normality_test(args, sample_module, collector):
    """
    Run the non-normality test for the given sample
    """

    sample = sample_module.get_reference_sample(args)
    sample.load()
    sample_module.preprocess(sample)
    test_normality(args, sample, collector)
