import glob
import logging
import os

import joblib
import nibabel
import nilearn
import nilearn.image
import nilearn.masking
import numpy as np
from nilearn.masking import apply_mask, intersect_masks

import stabilitest.mri_loader.constants as mri_constants
import stabilitest.pprinter as mrip


def load_derivative(path):
    realpath = os.path.realpath(path)
    return nibabel.load(realpath)


def mask_image(image, mask):
    masked_image = np.ma.where(mask, image.get_fdata(), 0)
    return nibabel.Nifti1Image(masked_image, image.affine)


def dump_image(filename, x, affine):
    image = nibabel.Nifti1Image(x, affine)
    nibabel.save(image, filename)


def dump_stat(target, stat_image, supermask, alpha, stat_name=""):
    """
    Generic dumping function
    """
    filename = target.get_filename().replace(
        ".nii.gz", f"_{stat_name}_{alpha:.3f}_.nii.gz"
    )
    mrip.print_debug(f"Dump {stat_name} {filename}")
    masked_image = np.where(supermask, stat_image, 0)
    image = nibabel.Nifti1Image(masked_image, target.affine)
    dump_image(filename, image.get_fdata(), image.affine)


def dump_failing_voxels(target, mask, alpha, p_values, fwh):
    confidence = 1 - alpha
    new_filename = f"_{confidence}_fwh_{fwh}.nii.gz"
    filename = target.get_filename().replace(".nii.gz", new_filename)
    mask = mask.get_fdata().astype("bool")
    fp_masked = np.logical_and(p_values <= alpha, mask)
    image = nibabel.Nifti1Image(fp_masked, target.affine)
    dump_image(filename, image.get_fdata(), image.affine)


def dump_mean(target, mean, supermask, alpha):
    """
    Dump mean for masked target
    """
    dump_stat(target, mean, supermask, alpha, stat_name="mean")


def dump_std(target, std, supermask, alpha):
    """
    Dump std for masked target
    """
    dump_stat(target, std, supermask, alpha, stat_name="std")


def dump_p_values(target, p_value, supermask, alpha):
    """
    Dump p-values (with alpha threshold) for masked target
    """
    dump_stat(target, p_value, supermask, alpha, stat_name="p_value")


def load_derivatives(root_paths, derivative_path):
    """
    Load derivatives from paths
    """
    derivatives = []
    for path in root_paths:
        dpath = glob.glob(os.path.join(path, derivative_path))
        if len(dpath) > 0:
            if len(dpath) > 1:
                logging.warning(f"More than one file found for {dpath}")
            image = load_derivative(dpath[0])
        else:
            continue
        derivatives.append(image)

    return np.array(derivatives)


def combine_mask(masks_list, operator):
    """
    Combine mask depending on the operator
    """
    if masks_list is None:
        raise ValueError("No mask list provided to compute mask combination")
    if operator == "union":
        threshold = 0
    elif operator == "intersection":
        threshold = 1
    else:
        threshold = 0.5
    return intersect_masks(masks_list, threshold=threshold)


def resample_image(source, target):
    return np.array([nilearn.image.resample_to_img(source, target)])


def resample_images(sources, target):
    resampled_images = []
    for source in sources:
        resampled_image = nilearn.image.resample_to_img(source, target)
        resampled_image.set_filename(source.get_filename())
        resampled_images.append(resampled_image)
    return np.array(resampled_images)


def normalize_ndarray(ndarray):
    return (ndarray - ndarray.min()) / (ndarray.max() - ndarray.min())


def normalize_image(image):
    voxels = image.get_fdata()
    normalized_image = normalize_ndarray(voxels)
    new = nibabel.Nifti1Image(normalized_image, image.affine)
    new.set_filename(image.get_filename())
    return new


def get_paths(prefix, dataset):
    regexp = os.path.join(prefix, f"*{dataset}*", "**")
    paths = glob.glob(regexp, recursive=True)
    return paths


def load(
    prefix,
    dataset,
    subject,
    template,
    datatype,
    derivative,
):
    """
    Returns T1 + mask images for given prefix, subject and dataset
    """
    root_paths = get_paths(prefix, dataset)
    derivative_path = mri_constants.get_derivatives_path(
        dataset=dataset,
        subject=subject,
        template=template,
        datatype=datatype,
        derivative=derivative,
    )

    derivatives = load_derivatives(root_paths, derivative_path)

    if len(derivatives) == 0:
        print(f"No derivatives {derivative} found")
        print("Root paths: ")
        for path in root_paths[:10]:
            print(f" - {path}")
        if len(root_paths) > 10:
            print("   ...")
            print(f"   [{len(root_paths)} found]")
        print(f"Derivative path: {derivative_path}")
        raise Exception("DerivativeEmpty")

    return derivatives


def load_t1w(prefix, dataset, subject, template, datatype):
    return load(
        prefix=prefix,
        dataset=dataset,
        subject=subject,
        template=template,
        datatype=datatype,
        derivative=mri_constants.Derivative.T1wPreproc,
    )


def load_brain_mask(prefix, dataset, subject, template, datatype):
    return load(
        prefix=prefix,
        dataset=dataset,
        subject=subject,
        template=template,
        datatype=datatype,
        derivative=mri_constants.Derivative.BrainMask,
    )


def get_masked_t1(t1, mask, smooth_kernel, normalize):
    if t1 is None:
        error = ValueError("No T1 provided")
        raise error
    if mask is None:
        error = ValueError(f"No mask provided for {t1.get_filename()} T1 image")
        raise error

    if smooth_kernel == 0:
        smooth_kernel = None
    try:
        masked = apply_mask(imgs=t1, mask_img=mask, smoothing_fwhm=smooth_kernel)
    except TypeError as e:
        raise e

    if normalize:
        masked = normalize_ndarray(masked)

    return masked


def get_unmasked_t1(t1, supermask):
    return nilearn.masking.unmask(t1, supermask)


def _get_masked_t1s(t1s, supermask, smooth_kernel, normalize, cpus):
    n_jobs = min(cpus, len(t1s))
    results = joblib.Parallel(n_jobs=n_jobs, batch_size=1, verbose=1)(
        joblib.delayed(get_masked_t1)(t1, supermask, smooth_kernel, normalize)
        for t1 in t1s
    )
    return results


def get_masked_t1s(t1s, supermask, smooth_kernel, normalize, cpus):
    results = []
    try:
        results = _get_masked_t1s(t1s, supermask, smooth_kernel, normalize, cpus)
    except Exception as e:
        error = ValueError(f"Error while masking T1s: {e}")
        raise e
    return np.stack(results)


def get_template(template):
    from templateflow import api as tflow

    image_path = tflow.get(
        template, desc=None, resolution=1, suffix="T1w", extension="nii.gz"
    )
    mask_path = tflow.get(
        template, desc="brain", resolution=1, suffix="mask", extension="nii.gz"
    )
    image = nibabel.load(image_path)
    mask = nibabel.load(mask_path)
    return mask_image(image, mask.get_fdata())
