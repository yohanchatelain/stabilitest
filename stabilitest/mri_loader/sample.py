import nibabel
import numpy as np
from icecream import ic

import stabilitest.mri_loader.image as mri_image
from stabilitest.sample import Sample


def preprocess(reference_sample, reference_ids, target_sample=None, target_ids=None):
    supermask = reference_sample.compute_supermask(reference_ids)
    reference_sample.set_supermask(supermask)
    masked_t1w_ref = reference_sample.get_preproc_t1w(reference_ids, supermask)
    reference_sample.set_preproc_t1w(masked_t1w_ref)

    if target_sample:
        resample_target_img = reference_sample.get_subsample_as_image(0)
        target_sample.set_supermask(supermask)
        target_sample.resample(resample_target_img)
        masked_t1w_target = target_sample.get_preproc_t1w(None, supermask)
        target_sample.set_preproc_t1w(masked_t1w_target)


class MRISample(Sample):
    def __init__(self, args):
        self.args = args
        self.t1ws = None
        self.t1ws_metadata = None
        self.brain_masks = None
        self.brain_masks_metadata = None
        # T1ws masked, smoothed and min-max scaled
        self.preproc_t1ws = None

    def copy(self, sample):
        self.args = sample.args
        self.t1ws = sample.t1ws
        self.t1ws_metadata = sample.t1ws_metadata
        self.brain_masks = sample.brain_masks
        self.brain_masks_metadata = sample.brain_masks_metadata
        self.preproc_t1ws = sample.preproc_t1ws

    def __str__(self):
        return str(self.t1ws)

    def load(self, force):
        pass

    def set_supermask(self, supermask):
        self.supermask = supermask

    def set_preproc_t1ws(self, preproc_t1ws):
        self.preproc_t1ws = preproc_t1ws

    def get_size(self):
        return self.t1ws.shape[0]

    def get_observation_shape(self):
        return self.t1ws.shape[1:]

    def _get_metadata(self, nifti):
        _metadata = [dict(n.header) | {"filename": n.get_filename()} for n in nifti]
        return np.array(_metadata)

    def _load_t1w(self, prefix, dataset, subject, template, datatype):
        self.t1ws = mri_image.load_t1w(
            prefix=prefix,
            dataset=dataset,
            subject=subject,
            template=template,
            datatype=datatype,
        )
        self.t1ws_metadata = self._get_metadata(self.t1ws)

    def _load_brain_mask(self, prefix, dataset, subject, template, datatype):
        self.brain_masks = mri_image.load_brain_mask(
            prefix=prefix,
            dataset=dataset,
            subject=subject,
            template=template,
            datatype=datatype,
        )
        self.brain_masks_metadata = self._get_metadata(self.t1ws)

    def _load_t1_and_brain_maks(
        self, prefix, dataset, subject, template, datatype, force=False
    ):
        self._load_t1w(prefix, dataset, subject, template, datatype)
        self._load_brain_mask(prefix, dataset, subject, template, datatype)

    def compute_supermask(self, indexes):
        brain_masks = self.brain_masks[self.__parse_index(indexes)]
        return mri_image.combine_mask(brain_masks, self.args.mask_combination)

    def get_preproc_t1w(self, indexes, supermask):
        t1ws = self.t1ws[self.__parse_index(indexes)]
        return mri_image.get_masked_t1s(self.args, t1ws, supermask)

    def get_subsample_as_image(self, indexes):
        return self.t1ws[self.__parse_index(indexes)]

    def set_preproc_t1w(self, t1w):
        self.preproc_t1ws = t1w

    def __parse_index(self, indexes):
        if indexes is None:
            return ...
        if indexes is ...:
            return ...
        if isinstance(indexes, np.ndarray):
            return indexes
        if isinstance(indexes, list):
            return np.array(indexes)
        try:
            indexes = int(indexes)
            return np.array(indexes)
        except Exception:
            raise Exception(f"Unknown index type {type(indexes)}")

    def get_subsample(self, indexes=None):
        return self.preproc_t1ws[self.__parse_index(indexes)]

    def get_subsample_id(self, indexes=None):
        meta = self.t1ws_metadata[self.__parse_index(indexes)]
        if isinstance(meta, np.ndarray):
            return [m["filename"] for m in meta]
        else:
            return meta["filename"]

    def resample(self, target):
        self.t1ws = mri_image.resample_images(self.t1ws, target)

    def dump(self, data_1d, filename):
        img = mri_image.get_unmasked_t1(data_1d, self.supermask)
        img.to_filename(filename)


class MRISampleReference(MRISample):
    def load(self, force=False):
        self._load_t1_and_brain_maks(
            prefix=self.args.reference_prefix,
            dataset=self.args.reference_dataset,
            subject=self.args.reference_subject,
            template=self.args.reference_template,
            datatype=self.args.datatype,
        )

    def get_info(self, indexes=None):
        if indexes is None or indexes is Ellipsis:
            sample_size = self.get_size()
        else:
            sample_size = len(indexes)
        info = {
            "reference_version": self.args.reference_version,
            "reference_architecture": self.args.reference_architecture,
            "reference_perturbation": self.args.reference_perturbation,
            "reference_prefix": self.args.reference_prefix,
            "reference_dataset": self.args.reference_dataset,
            "reference_subject": self.args.reference_subject,
            "reference_template": self.args.reference_template,
            "reference_sample_size": sample_size,
            "reference_fwhm": self.args.smooth_kernel,
            "reference_mask": self.args.mask_combination,
        }

        return info

    def as_target(self):
        """
        return reference sample as a target sample
        """
        args = self.args
        args.target_version = args.reference_version
        args.target_perturbation = args.reference_perturbation
        args.target_architecture = args.reference_architecture
        args.target_prefix = args.reference_prefix
        args.target_dataset = args.reference_dataset
        args.target_subject = args.reference_subject
        args.target_template = args.reference_template
        target_sample = MRISampleTarget(self.args)
        target_sample.copy(self)
        return target_sample


class MRISampleTarget(MRISample):
    def load(self, force=False):
        self._load_t1_and_brain_maks(
            prefix=self.args.target_prefix,
            dataset=self.args.target_dataset,
            subject=self.args.target_subject,
            template=self.args.target_template,
            datatype=self.args.datatype,
        )

    def get_info(self, indexes):
        info = {
            "target_version": self.args.target_version,
            "target_architecture": self.args.target_architecture,
            "target_perturbation": self.args.target_perturbation,
            "target_prefix": self.args.target_prefix,
            "target_dataset": self.args.target_dataset,
            "target_subject": self.args.target_subject,
            "target_template": self.args.target_template,
            "target_filename": self.get_subsample_id(indexes),
        }

        return info


def get_reference_sample(args):
    return MRISampleReference(args)


def get_target_sample(args):
    return MRISampleTarget(args)
