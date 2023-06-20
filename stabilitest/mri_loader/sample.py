import numpy as np

import stabilitest.mri_loader.image as mri_image
from stabilitest.sample import Sample


class MRISample(Sample):
    def __init__(self, args):
        self.args = args
        self.t1ws = None
        self.t1ws_metadata = None
        self.brain_masks = None
        self.brain_masks_metadata = None
        self.supermask = None
        # T1ws masked, smoothed and min-max scaled
        self.preproc_t1ws = None

    def copy(self, sample):
        self.args = sample.args
        self.t1ws = sample.t1ws
        self.t1ws_metadata = sample.t1ws_metadata
        self.brain_masks = sample.brain_masks
        self.brain_masks_metadata = sample.brain_masks_metadata
        self.supermask = sample.supermask
        self.preproc_t1ws = sample.preproc_t1ws

    def load(self, force):
        pass

    def get_size(self):
        return self.t1ws.shape[0]

    def get_observation_shape(self):
        return self.t1ws.shape[1:]

    def _get_metadata(self, nifti):
        _metadata = [dict(n.header) | {"filename": nifti.get_filename()} for n in nifti]
        return np.array(_metadata)

    def load_t1w(self, prefix, dataset, subject, template, datatype, force=False):
        if self.t1ws is None or force:
            self.t1ws = mri_image.load_t1w(
                prefix=prefix,
                dataset=dataset,
                subject=subject,
                template=template,
                datatype=datatype,
            )
            self.t1ws_metadata = self._get_metadata(self.t1ws)

    def load_brain_mask(
        self, prefix, dataset, subject, template, datatype, force=False
    ):
        if self.brain_masks is None or force:
            self.brain_masks = mri_image.load_brain_mask(
                prefix=prefix,
                dataset=dataset,
                subject=subject,
                template=template,
                datatype=datatype,
            )
            self.brain_masks_metadata = self._get_metadata(self.t1ws)

    def load_t1_and_brain_maks(
        self, prefix, dataset, subject, template, datatype, force=False
    ):
        self.load_t1w(prefix, dataset, subject, template, datatype, force)
        self.brain_masks(prefix, dataset, subject, template, datatype, force)

    def __compute_supermask(self, force=False):
        if force or self.supermask is None:
            self.supermask = mri_image.combine_mask(
                self.brain_masks, self.args.mask_combination
            )

    def __mask_sample(self, force=False):
        if self.preproc_t1ws is None or force:
            self.__compute_supermask(force)
            self.preproc_t1ws = mri_image.get_masked_t1s(
                self.args, self.t1ws, self.supermask
            )

    def __parse_index(self, indexes):
        if indexes is None:
            return ...
        if isinstance(indexes, int):
            return np.array(indexes)
        if isinstance(indexes, list):
            return np.array(indexes)
        raise Exception(f"Unknown index type {type(indexes)}")

    def get_subsample(self, indexes=None):
        return self.preproc_t1ws[self.__parse_index(indexes)]

    def get_subsample_id(self, indexes=None):
        meta = self.t1ws_metadata[self.__parse_index(indexes)]
        return [m["filename"] for m in meta]

    def resample(self, target):
        self.preproc_t1ws = mri_image.resample_image(self.preproc_t1ws, target)


class MRISampleReference(MRISample):
    def load(self, force=False):
        self.load_t1_and_brain_maks(
            prefix=self.args.reference_prefix,
            dataset=self.args.reference_dataset,
            subject=self.args.reference_subject,
            template=self.args.reference_template,
            datatype=self.args.datatype,
        )
        self.__mask_sample(force)

    def get_info(self, indexes):
        info = {
            "reference_prefix": self.args.reference_prefix,
            "reference_dataset": self.args.reference_dataset,
            "reference_template": self.args.reference_template,
            "reference_sample_size": self.get_size(),
            "reference_fwhm": self.args.smooth_kernel,
            "reference_mask": self.args.combination_mask,
        }

        return info

    def as_target(self):
        """
        return reference sample as a target sample
        """
        args = self.args
        args.target_prefix = args.reference_prefix
        args.target_dataset = args.reference_dataset
        args.target_subject = args.reference_subject
        args.target_template = args.reference_template
        target_sample = MRISampleTarget(self.args)
        target_sample.copy(self)
        return target_sample


class MRISampleTarget(MRISample):
    def load(self, force=False):
        self.load_t1_and_brain_maks(
            prefix=self.args.target_prefix,
            dataset=self.args.target_dataset,
            subject=self.args.target_subject,
            template=self.args.target_template,
            datatype=self.args.datatype,
        )
        self.__mask_sample(force)

    def get_info(self, indexes):
        info = {
            "target_prefix": self.args.target_prefix,
            "target_dataset": self.args.target_dataset,
            "target_template": self.args.target_template,
            "target_filename": self.get_subsample_id(indexes),
        }

        return info


def get_reference_sample(args):
    return MRISampleReference(args)


def get_target_sample(args):
    return MRISampleTarget(args)
