from distutils import extension
import numpy as np
from icecream import ic
import json
import faker
import os

import stabilitest.mri_loader.image as mri_image
from stabilitest.sample import Sample


def configurator(args):
    fake = faker.Faker()
    config = {
        "general": {
            "output": fake.file_name(extension="pkl"),
            "verbose": fake.boolean(),
            "cpus": fake.pyint(),
            "cached": fake.boolean(),
        },
        "confidence": [
            fake.pyfloat(left_digits=1, right_digits=4, min_value=0, max_value=1)
        ],
        "distribution": [fake.word()],
        "parallel-fitting": fake.boolean(),
        "multiple-comparison-tests": [fake.word()],
        "model": fake.word(),
        "k-fold-rounds": fake.pyint(),
        "reference": {
            "prefix": os.path.dirname(fake.file_path(depth=3)),
            "dataset": fake.numerify("ds######"),
            "subject": fake.numerify("sub-######"),
            "template": fake.numerify("MNI###NLin###Asym"),
            "version": fake.numerify("v#.#.#"),
            "architecture": fake.word(),
            "perturbation": fake.word(),
        },
        "target": {
            "prefix": os.path.dirname(fake.file_path(depth=3)),
            "dataset": fake.numerify("ds######"),
            "subject": fake.numerify("sub-######"),
            "template": fake.numerify("MNI###NLin###Asym"),
            "version": fake.numerify("v#.#.#"),
            "architecture": fake.word(),
            "perturbation": fake.word(),
        },
        "normalize": fake.boolean(),
        "mask-combination": fake.word(),
        "smooth-kernel": [fake.pyint()],
    }
    return json.dumps(config, indent=2)


class MRISample(Sample):
    def __init__(self, args):
        self.args = args
        self._config = self._load_config(args.config_file)
        self.t1ws = None
        self.t1ws_metadata = None
        self.brain_masks = None
        self.brain_masks_metadata = None
        self._supermask = None
        # T1ws masked, smoothed and min-max scaled
        self._preproc_t1ws = None

    def _load_config(self, config_file):
        return json.load(open(config_file))

    def copy(self, sample):
        self.args = sample.args
        self.config_file = sample.config_file
        self.t1ws = sample.t1ws
        self.t1ws_metadata = sample.t1ws_metadata
        self.brain_masks = sample.brain_masks
        self.brain_masks_metadata = sample.brain_masks_metadata
        self.preprocessed_t1ws = sample.preproc_t1ws

    def __str__(self):
        return str(self.t1ws)

    def load(self, force):
        pass

    @property
    def supermask(self):
        return self._supermask

    @supermask.setter
    def supermask(self, supermask):
        self._supermask = supermask

    @property
    def preprocessed_t1ws(self):
        return self._preproc_t1ws

    @preprocessed_t1ws.setter
    def preprocessed_t1ws(self, preproc_t1ws):
        self._preproc_t1ws = preproc_t1ws

    def _preprocess_t1w(self, indexes, supermask):
        if self.t1ws is None:
            raise Exception("Sample not loaded")
        t1ws = self.t1ws[self.__parse_index(indexes)]
        return mri_image.get_masked_t1s(self.args, t1ws, supermask)

    @property
    def size(self):
        if self.t1ws is None:
            raise Exception("Sample not loaded")
        return self.t1ws.shape[0]

    @property
    def data(self):
        return self.preprocessed_t1ws

    @property
    def metadata(self):
        return self.t1ws_metadata

    def get_observation_shape(self):
        if self.t1ws is None:
            raise Exception("Sample not loaded")
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
        if self.brain_masks is None:
            raise Exception("Brain masks sample not loaded")
        brain_masks = self.brain_masks[self.__parse_index(indexes)]
        return mri_image.combine_mask(brain_masks, self.args.mask_combination)

    def _get_raw_data_item(self, indexes):
        if self.t1ws is None:
            raise Exception("Sample not loaded")
        return self.t1ws[self.__parse_index(indexes)]

    # def __getitem__(self, indexes):
    #     if self.t1ws is None:
    #         raise Exception("Sample not loaded")
    #     return self.t1ws[self.__parse_index(indexes)]

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
        if self.preprocessed_t1ws is None:
            raise Exception("Sample not loaded")
        return self.preprocessed_t1ws[self.__parse_index(indexes)]

    def get_subsample_id(self, indexes=None):
        if self.t1ws_metadata is None:
            raise Exception("Sample not loaded")
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
        ic("Load reference sample")
        self._load_t1_and_brain_maks(
            prefix=self.args.reference_prefix,
            dataset=self.args.reference_dataset,
            subject=self.args.reference_subject,
            template=self.args.reference_template,
            datatype=self.args.datatype,
        )

    def get_info(self, indexes=None):
        if indexes is None or indexes is Ellipsis:
            sample_size = self.size
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
        ic("Load target sample")
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


def preprocess(
    reference_sample: MRISample, reference_ids=None, target_sample=None, target_ids=None
):
    """
    Preprocess reference and target samples
    """
    if reference_ids is None:
        reference_ids = list(range(reference_sample.size))

    reference_sample.supermask = reference_sample.compute_supermask(reference_ids)
    reference_sample.preprocessed_t1ws = reference_sample._preprocess_t1w(
        reference_ids, reference_sample.supermask
    )

    if target_sample:
        target_sample.supermask = reference_sample.supermask
        target_sample.resample(reference_sample._get_raw_data_item(0))
        target_sample.preprocessed_t1ws = target_sample._preprocess_t1w(
            None, reference_sample.supermask
        )
