import json
import os

import faker
import numpy as np

import stabilitest.mri_loader.image as mri_image
import stabilitest.mri_loader.parse_args as mri_args
from stabilitest.parse_args import _default_confidence_values
from stabilitest.sample import Sample
from stabilitest.statistics.distribution import get_distribution_names
from stabilitest.statistics.multiple_testing import get_method_names


def configurator(as_string=False):
    fake = faker.Faker()
    config = {
        "output": "output.pkl",
        "verbose": False,
        "cpus": 1,
        "cached": False,
        "confidence": _default_confidence_values,
        "distribution": get_distribution_names(),
        "parallel-fitting": False,
        "multiple-comparison-tests": get_method_names(),
        "datatype": "anat",
        "reference": {
            "prefix": os.path.dirname(fake.file_path(depth=3)),
            "dataset": fake.numerify("ds######"),
            "subject": fake.numerify("sub-######"),
            "template": fake.numerify("MNI152NLin2009cAsym"),
            "version": fake.numerify("v#.#.#"),
            "architecture": fake.word(),
            "perturbation": fake.word(),
        },
        "target": {
            "prefix": os.path.dirname(fake.file_path(depth=3)),
            "dataset": fake.numerify("ds######"),
            "subject": fake.numerify("sub-######"),
            "template": fake.numerify("MNI152NLin2009cAsym"),
            "version": fake.numerify("v#.#.#"),
            "architecture": fake.word(),
            "perturbation": fake.word(),
        },
        "normalize": True,
        "hyperparameters": {
            "mask-combination": mri_args._defaults_mask_combination,
            "smooth-kernel": mri_args._defaults_smoothing_kernel,
        },
    }
    if as_string:
        return json.dumps(config, indent=2)
    return config


class MRISample(Sample):
    def __init__(self, config, hyperparameters):
        self.config = config
        self.hyperparameters = hyperparameters
        self.t1ws = None
        self.t1ws_metadata = None
        self.brain_masks = None
        self.brain_masks_metadata = None
        self._supermask = None
        # T1ws masked, smoothed and min-max scaled
        self._preproc_t1ws = None
        self.smooth_kernel = hyperparameters["smooth-kernel"]
        self.mask_combination = hyperparameters["mask-combination"]

    def copy(self, sample: "MRISample") -> None:
        self.config = sample.config
        self.hyperparameters = sample.hyperparameters
        self.t1ws = sample.t1ws
        self.t1ws_metadata = sample.t1ws_metadata
        self.brain_masks = sample.brain_masks
        self.brain_masks_metadata = sample.brain_masks_metadata
        self.preprocessed_t1ws = sample.preprocessed_t1ws
        self.smooth_kernel = sample.smooth_kernel
        self.mask_combination = sample.mask_combination

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
        return mri_image.get_masked_t1s(
            t1ws,
            supermask,
            self.smooth_kernel,
            self.config["normalize"],
            self.config["cpus"],
        )

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
        return mri_image.combine_mask(brain_masks, self.mask_combination)

    def _get_raw_data_item(self, indexes):
        if self.t1ws is None:
            raise Exception("Sample not loaded")
        return self.t1ws[self.__parse_index(indexes)]

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
        print("Load reference sample")
        self._load_t1_and_brain_maks(
            prefix=self.config["reference"]["prefix"],
            dataset=self.config["reference"]["dataset"],
            subject=self.config["reference"]["subject"],
            template=self.config["reference"]["template"],
            datatype=self.config["datatype"],
        )
        print(f"Reference sample size: {self.size}")

    def get_info(self, indexes=None):
        if indexes is None or indexes is Ellipsis:
            sample_size = self.size
        else:
            sample_size = len(indexes)
        info = {
            "reference_version": self.config["reference"]["version"],
            "reference_architecture": self.config["reference"]["architecture"],
            "reference_perturbation": self.config["reference"]["perturbation"],
            "reference_prefix": self.config["reference"]["prefix"],
            "reference_dataset": self.config["reference"]["dataset"],
            "reference_subject": self.config["reference"]["subject"],
            "reference_template": self.config["reference"]["template"],
            "reference_sample_size": sample_size,
            "reference_fwhm": self.smooth_kernel,
            "reference_mask": self.mask_combination,
        }

        return info

    def as_target(self):
        """
        return reference sample as a target sample
        """
        config = self.config
        if "target" not in config:
            config["target"] = {}
        config["target"]["version"] = config["reference"]["version"]
        config["target"]["perturbation"] = config["reference"]["perturbation"]
        config["target"]["architecture"] = config["reference"]["architecture"]
        config["target"]["prefix"] = config["reference"]["prefix"]
        config["target"]["dataset"] = config["reference"]["dataset"]
        config["target"]["subject"] = config["reference"]["subject"]
        config["target"]["template"] = config["reference"]["template"]
        target_sample = MRISampleTarget(self.config, self.hyperparameters)
        target_sample.copy(self)
        return target_sample


class MRISampleTarget(MRISample):
    def load(self, force=False):
        print("Load target sample")
        self._load_t1_and_brain_maks(
            prefix=self.config["target"]["prefix"],
            dataset=self.config["target"]["dataset"],
            subject=self.config["target"]["subject"],
            template=self.config["target"]["template"],
            datatype=self.config["datatype"],
        )
        print(f"Target sample size: {self.size}")

    def get_info(self, indexes):
        info = {
            "target_version": self.config["target"]["version"],
            "target_architecture": self.config["target"]["architecture"],
            "target_perturbation": self.config["target"]["perturbation"],
            "target_prefix": self.config["target"]["prefix"],
            "target_dataset": self.config["target"]["dataset"],
            "target_subject": self.config["target"]["subject"],
            "target_template": self.config["target"]["template"],
            "target_filename": self.get_subsample_id(indexes),
        }

        return info


def get_reference_sample(config, hyperparameters=None):
    return MRISampleReference(config, hyperparameters)


def get_target_sample(config, hyperparameters=None):
    return MRISampleTarget(config, hyperparameters)


def preprocess(
    reference_sample: MRISample, reference_ids=None, target_sample=None, target_ids=None
):
    """
    Preprocess reference and target samples
    """
    if reference_ids is None:
        reference_ids = list(range(reference_sample.size))

    print("Compute supermask")
    reference_sample.supermask = reference_sample.compute_supermask(reference_ids)
    print("Preprocess reference sample")
    reference_sample.preprocessed_t1ws = reference_sample._preprocess_t1w(
        reference_ids, reference_sample.supermask
    )

    if target_sample:
        target_sample.supermask = reference_sample.supermask
        print("Resample target images onto reference's shape (if needed)")
        target_sample.resample(reference_sample._get_raw_data_item(0))
        print("Preprocess target sample")
        target_sample.preprocessed_t1ws = target_sample._preprocess_t1w(
            None, reference_sample.supermask
        )
