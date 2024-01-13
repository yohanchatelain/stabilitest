import glob
import json
import os

import faker
import numpy as np

from stabilitest.parse_args import _default_confidence_values
from stabilitest.sample import Sample
from stabilitest.statistics.distribution import get_distribution_names
from stabilitest.statistics.multiple_testing import get_method_names


def preprocess(
    reference_sample, reference_ids=None, target_sample=None, target_ids=None
):
    pass


def configurator(args):
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
        "reference": os.path.dirname(fake.file_path(depth=3)),
        "target": os.path.dirname(fake.file_path(depth=3)),
        "normalize": True,
        "hyperparameters": {},
    }
    return json.dumps(config, indent=2)


class NumpySample(Sample):
    def __init__(self, config, hyperparameters=None):
        self.config = config
        self.hyperparameters = hyperparameters
        self._data = None
        self._size = None
        self.paths = None

    def copy(self, sample):
        self.config = sample.config
        self.hyperparameters = sample.hyperparameters
        self.data = sample.data
        self.size = sample.size
        self.paths = sample.paths

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = size

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def metadata(self):
        return self.paths

    def get_subsample(self, indexes=None):
        if self.data is None:
            raise Exception("Data not loaded")
        return self.data[self.__parse_index(indexes)]

    def get_subsample_id(self, indexes):
        if self.paths is None:
            raise Exception("Data not loaded")
        return self.paths[self.__parse_index(indexes)]

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

    def _load(self, prefix, force):
        if self.data is None or force:
            path_re = os.path.join(prefix, "*", "*.npy")
            self.paths = glob.glob(path_re)
            self.data = np.stack([np.load(path) for path in self.paths])
            self.size = self.data.shape[0]

    def resample(self, target):
        pass

    def load(self, force):
        pass

    def dump(self, data_1d, filename):
        np.save(filename, data_1d)


class NumpySampleReference(NumpySample):
    def load(self, force=False):
        self._load(self.config["reference"], force)

    def get_info(self, indexes=None):
        info = {
            "reference": self.config["reference"],
            "normalize": self.config["normalize"],
        }

        return info

    def as_target(self):
        config = self.config
        config["target"] = config["reference"]
        sample = NumpySampleTarget(config)
        sample.copy(self)
        return sample


class NumpySampleTarget(NumpySample):
    def load(self, force=False):
        self._load(self.config["target"], force)

    def get_info(self, indexes):
        info = {
            "target": self.config["target"],
            "target_filename": self.get_subsample_id(indexes),
            "normalize": self.config["normalize"],
        }
        return info


def get_reference_sample(config, hyperparameters=None):
    return NumpySampleReference(config, hyperparameters)


def get_target_sample(config, hyperparameters=None):
    return NumpySampleTarget(config, hyperparameters)
