import glob
import os

import numpy as np

from stabilitest.sample import Sample


def preprocess(reference_sample, reference_ids, target_sample, target_ids):
    pass


class NumpySample(Sample):
    def __init__(self, args):
        self.args = args
        self.data = None
        self.size = None
        self.paths = None

    def copy(self, sample):
        self.args = sample.args
        self.data = sample.data
        self.size = sample.size
        self.paths = sample.paths

    def get_size(self):
        return self.size

    def get_subsample(self, indexes=None):
        return self.data[self.__parse_index(indexes)]

    def get_subsample_id(self, indexes):
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


class NumpySampleReference(NumpySample):
    def load(self, force=False):
        self._load(self.args.reference, force)

    def get_info(self, indexes=None):
        info = {"reference": self.args.reference, "normalize": self.args.normalize}

        return info

    def as_target(self):
        args = self.args
        args.target = args.reference
        args.normalize = args.normalize
        sample = NumpySampleTarget(args)
        sample.copy(self)
        return sample


class NumpySampleTarget(NumpySample):
    def load(self, force=False):
        self._load(self.args.target, force)

    def get_info(self, indexes):
        info = {
            "target": self.args.target,
            "target_filename": self.get_subsample_id(indexes),
            "normalize": self.args.normalize,
        }
        return info


def get_reference_sample(args):
    return NumpySampleReference(args)


def get_target_sample(args):
    return NumpySampleTarget(args)
