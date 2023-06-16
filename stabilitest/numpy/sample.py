import glob
import os

import numpy as np

from stabilitest.sample import Sample


class NumpySample(Sample):
    def __init__(self, args):
        self.args = args
        self.data = None
        self.size = None

    def get_size(self):
        return self.size

    def get_subsample(self, indexes=None):
        return self.data[self.__parse_index(indexes)]

    def __parse_index(self, indexes):
        if indexes is None:
            return ...
        if isinstance(indexes, int):
            return np.array(indexes)
        if isinstance(indexes, list):
            return np.array(indexes)
        raise Exception(f"Unknown index type {type(indexes)}")

    def _load(self, prefix, force):
        if self.data is None or force:
            path_re = os.path.join(prefix, "*", "*.npy")
            paths = glob.glob(path_re)
            self.data = np.stack([np.load(path) for path in paths])
            self.size = self.data.shape[0]

    def load(self, force):
        pass


class NumpySampleReference(NumpySample):
    def load(self, force=False):
        self._load(self.args.reference, force)


class NumpySampleTarget(NumpySample):
    def load(self, force=False):
        self._load(self.args.target, force)


def get_reference_sample(args):
    return NumpySampleReference(args)


def get_target_sample(args):
    return NumpySampleTarget(args)
