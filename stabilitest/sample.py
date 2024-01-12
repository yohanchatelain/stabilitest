from abc import ABC, abstractmethod
import numpy as np


def __parse_index(indexes):
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


class AbstractArrayEmpty(Exception):
    pass


class AbstractArray:
    def __init__(self, data=None):
        self._data = data

    def __getitem__(self, indexes):
        if self._data is None:
            raise AbstractArrayEmpty
        return self._data[__parse_index(indexes)]

    def not_(self):
        return not self._data

    def is_(self, other):
        return self._data is other

    def is_not(self, other):
        return self._data is not other

    def __len__(self):
        if self._data is None:
            raise AbstractArrayEmpty
        return len(self._data)

    def truth(self):
        return bool(self._data)

    def __bool__(self):
        return bool(self._data)


class Sample(ABC):
    @property
    @abstractmethod
    def size(self):
        pass

    @property
    @abstractmethod
    def data(self):
        pass

    @property
    @abstractmethod
    def metadata(self):
        pass

    @abstractmethod
    def load(self, force):
        pass

    @abstractmethod
    def get_subsample(self, indexes):
        pass

    @abstractmethod
    def get_subsample_id(self, indexes):
        pass

    @abstractmethod
    def resample(self, target):
        pass

    @abstractmethod
    def get_info(self, indexes):
        pass


class SampleReference(Sample):
    @abstractmethod
    def as_target(self, sample):
        pass


class SampleTarget(Sample):
    pass
