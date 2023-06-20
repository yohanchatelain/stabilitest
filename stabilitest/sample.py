from abc import ABC, abstractmethod


class Sample(ABC):
    @abstractmethod
    def get_size(self):
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
