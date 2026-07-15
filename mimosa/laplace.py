from abc import abstractmethod
import equinox as eqx

from mimosa.data_structures import Dataset


class LaplaceApproximator(eqx.Module):
    @abstractmethod
    def wrap(self, dataset: Dataset) -> Dataset:
        pass

    @abstractmethod
    def unwrap(self, dataset: Dataset) -> Dataset:
        pass

class IdentityLaplaceApproximator(LaplaceApproximator):
    def wrap(self, dataset: Dataset) -> Dataset:
        return dataset

    def unwrap(self, dataset: Dataset) -> Dataset:
        return dataset