from abc import *


class DiffusionModel(metaclass=ABCMeta):
    @abstractmethod
    def compute_noise_preds(self, x_t):
        """
        x_t -> epsilon_t
        """
        pass