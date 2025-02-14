from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from .math import mean_quantile


@dataclass
class RealExperimentResultForModel:
    """
    Results of running a real experiment for a model on a list of exons.

    :param actual: are the real log PSI values for the exons. Shape ``(num_exons,)``.
    :param predicted: the predicted log PSI values for the exons. Shape ``(num_seeds, num_exons)``.
    """

    actual: np.ndarray  # (N,) floats
    predicteds: List[np.ndarray]  # (S, N) floats

    def compute_mean_quantile_each(self, masks):
        """
        Compute the mean quantile for each mask. The mean quantile is the mean of the quantiles
        of the predicted values among the 100 closest predictions among the actual values.

        :param masks: a list of tuples, each containing a mask and a name. The mask is a boolean array
            of shape ``(num_exons,)`` indicating which exons were used in the experiment. The name is a string
            identifying the mask.

        :return: an array of shape ``(num_seeds, num_masks)`` containing the mean quantile for each mask.
        """
        return [
            mean_quantile(
                self.actual, predicted, np.array([mask for mask, _ in masks]), k=100
            )
            for predicted in self.predicteds
        ]


@dataclass
class FullRealExperimentResult:
    """
    Results of running a real experiment for multiple models on a list of exons.

    :param er_by_model: a dictionary mapping model names to ``fac.RealExperimentResultForModel``.
    :param masks_each: a list of tuples, each containing a mask and a name. The mask is a boolean array
        of shape ``(num_exons,)`` indicating which exons were used in the experiment. The name is a string
        identifying the mask.
    """

    er_by_model: Dict[str, RealExperimentResultForModel]
    masks_each: List[Tuple[str, np.ndarray]]

    def mean_quantiles_each(self):
        """
        Mean quantiles for each model, for each mask.

        :return: a dictionary mapping model names to arrays of shape ``(num_seeds, num_masks)``.
        """
        return {
            name: er.compute_mean_quantile_each(self.masks_each)
            for name, er in self.er_by_model.items()
        }

    def filter_models(self, func: Callable[[str], bool]) -> "FullRealExperimentResult":
        """
        Filter the models in this result, keeping only those for which ``func`` returns ``True``
        when called with the model name.
        """
        return FullRealExperimentResult(
            {name: er for name, er in self.er_by_model.items() if func(name)},
            self.masks_each,
        )

    def map_model_keys(self, func: Callable[[str], str]) -> "FullRealExperimentResult":
        """
        Map the model names in this result using ``func``.
        """
        return FullRealExperimentResult(
            {func(name): er for name, er in self.er_by_model.items()},
            self.masks_each,
        )

    @classmethod
    def merge(
        cls, er_by_models: List["FullRealExperimentResult"]
    ) -> "FullRealExperimentResult":
        """
        Merge multiple ``FullRealExperimentResult`` instances into one. The masks must be the same
        for all instances.
        """
        er_by_model = {}
        masks_each = None
        for er_by_model_this in er_by_models:
            if masks_each is None:
                masks_each = er_by_model_this.masks_each
            else:
                from numpy.testing import assert_array_equal

                for (mask1, name1), (mask2, name2) in zip(
                    masks_each, er_by_model_this.masks_each
                ):
                    assert_array_equal(mask1, mask2)
                    assert name1 == name2
            er_by_model.update(er_by_model_this.er_by_model)
        return cls(er_by_model, masks_each)
