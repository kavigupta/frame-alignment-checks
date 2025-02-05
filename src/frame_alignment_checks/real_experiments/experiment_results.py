from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .math import mean_quantile


@dataclass
class RealExperimentResultForModel:
    actual: np.ndarray  # (N,) floats
    predicteds: List[np.ndarray]  # (S, N) floats

    def compute_mean_quantile_each(self, masks):
        return [
            mean_quantile(
                self.actual, predicted, np.array([mask for mask, _ in masks]), k=100
            )
            for predicted in self.predicteds
        ]


@dataclass
class FullRealExperimentResult:
    er_by_model: Dict[str, RealExperimentResultForModel]
    masks_each: List[Tuple[str, np.ndarray]]

    def mean_quantiles_each(self):
        return {
            name: er.compute_mean_quantile_each(self.masks_each)
            for name, er in self.er_by_model.items()
        }

    def filter_models(self, func):
        return FullRealExperimentResult(
            {name: er for name, er in self.er_by_model.items() if func(name)},
            self.masks_each,
        )

    def map_model_keys(self, func):
        return FullRealExperimentResult(
            {func(name): er for name, er in self.er_by_model.items()},
            self.masks_each,
        )

    @classmethod
    def merge(cls, er_by_models):
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
