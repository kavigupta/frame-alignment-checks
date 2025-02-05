import torch
from frame_alignment_checks import (
    accuracy_delta_given_deletion_experiment,
    ModelToAnalyze,
)
from frame_alignment_checks.models import calibration_thresholds

from .lssi import load_with_remapping_pickle

cl_models = 100


class SpliceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.acceptor = load_with_remapping_pickle(
            "tests/data/acceptor.m",
            weights_only=False,
            map_location=torch.device("cpu"),
        )
        self.donor = load_with_remapping_pickle(
            "tests/data/donor.m",
            weights_only=False,
            map_location=torch.device("cpu"),
        )
        for m in [self.acceptor, self.donor]:
            m.conv_layers[0].clipping = "none"

    def forward(self, x):
        acceptor = self.acceptor(x).log_softmax(-1)[:, :, [1]]
        donor = self.donor(x).log_softmax(-1)[..., [2]]
        null = 1 - torch.exp(acceptor) - torch.exp(donor)
        return torch.cat([null, acceptor, donor], dim=-1)[
            :, cl_models // 2 : -(cl_models // 2)
        ]


def lssi_model():
    m = SpliceModel().eval()
    thresholds = calibration_thresholds(m)
    return ModelToAnalyze(m, cl_models, cl_models, thresholds)
