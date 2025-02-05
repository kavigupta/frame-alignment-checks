import torch

from frame_alignment_checks import ModelToAnalyze
from frame_alignment_checks.models import calibration_thresholds

from .lssi import load_with_remapping_pickle

cl_models = 100


def clip(yp):
    return yp[:, cl_models // 2 : -(cl_models // 2)]


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
        yp = self.compute_without_cl(x)
        return clip(yp)

    def compute_without_cl(self, x):
        acceptor = self.acceptor(x).log_softmax(-1)[:, :, [1]]
        donor = self.donor(x).log_softmax(-1)[..., [2]]
        null = torch.log(1 - torch.exp(acceptor) - torch.exp(donor))
        yp = torch.cat([null, acceptor, donor], dim=-1)
        return yp


def lssi_model():
    m = SpliceModel().eval()
    thresholds = calibration_thresholds(m)
    return ModelToAnalyze(m, cl_models, cl_models, thresholds)
