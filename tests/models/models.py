import torch

from frame_alignment_checks import ModelToAnalyze
from frame_alignment_checks.models import calibration_accuracy_and_thresholds

from .lssi import load_with_remapping_pickle


class SpliceModel(torch.nn.Module):

    def __init__(self, cl_model):
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
        self.cl_model = cl_model

    def forward(self, x):
        yp = self.compute_without_cl(x)
        return self.clip(yp)

    def compute_without_cl(self, x):
        acceptor = self.acceptor(x).log_softmax(-1)[..., [1]]
        donor = self.donor(x).log_softmax(-1)[..., [2]]
        null = torch.log(1 - torch.exp(acceptor) - torch.exp(donor))
        yp = torch.cat([null, acceptor, donor], dim=-1)
        return yp

    def clip(self, yp):
        return yp[:, self.cl_model // 2 : -(self.cl_model // 2)]


def calibrated_model(m):
    m = m.eval()
    acc, thresholds = calibration_accuracy_and_thresholds(m, m.cl_model)
    print(acc, thresholds)
    return ModelToAnalyze(m, m.cl_model, m.cl_model, thresholds)


def lssi_model():
    return calibrated_model(SpliceModel(cl_model=100))
