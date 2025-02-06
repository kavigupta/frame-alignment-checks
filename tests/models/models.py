import numpy as np
import torch

from frame_alignment_checks import ModelToAnalyze
from frame_alignment_checks.compute_stop_codons import all_frames_closed
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


class SpliceModelWithORF(torch.nn.Module):
    def __init__(self, cl_model, orf_radius=100):
        super().__init__()
        self.orf_radius = orf_radius
        self.splice_model = SpliceModel(cl_model)

    def forward(self, x):
        yp = self.splice_model.compute_without_cl(x)
        sites = np.where(yp[..., 1:] > -10)
        for batch_idx, seq_idx, site_type in zip(*sites):
            if site_type == 0:
                grab_zone = x[batch_idx, seq_idx + 2 : seq_idx + self.orf_radius]
            elif site_type == 1:
                grab_zone = x[
                    batch_idx, max(seq_idx - self.orf_radius, 0) : max(seq_idx - 2, 0)
                ]
            else:
                raise ValueError(site_type)
            [is_closed] = all_frames_closed([grab_zone])
            if is_closed:
                yp[batch_idx, seq_idx, 0] = 0
                yp[batch_idx, seq_idx, 1:] = -1000
        return self.splice_model.clip(yp)

    @property
    def cl_model(self):
        return self.splice_model.cl_model


def calibrated_model(m):
    m = m.eval()
    acc, thresholds = calibration_accuracy_and_thresholds(m, m.cl_model)
    print(acc, thresholds)
    return ModelToAnalyze(m, m.cl_model, m.cl_model, thresholds)


def lssi_model():
    return calibrated_model(SpliceModel(cl_model=100))


def lssi_model_with_orf():
    return calibrated_model(SpliceModelWithORF(cl_model=400, orf_radius=200))
