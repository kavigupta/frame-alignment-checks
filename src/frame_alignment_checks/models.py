from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ModelToAnalyze:
    model: torch.nn.Module
    model_cl: int
    cl_model_clipped: int
    thresholds: np.ndarray
