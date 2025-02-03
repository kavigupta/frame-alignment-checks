from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch


@dataclass
class ModelToAnalyze:
    model: torch.nn.Module
    model_cl: int
    thresholds: np.ndarray
