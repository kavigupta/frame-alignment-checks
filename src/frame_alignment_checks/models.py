from typing import Dict, List

import numpy as np
import torch


from dataclasses import dataclass


@dataclass
class ModelToAnalyze:
    model: torch.nn.Module
    model_cl: int
    thresholds: np.ndarray


@dataclass
class ModelsToAnalyze:
    models: Dict[str, List[ModelToAnalyze]]
