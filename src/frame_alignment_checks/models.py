from dataclasses import dataclass

import numpy as np
import torch
import tqdm.auto as tqdm
from permacache import permacache

from .data.load import load_long_canonical_internal_coding_exons, load_validation_gene
from .utils import stable_hash_cached


@dataclass
class ModelToAnalyze:
    model: torch.nn.Module
    model_cl: int
    cl_model_clipped: int
    thresholds: np.ndarray


@permacache(
    "frame_alignment_checks/models/calibration_thresholds_2",
    key_function=dict(m=stable_hash_cached),
)
def calibration_thresholds(m, limit=None):
    """
    Compute calibration thresholds on the genes in the validation set. This is used internally
    for testing, and can be used by a user as well; though we recommend using a larger set of
    genes for calibration.

    :param m: The model to compute calibration thresholds for. It is assumed to output a 3-dimensional
        tensor of shape (N, T, 3) where N is the batch size, T is the sequence length, and 3 is the
        number of classes. They are assumed to be log probabilities.
    :param limit: The number of genes to use for calibration. If None, all genes will be used.

    :returns:
        thresholds: The calibration thresholds for the model. Will be of shape (2,). These thresholds
            are such that the model will predict the correct number of positive examples on average
            in each channel
    """
    m = m.eval().cpu()
    gene_idxs = sorted(
        {exon.gene_idx for exon in load_long_canonical_internal_coding_exons()}
    )
    y_all = []
    yp_all = []
    for gene_idx in tqdm.tqdm(gene_idxs[:limit]):
        # pylint: disable=unsubscriptable-object
        x, y = load_validation_gene(gene_idx)
        with torch.no_grad():
            [yp] = m(torch.tensor(x[None])).softmax(-1)[..., 1:].numpy()
        y_all.append(y)
        yp_all.append(yp)
    yp_all = np.concatenate(yp_all)
    y_all = np.concatenate(y_all)
    frac_actual = y_all.mean(0)[1:]
    thresholds = [np.quantile(yp_all[:, c], 1 - frac_actual[c]) for c in range(2)]
    return np.array(thresholds)
