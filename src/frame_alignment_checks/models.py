from dataclasses import dataclass

import numpy as np
import torch
import tqdm.auto as tqdm
from permacache import permacache

from .load_data import load_long_canonical_internal_coding_exons, load_validation_gene
from .utils import stable_hash_cached


@dataclass
class ModelToAnalyze:
    """
    Model to analyze for frame alignment checks. Contains the module as well as metadata
    regarding the model.

    :param model: The model to analyze. This model is assumed to output a 3-dimensional tensor
        of shape (N, T, 3) where N is the batch size, T is the sequence length, and 3 is the
        number of classes. The model is assumed to output logit probabilities (they ill be
        softmaxed internally).
    :param model_cl: The context length of the model. This is the number of bases on each side
        of the central base that the model uses for prediction. This is used for padding the
        input sequences.
    :param cl_model_clipped: The amount the model clips from each side. The amount the
        model clips from the input sequence. I.e.., if the input is of size (N, T, 4), the
        output will be of size (N, T - cl_model_clipped, 3). For some models, this is the
        same as model_cl (e.g., SpliceAI-400 has 400 for both), but for others, this is
        smaller (e.g., SAM-AM requires 5400nt of context but only clips 400 for efficiency).
    :param thresholds: The calibration thresholds for the model. These are such that the model
        will predict the correct number of positive examples on average in each channel on
        a valadition set of interest. Shape: (2,), no threshold for the first channel. These
        thresholds should be in the range (0, 1), i.e., softmaxed probabilities.
    """

    model: torch.nn.Module
    model_cl: int
    cl_model_clipped: int
    thresholds: np.ndarray


@permacache(
    "frame_alignment_checks/models/calibration_thresholds_2",
    key_function=dict(m=stable_hash_cached),
)
def calibration_accuracy_and_thresholds(m, mcl, *, limit=None):
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
        x = np.pad(x, ((mcl // 2, mcl // 2), (0, 0)))
        with torch.no_grad():
            [yp] = m(torch.tensor(x[None])).softmax(-1)[..., 1:].numpy()
        y_all.append(y[:, 1:])
        yp_all.append(yp)
    yp_all = np.concatenate(yp_all)
    y_all = np.concatenate(y_all)
    assert yp_all.shape == y_all.shape
    frac_actual = y_all.mean(0)
    thresholds = [np.quantile(yp_all[:, c], 1 - frac_actual[c]) for c in range(2)]
    acc = ((yp_all > thresholds) & (y_all > 0.5)).sum(0) / (y_all > 0.5).sum(0)
    return acc, np.array(thresholds)
