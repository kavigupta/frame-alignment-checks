from functools import lru_cache

import numpy as np
import tqdm.auto as tqdm
from permacache import permacache
from run_batched import run_batched

from ..load_data import load_non_stop_donor_windows
from ..models import ModelToAnalyze
from ..utils import all_seqs, device_of, extract_center, stable_hash_cached

histogram_min_value = -20
histogram_resolution = 0.001


@lru_cache(maxsize=None)
def all_9mers():
    return np.array(list(all_seqs(9)))


def phase_handedness_self_agreement_score_for_multiple_series(models, can_seq, mode):
    """
    Compute swapping results for the given models.

    models: A dictionary of models, where the keys are the names of the models and the values are lists of models.
    cl_model: The context length of the model.
    can_seq: The canonical 9mers for each phase.
    mode: The mode to use. Can be "log_prob" or "quantile".

    Returns a dictionary of the results, where the keys are the names of the models and the values are the results
        for each seed. The results are the mean self-agreement scores for each pair of phases.
    """
    multiplier = 1
    if mode == "percentile":
        multiplier = 100
        mode = "quantile"
    results = {}
    for name in tqdm.tqdm(models):
        results[name] = [
            multiplier * phase_handedness_self_agreement_score(m, can_seq, mode=mode)
            for m in models[name]
        ]
    return results


def phase_handedness_self_agreement_score(m: ModelToAnalyze, can_seq, *, mode):
    """
    Returns self-agreement scores, per pair of phases.

    A self-agreement score is how much a model agrees with itself when the 9mer phase and the donor phase are the same
        versus when they are different.
    """
    by_phase = phase_swapping_experiment(
        m.model, m.cl_model_clipped, can_seq, mode=mode
    )
    diag = np.eye(3, dtype=np.bool)
    return np.mean(by_phase[diag]) - np.mean(by_phase[~diag])


@permacache(
    "frame_alignment_checks/phase_handedness/compute_self_agreement/phase_swapping_experiment",
    key_function=dict(m=stable_hash_cached),
)
def phase_swapping_experiment(m, cl_model, can_seq, *, mode):
    """
    Returns by_phase:
        by_phase[9mer_phase][donor_phase] is the mean log likelihood of the canonical 9mer set for the 9mer phase
        evaluated with donors of the donor phase.
    """
    result = []
    pbar = tqdm.tqdm(total=sum(len(can_seq[phase]) for phase in range(3)))
    for phase in range(3):
        to_mean = []
        for seq in all_9mers()[can_seq[phase]].argmax(-1):
            to_mean.append(mean_score_by_phase(m, cl_model, seq, mode=mode))
            pbar.update()
        result.append(np.mean(to_mean, 0))
    pbar.close()
    return np.array(result)


@permacache(
    "frame_alignment_checks/phase_handedness/compute_self_agreement/compute_donor_scores",
    key_function=dict(m=stable_hash_cached),
)
def compute_donor_scores(m, cl_model, donor_9mer):
    """
    Compute the donor scores for the given model and data, optionally replacing the core
    donor 9mer with donor_9mer.
    """
    good_sequences, _, _ = load_non_stop_donor_windows()
    center = good_sequences.shape[1] // 2
    data = good_sequences[
        :, center - cl_model // 2 : center + cl_model // 2 + 1
    ].astype(np.int8)
    if donor_9mer is not None:
        data[:, cl_model // 2 - 2 : cl_model // 2 + 7] = donor_9mer
    data = np.eye(4, dtype=np.float32)[data] * (data >= 0)[..., None]
    result = run_batched(
        lambda x: extract_center(m, x),
        data,
        128,
        pbar=tqdm.tqdm,
        device=device_of(m),
    )[:, 2]
    return result


def mean_score_by_phase(m, cl_model, donor_9mer, *, mode):
    """
    Mean log likelihood of the canonical donor set, grouped by donor phase.
    """
    _, good_phases, _ = load_non_stop_donor_windows()
    scores = compute_donor_scores(m, cl_model, donor_9mer)
    scores = np.log(scores)
    if mode == "log_prob":
        pass
    elif mode == "quantile":
        cumul = cumulative_prob(m, cl_model)
        scores = quantile_scores(cumul, scores)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    result = np.array([scores[good_phases == phase].mean() for phase in range(3)])
    return result


@permacache(
    "frame_alignment_checks/phase_handedness/compute_self_agreement/cumulative_prob",
    key_function=dict(m=stable_hash_cached),
)
def cumulative_prob(m, cl_model):
    """
    Cumulative probability of the model scores. This is the probability that a score is less than or equal to the given
    score.

    Can be used to convert a score to a percentile.

    m: The model to use.
    path_data: The path to the data.
    cl_data: The context length of the data.
    cl_model: The context length of the model.
    """
    scores = np.log(compute_donor_scores(m, cl_model, None))
    histo = np.zeros(int((0 - histogram_min_value) // histogram_resolution) + 1)
    np.add.at(
        histo,
        ((scores - histogram_min_value) // histogram_resolution).astype(np.int32),
        1,
    )
    histo = histo / histo.sum()
    return np.cumsum(histo)


def quantile_scores(cumul, scores, offset=0):
    scores_idxs = ((scores - histogram_min_value) // histogram_resolution).astype(
        np.int32
    )
    scores = cumul[scores_idxs + offset]
    return scores
