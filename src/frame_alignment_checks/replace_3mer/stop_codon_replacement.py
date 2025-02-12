from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import tqdm.auto as tqdm
from permacache import permacache, stable_hash
from run_batched import run_batched

from ..load_data import load_long_canonical_internal_coding_exons, load_validation_gene
from ..models import ModelToAnalyze
from ..utils import (
    all_3mers,
    collect_windows,
    device_of,
    extract_center,
    stable_hash_cached,
)
from .stop_codon_replacement_no_undesired_changes import no_undesired_changes_mask


@dataclass
class Replace3MerResult:
    """
    Contains the results of a replace 3mer experiment.

    :param acc_delta: The delta in accuracy for all exons. Shape ```(num_seeds, num_exons, 2, 3, 64)```
       ```acc_delta[seed_idx, batch_idx, distance_from_which, phase, codon]``` is the delta in accuracy,
       in percentage points, when you replace the codon at distance_out from the acceptor or donor
       (A if ```distance_from_which == 0```, D if ```distance_from_which == 1```) with the codon at index codon
    :param no_undesired_changes_mask: Whether or not undesired changes might be caused by the
        substitution. Shape ```(num_exons, 2, 3, 64)```.
    """

    acc_delta: np.ndarray
    no_undesired_changes_mask: np.ndarray

    @classmethod
    def merge(cls, results: List["Replace3MerResult"]) -> "Replace3MerResult":
        nucs = np.array([r.no_undesired_changes_mask for r in results])
        assert (nucs == nucs[0]).all()
        return cls(
            acc_delta=np.concatenate([r.acc_delta for r in results], axis=0),
            no_undesired_changes_mask=nucs[0],
        )


@permacache(
    "frame_alignment_checks/replace_3mer/stop_codon_replacement_delta_accuracy_for_multiple_series",
    key_function=dict(models=stable_hash),
)
def stop_codon_replacement_delta_accuracy_for_multiple_series(
    models: Dict[str, List[ModelToAnalyze]], distance_out, limit=None
) -> Dict[str, Replace3MerResult]:
    """
    A wrapper around ``fac.replace_3mer.experiment`` that takes a dictionary of
    model series and returns a dictionary of results. The keys of the input dictionary are used as the keys
    of the output dictionary.

    See ``fac.replace_3mer.experiment`` for more details.
    """
    results = {}
    for name in models:
        results[name] = _stop_codon_replacement_delta_accuracy_for_series(
            models[name], name=name, distance_out=distance_out, limit=limit
        )
    nuc_masks = np.array([r.no_undesired_changes_mask for r in results.values()])
    assert (nuc_masks == nuc_masks[0]).all()
    return nuc_masks[0], results


def _stop_codon_replacement_delta_accuracy_for_series(
    ms: List[ModelToAnalyze], *, name=None, distance_out, limit=None
) -> Replace3MerResult:
    return Replace3MerResult.merge(
        [
            stop_codon_replacement_delta_accuracy(
                model_for_analysis=m, distance_out=distance_out, limit=limit
            )
            for m in tqdm.tqdm(ms, desc=name)
        ]
    )


def stop_codon_replacement_delta_accuracy(
    *, model_for_analysis: ModelToAnalyze, distance_out, limit=None
) -> Replace3MerResult:
    """
    Compute the delta in accuracy when replacing codons at all 3 phases with all 64 possible codons.

    :param model_for_analysis: The model to compute the delta in accuracy for
    :param distance_out: The distance from the acceptor and donor sites to mutate the codons at
    :param limit: The number of exons to run the experiment on. If None, run on all exons

    :returns: the results of the experiment
    """
    original_seqs_all, yps_base, yps_mut = mutate_codons_experiment_all(
        model=model_for_analysis.model,
        model_cl=model_for_analysis.model_cl,
        distance_out=distance_out,
        limit=limit,
    )
    yps_base, yps_mut = [
        (x > model_for_analysis.thresholds).astype(np.float64)
        for x in (yps_base, yps_mut)
    ]
    yps_mut_near_exon = yps_mut[:, [0, 1], :, :, [0, 1]]
    delta = 100 * (yps_mut_near_exon.transpose(1, 0, 2, 3) - yps_base[:, :, None, None])
    return Replace3MerResult(delta[None], no_undesired_changes_mask(original_seqs_all))


def with_all_codons(original_seq, codon_start_loc):
    """
    Takes an original sequence and returns a batched sequence with
    all possible codons at codon_start_loc.

    :param original_seq: the original sequence (L, 4)
    :param codon_start_loc: the location of the start of the codon

    Returns: (64, L, 4)
    """
    seqs = np.repeat(original_seq[None], 64, axis=0)
    seqs[:, codon_start_loc : codon_start_loc + 3] = all_3mers().astype(seqs.dtype)
    return seqs


@permacache(
    "modular_splicing/frame_alignment/codon_stop_replacement/mutated_codons_experiment",
    key_function=dict(
        model=stable_hash_cached,
        ex=lambda ex: sorted(ex.__dict__.items()),
    ),
)
def mutated_codons_experiment(*, model, model_cl, ex, target_codon_start):
    """
    Run an experiment to see how mutating "codons" (3mers) that are in and out of frame
    with respect to the exon affects the model's predictions.

    :param model: The model to use
    :param model_cl: The context length of the model
    :param x: The sequence to run the experiment on. Should be of shape (L, 4)
    :param ex: The exon to run the experiment on, as a CodingExon
    :param target_codon_start: The location of the start of the codon to mutate

    Returns: (original_seq, original_pred, mutated_preds)

    original_seq: The original 9mer sequence where the targeted codon is in the middle
    original_pred: The model's prediction for the original sequence. Shape (2,) (acceptor, donor)
    mutated_preds: The model's predictions for the mutated sequences. Shape (3, 64, 2)
        mutated_preds[offset + 1, c] is the model's prediction for when you replace the 3mer
        at target_codon_start + offset with the 3mer at index c in all_3mers().
        I.e., mutated_preds[1] is a map from codon to the result of mutating the targeted
        codon with that codon, and mutated_preds[0] and mutated_preds[2] are the predictions
        for the off-frame codon mutations of phases -1 and +1. (aka 2 and 1).
    """
    # this is subscriptable. Not sure why pylint thinks it isn't
    # pylint: disable=unsubscriptable-object
    x, _ = load_validation_gene(ex.gene_idx)
    target_codon_start += (-(target_codon_start - (ex.acceptor - ex.phase_start))) % 3
    original_seq = x[target_codon_start - 3 : target_codon_start + 6].argmax(-1)
    acc, don = ex.acceptor, ex.donor

    x, acc, don, target_codon_start = clip_for_efficiency(
        model_cl, ex, target_codon_start, x, acc, don
    )

    with_mutated_codons = [
        with_all_codons(x, loc)
        for loc in [target_codon_start - 1, target_codon_start, target_codon_start + 1]
    ]
    wmc_windows_flat = np.array(
        [
            collect_windows(wmc, [acc, don], model_cl)
            for wmcs in [[x], *with_mutated_codons]
            for wmc in wmcs
        ]
    )
    inital_shape = wmc_windows_flat.shape[:2]
    assert wmc_windows_flat.shape[2:] == (model_cl + 1, 4)
    wmc_windows_flat = wmc_windows_flat.reshape((-1, model_cl + 1, 4))

    out = run_batched(
        lambda x: extract_center(model, x),
        wmc_windows_flat,
        128,
        device=device_of(model),
    )
    out = out.reshape(inital_shape + (3,))

    # A and D
    out = out[:, [0, 1], [1, 2]]
    orig_pred = out[0]
    mut_preds = out[1:]
    mut_preds = mut_preds.reshape((3, 64, 2))
    return original_seq, orig_pred, mut_preds


def clip_for_efficiency(model_cl, ex, target_codon_start, x, acc, don):
    """
    Clips the sequence for efficiency. This is done by taking a window around the exon.

    Also offsets the acceptor, donor, and target_codon_start to be relative to the clipped sequence.
    """
    # pylint: disable=too-many-positional-arguments
    startloc, endloc = -10 + ex.acceptor - model_cl // 2, 10 + ex.donor + model_cl // 2
    startloc, endloc = max(startloc, 0), min(endloc, len(x))
    x = x[startloc:endloc]
    target_codon_start -= startloc
    acc, don = acc - startloc, don - startloc
    return x, acc, don, target_codon_start


def mutate_codons_experiment_all(*, model, model_cl, distance_out, limit):
    """
    See mutated_codons_experiment, which this wraps, for details.

    This function runs mutated_codons_experiment on all exons in exons.

    Returns: (orig_preds_all, mut_preds_all)

    orig_preds_all: The original predictions for all exons. Shape (N, 2)
    mut_preds_all: The mutated predictions for all exons. Shape (N, 2, 3, 64, 2)
        mut_preds_all[batch_idx, distance_from_which, ...] is the model's prediction
        for when you mutate exon at batch_idx at a distance `distance_out` from
        the acceptor or donor site (A if distance_from_which == 0, D if distance_from_which == 1)
    """
    original_seqs_all = []
    mut_preds_all = []
    orig_preds_all = []
    for ex in tqdm.tqdm(load_long_canonical_internal_coding_exons()[:limit]):
        assert ex.donor - ex.acceptor >= 2 * distance_out
        original_seq_acc, orig_pred_acc, mut_preds_acc = mutated_codons_experiment(
            model=model,
            model_cl=model_cl,
            ex=ex,
            target_codon_start=ex.acceptor + distance_out,
        )
        original_seq_don, orig_pred_don, mut_preds_don = mutated_codons_experiment(
            model=model,
            model_cl=model_cl,
            ex=ex,
            target_codon_start=ex.donor - distance_out,
        )
        original_seqs_all.append((original_seq_acc, original_seq_don))
        assert np.allclose(orig_pred_acc, orig_pred_don)
        orig_preds_all.append(orig_pred_acc)
        mut_preds_all.append((mut_preds_acc, mut_preds_don))
    original_seqs_all = np.array(original_seqs_all)
    orig_preds_all = np.array(orig_preds_all)
    mut_preds_all = np.array(mut_preds_all)
    return original_seqs_all, orig_preds_all, mut_preds_all
