from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
from permacache import permacache, stable_hash
import torch
import tqdm.auto as tqdm
from run_batched import run_batched

from .coding_exon import CodingExon
from .construct import construct
from .data.load import load_long_canonical_internal_coding_exons, load_validation_gene
from .deletion_repair import repair_strategy_types
from .utils import collect_windows, extract_center, stable_hash_cached


@dataclass
class ModelForDeletion:
    model: torch.nn.Module
    model_cl: int
    thresholds: np.ndarray


def accuracy_delta_given_deletion_experiment(
    mod,
    repair_spec,
    distance_out,
    binary_metric,
    mod_for_base,
):
    yps_base_orig, yps_deletions, _ = accuracy_given_deletion_experiment(
        mod, repair_spec, distance_out=distance_out
    )
    if mod_for_base.model is not None:
        yps_base, _, _ = accuracy_given_deletion_experiment(
            mod_for_base, repair_spec, distance_out=distance_out
        )
    else:
        yps_base = yps_base_orig
    if binary_metric and mod.model is not None:
        thresh_dada = mod.thresholds[[1, 0, 1, 0]]
        thresh_base_dada = (
            mod_for_base.thresholds
            if mod_for_base.model is not None
            else mod.thresholds
        )[[1, 0, 1, 0]]
        yps_deletions = (yps_deletions > thresh_dada).astype(np.float64)
        yps_base = (yps_base > thresh_base_dada).astype(np.float64)
    delta = yps_deletions - yps_base[:, None, None, :]
    return delta


def accuracy_given_deletion_experiment(
    model_for_deletion, repair_strategy_spec, **kwargs
):
    return basic_deletion_experiment_multi(
        load_long_canonical_internal_coding_exons(),
        model_for_deletion.model,
        model_for_deletion.model_cl,
        repair_strategy_spec,
        **kwargs,
    )


@permacache(
    "modular_splicing/frame_alignment/deletion_experiment_multi_2",
    key_function=dict(
        exons=lambda x: stable_hash([e.__dict__ for e in x]), model=stable_hash_cached
    ),
)
def basic_deletion_experiment_multi(
    exons, model, model_cl, repair_strategy_spec, **kwargs
):
    """
    Runs a basic deletion experiment on multiple exons.
    """
    res_base, res_del, metas = zip(
        *[
            basic_deletion_experiment(
                e,
                model,
                model_cl,
                repair_strategy_spec,
                **kwargs,
            )
            for e in tqdm.tqdm(exons)
        ]
    )
    if deletion_experiment.shelf.shelf:
        deletion_experiment.shelf.shelf.sync()
    return np.array(res_base), np.array(res_del), np.array(metas)


def basic_deletion_experiment(
    ex, model, model_cl, repair_strategy_spec, *, distance_out, delete_up_to=9
):
    """
    Run a basic deletion experiment on the given exon. Deletes
        - A - distance_out to A - distance_out - delete_up_to (incl)
        - A + distance_out to A + distance_out + delete_up_to (incl)
        - D - distance_out to D - distance_out - delete_up_to (incl)
        - D + distance_out to D + distance_out + delete_up_to (incl)

    :returns: yps_base, yps_deletions
        yps_base: The predictions for the exon and flanking bases without deletions; shape (4,)
        yps_deletions: The predictions for the deletions; shape (delete_up_to, 4, 4)
    """
    assert (
        distance_out + delete_up_to
    ) * 2 < ex.donor - ex.acceptor, (
        f"This deletion experiment {distance_out} is too large for the exon {ex}"
    )
    deletion_ranges_incl = []
    for delete in range(1, delete_up_to + 1):
        deletion_ranges_incl.extend(
            [
                (ex.acceptor - distance_out - delete, ex.acceptor - distance_out - 1),
                (ex.acceptor + distance_out + 1, ex.acceptor + distance_out + delete),
                (ex.donor - distance_out - delete, ex.donor - distance_out - 1),
                (ex.donor + distance_out + 1, ex.donor + distance_out + delete),
            ]
        )
    deletion_ranges_half_excl = [
        (start, end + 1) for start, end in deletion_ranges_incl
    ]
    yps, metas = deletion_experiment(
        ex, model, model_cl, deletion_ranges_half_excl, repair_strategy_spec
    )
    yps_base, yps_deletions = yps[0], yps[1:]
    yps_deletions = yps_deletions.reshape(delete_up_to, 4, 4)
    metas = np.array(metas).reshape(delete_up_to, 4)
    return yps_base, yps_deletions, metas


@permacache(
    "modular_splicing/frame_alignment/deletion_experiment_2",
    key_function=dict(ex=lambda x: x.__dict__, model=stable_hash_cached),
)
def deletion_experiment(
    ex: CodingExon,
    model,
    model_cl,
    deletion_ranges: List[Tuple[int, int]],
    repair_strategy_spec,
) -> Tuple[np.ndarray, List[object]]:
    """
    Perform a deletion experiment on the given exon. Deletes the given ranges and returns the predictions.

    :param ex: The exon to perform the deletion experiment on.
    :param model: The model to use for predictions.
    :param model_cl: The context length of the model.
    :param deletion_ranges: The deletion ranges to use. Length N.
    :param repair_strategy_spec: The repair strategy to use, as a specification.

    :returns:
        yps: The predictions for the deletions; shape (N, 4). These are probabilities (real not log)
            for the previous donor, acceptor, donor, and next acceptor.
        metas: The metadata for each deletion; shape (N,). This is the metadata returned by the repair strategy.
    """
    x, _ = load_validation_gene(ex.gene_idx)
    repair_strategy = construct(repair_strategy_types(), repair_strategy_spec)
    repair = repair_strategy.repair
    locs = ex.all_locations
    x_windows = [collect_windows(x, locs, model_cl)]
    metas = []
    for deletion_range in deletion_ranges:
        seq_mut, meta, locs_mut = perform_deletion(x, deletion_range, locs, repair)
        x_windows.append(collect_windows(seq_mut, locs_mut, model_cl))
        metas.append(meta)
    x_windows = np.concatenate(x_windows)
    if model is not None:
        yps = run_batched(lambda x: extract_center(model, x), x_windows, 128)
    else:
        yps = np.empty((len(x_windows), 4))
        yps[:] = np.nan
    yps = yps.reshape(-1, 4, yps.shape[-1])
    yps = yps[:, [0, 1, 2, 3], [2, 1, 2, 1]]
    return yps, metas


def perform_deletion(
    sequence: np.ndarray,
    deletion_range: Tuple[int, int],
    indices: Tuple[int, int, int, int],
    repair: Callable[[np.ndarray], Tuple[np.ndarray, object]],
) -> Tuple[np.ndarray, object, Tuple[int, int, int, int]]:
    """
    Actually perform the deletion on the given sequence. This deletes the given range and repairs the sequence.

    :param sequence: The sequence to perform the deletion on.
    :param deletion_range: The range to delete.
    :param indices: The indices of the exon in the sequence.
    :param repair: The repair function to use.

    :returns:
        sequence: The sequence with the deletion and repair performed.
        meta: The metadata returned by the repair function.
        indices: The indices of the exon in the repaired sequence.
    """
    delete_start, delete_end = deletion_range
    delete_length = delete_end - delete_start
    assert not any(
        delete_start <= i < delete_end for i in indices
    ), "should not delete a boundary"
    assert sorted(indices) == list(indices), "indices should be sorted"
    sequence = np.concatenate([sequence[:delete_start], sequence[delete_end:]], axis=0)
    indices = tuple(i if i < delete_start else i - delete_length for i in indices)
    acc, don = indices[1], indices[2]
    repaired_exon, meta = repair(sequence[acc : don + 1])
    repair_delta = len(repaired_exon) - (don - acc + 1)
    indices = (
        indices[0],
        indices[1],
        indices[2] + repair_delta,
        indices[3] + repair_delta,
    )
    sequence = np.concatenate(
        [sequence[:acc], repaired_exon, sequence[don + 1 :]],
        axis=0,
    )
    return sequence, meta, indices
