from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import tqdm.auto as tqdm
from dconstruct import construct
from permacache import permacache, stable_hash
from run_batched import run_batched

from ..coding_exon import CodingExon
from ..load_data import load_long_canonical_internal_coding_exons, load_validation_gene
from ..models import ModelToAnalyze
from ..utils import collect_windows, device_of, extract_center, stable_hash_cached
from .deletion_repair import repair_strategy_types

mutation_locations = [
    "left of A",
    "right of A",
    "left of D",
    "right of D",
]

affected_splice_sites = ["PD", "A", "D", "NA"]


@dataclass
class DeletionAccuracyDeltaResult:
    """
    Contains the raw output of the deletion experiment, as well as several
    methods for summarization.

    :param raw_data: The raw data from the deletion experiment.
        Called as
        raw_data[seed, exon_id, deletion - 1, deletion_location, affected_splice_site]
    """

    raw_data: np.ndarray

    @classmethod
    def concatenate(
        cls, results: List["DeletionAccuracyDeltaResult"]
    ) -> "DeletionAccuracyDeltaResult":
        return cls(np.concatenate([r.raw_data for r in results], axis=0))

    def mean_effect_matrix(self, num_deletions: int) -> np.ndarray:
        """
        Returns a matrix representing the mean effect of ``num_deletions`` deletions at
        each location on each splice site.

        :param num_deletions: The number of deletions to consider.
        :return: The mean effect matrix. Shape (4, 4). The rows represent deletion
            locations, and the columns represent affected splice sites.
        """
        if not 1 <= num_deletions <= self.raw_data.shape[2]:
            raise ValueError(
                f"num_deletions should be between 1 and {self.raw_data.shape[2]}, inclusive"
            )
        return self.raw_data[:, :, num_deletions - 1].mean((0, 1))

    def mean_effect_series(
        self, mutation_location: str, affected_splice_site: str, mean=True
    ) -> np.ndarray:
        """
        Returns the mean effect of deletions at the given location on the given splice site.

        :param mutation_location: The location of the deletion. One of
            ``fac.deletion.mutation_locations``.
        :param affected_splice_site: The affected splice site. One of
            ``fac.deletion.affected_splice_sites``.
        :param mean: Whether to take the mean across all exons.
        :return: The mean effect by deletion location. This is not aggregated over
            deletions or seeds. Shape: ``(num_seeds, num_deletions)``. If ``mean`` is
            False, the shape is ``(num_seeds, num_exons, num_deletions)``.
        """
        deletion_location_idx = mutation_locations.index(mutation_location)
        affected_splice_site_idx = affected_splice_sites.index(affected_splice_site)
        result = self.raw_data[:, :, :, deletion_location_idx, affected_splice_site_idx]
        if not mean:
            return result
        return result.mean(1)

    def mean_effect_masked(
        self,
        mask=None,
        mutation_locations_to_use: Tuple[str] = ("right of A", "left of D"),
        affected_splice_sites_to_use: Tuple[str] = ("A", "D"),
    ) -> np.ndarray:
        """
        Compute the mean effect of deletions on the given deletion locations and splice sites, with a mask.

        :param mask: A mask to apply to the data. If provided, the mask should be
            of the shape mask[exon_id, num_deletions - 1, deletion_location out of deletion_locations].
        :param mutation_locations_to_use: The deletion locations to consider. Each must be one of
            ``fac.deletion.mutation_locations``.
        :param affected_splice_sites_to_use: The affected splice sites to consider. Each must be one of
            ``fac.deletion.affected_splice_sites``.
        :return: The mean effect of deletions at the given locations on the given sites.
            Shape ``(num_seeds, num_deletions)``. Only averaged over sites where the mask is on.
        """
        mask_shape = (
            self.raw_data.shape[1],
            self.raw_data.shape[2],
            len(mutation_locations_to_use),
        )
        if mask is None:
            mask = np.ones(mask_shape)
        assert mask.shape == mask_shape
        selected_data = np.stack(
            [
                np.mean(
                    [
                        self.mean_effect_series(
                            deletion_location, affected_splice_site, mean=False
                        )
                        for affected_splice_site in affected_splice_sites_to_use
                    ],
                    axis=0,
                )
                for deletion_location in mutation_locations_to_use
            ],
            axis=-1,
        )
        # selecetd_data is of shape (num_seeds, num_exons, num_deletions, num_deletion_locations)
        assert selected_data.shape[1:] == mask.shape
        # numerator aggregates over exon_id and deletion_location
        numer = (selected_data * mask).sum((1, 3))
        # denominator aggregates over the same
        denom = mask.sum((0, 2))
        frac = numer.copy()
        frac[:, denom > 0] /= denom[denom > 0]
        frac[:, denom == 0] = np.nan
        return frac


def accuracy_delta_given_deletion_experiment_for_multiple_series(
    mods: Dict[str, List[ModelToAnalyze]],
    *,
    repair_spec=dict(type="NoRepair"),
    distance_out,
    binary_metric=True,
):
    """
    A wrapper around ``fac.deletion.experiment`` that takes a dictionary of
    model series and returns a dictionary of results. The keys of the input dictionary are used as the keys
    of the output dictionary.

    See ``fac.deletion.experiment`` for more details.
    """
    return {
        name: _accuracy_delta_given_deletion_experiment_for_series(
            mod,
            repair_spec=repair_spec,
            distance_out=distance_out,
            binary_metric=binary_metric,
        )
        for name, mod in mods.items()
    }


def _accuracy_delta_given_deletion_experiment_for_series(
    mod: List[ModelToAnalyze],
    *,
    repair_spec=dict(type="NoRepair"),
    distance_out,
    binary_metric=True,
    mod_for_base=None,
):
    return DeletionAccuracyDeltaResult.concatenate(
        [
            accuracy_delta_given_deletion_experiment(
                m,
                repair_spec=repair_spec,
                distance_out=distance_out,
                binary_metric=binary_metric,
                mod_for_base=mod_for_base,
            )
            for m in mod
        ]
    )


def accuracy_delta_given_deletion_experiment(
    mod: ModelToAnalyze,
    *,
    repair_spec=dict(type="NoRepair"),
    distance_out,
    binary_metric=True,
    mod_for_base=None,
    limit=None,
) -> DeletionAccuracyDeltaResult:
    """
    Accuracy delta given deletion experiment. This function runs a deletion experiment on the given model
    and returns the accuracy delta for each deletion.

    :param mod: The model to run the deletion experiment on.
    :param repair_spec: The repair strategy to use.
    :param distance_out: The distance out to use for deletions.
    :param binary_metric: Whether to use a binary metric for the predictions.

    :returns: The accuracy delta for each deletion, as a ``DeletionAccuracyDeltaResult``.
    """
    assert mod.model is not None
    if mod_for_base is None:
        mod_for_base = mod
    _, yps_deletions, _ = accuracy_given_deletion_experiment(
        mod, repair_spec, distance_out=distance_out, limit=limit
    )
    yps_base, _, _ = accuracy_given_deletion_experiment(
        mod_for_base, repair_spec, distance_out=distance_out, limit=limit
    )
    if binary_metric:
        thresh_dada = mod.thresholds[[1, 0, 1, 0]]
        thresh_base_dada = mod_for_base.thresholds[[1, 0, 1, 0]]
        yps_deletions = (yps_deletions > thresh_dada).astype(np.float64)
        yps_base = (yps_base > thresh_base_dada).astype(np.float64)
    delta = yps_deletions - yps_base[:, None, None, :]
    return DeletionAccuracyDeltaResult(delta[None])


def accuracy_given_deletion_experiment(
    model_for_deletion, repair_strategy_spec, *, limit=None, **kwargs
):
    return basic_deletion_experiment_multi(
        load_long_canonical_internal_coding_exons()[:limit],
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
            # literally, there's **kwargs right there!
            # pylint: disable=missing-kwoa
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
        yps = run_batched(
            lambda x: extract_center(model, x), x_windows, 128, device=device_of(model)
        )
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
