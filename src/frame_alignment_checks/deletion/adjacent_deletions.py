from typing import Dict, List, Tuple

import numpy as np
import tqdm.auto as tqdm
from matplotlib import pyplot as plt
from permacache import drop_if_equal, permacache
from run_batched import run_batched
from torch import nn

from frame_alignment_checks.coding_exon import CodingExon
from frame_alignment_checks.load_data import (
    load_canonical_internal_coding_exons,
    load_validation_gene,
)
from frame_alignment_checks.models import ModelToAnalyze
from frame_alignment_checks.plotting.colors import bar_color, line_color
from frame_alignment_checks.utils import bootstrap_series, device_of, stable_hash_cached

conditions: Tuple[int, int] = [
    (0, 0),
    (1, 0),
    (0, 2),
    (1, 2),
    (2, 0),
    (0, 1),
    (2, 1),
]


def perform_adjacent_deletions(
    first: CodingExon, second: CodingExon, context: int, *, outside: bool
):
    """
    Compute adjacent deletions on a pair of exons. Specifically, we
    delete from the first exon and the second exon.

    :param first: The first exon to delete from.
    :param second: The second exon to delete from. This must be the
        exon that follows the first exon.
    :param context: The number of nucleotides to keep around the exons.
    :param outside: If True, deletions are made on the outer edges of the exons.
        This only exists for checking the asymmetry results.

    :return: (x_mut, y_idxs_mut)
        x_mut: The mutated sequences. Numpy array of shape (C, n, 4), where
            C is the number of conditions, and n is the length of the sequence.
        y_idxs_mut: The mutated indices. Numpy array of shape (C, 4), where
            each row corresponds to a condition.
            The columns are (first.acceptor, first.donor, second.acceptor, second.donor).
    """
    assert context % 2 == 0
    assert second.prev_donor == first.donor
    x, _ = load_validation_gene(first.gene_idx)
    yidx_original = (first.acceptor, first.donor, second.acceptor, second.donor)
    x_mut, y_idxs = zip(
        *[
            multiple_deletions(
                x,
                yidx_original,
                [
                    (s, l)
                    for s, l in deletion_specifications(
                        first, second, off_first, off_second, outside=outside
                    )
                    if l > 0
                ],
            )
            for off_first, off_second in conditions
        ]
    )
    x_mut = np.array(x_mut)
    y_idxs = np.array(y_idxs)
    x_mut = np.pad(
        x_mut, ((0, 0), (context // 2, context // 2), (0, 0)), mode="constant"
    )
    y_idxs += context // 2
    start, end = y_idxs.min() - context // 2, y_idxs.max() + context // 2
    assert start >= 0
    assert end < x_mut.shape[1]
    x_mut = x_mut[:, start : end + 1]
    y_idxs -= start
    return x_mut, y_idxs


def deletion_specifications(first, second, off_first, off_second, *, outside):
    if outside:
        return [
            (first.acceptor + 10, off_first),
            (second.donor - 10 - off_second, off_second),
        ]
    return [
        (first.donor - 10 - off_first, off_first),
        (second.acceptor + 10, off_second),
    ]


def adjacent_coding_exons() -> List[Tuple[CodingExon, CodingExon]]:
    """
    All pairs of consecutive coding exons in the dataset we are
    using for evaluation. This is a list of tuples, where each tuple
    is a pair of CodingExon objects. The first element is the
    first exon, and the second element is the second exon. These are
    guaranteed to be consecutive in the sense that the first one's
    next_acceptor is the second one's acceptor.
    """
    cice = load_canonical_internal_coding_exons()
    consecutive_coding_exons = []
    for first, second in zip(cice, cice[1:]):
        if first.gene_idx != second.gene_idx:
            continue
        assert first.next_acceptor == second.acceptor
        consecutive_coding_exons.append((first, second))
    return consecutive_coding_exons


def close_consecutive_coding_exons() -> List[Tuple[CodingExon, CodingExon]]:
    """
    All pairs of consecutive coding exons in the dataset we are
    using for evaluation, with the additional constraint that
    the distance between the first exon and the second exon (the
    intron in between) is less than 1000 bp, and the total length
    of the two exons is less than 4000 bp.

    (The second constraint, given the first, is that the length
    of the two exons sums to less than 3000 bp, which is not
    particularly strict.)
    """
    return [
        (first, second)
        for first, second in adjacent_coding_exons()
        if second.acceptor - first.donor < 1000 and second.donor - first.acceptor < 4000
    ]


def run_on_all_adjacent_deletions(
    model: ModelToAnalyze, *, limit=None, outside=False
) -> np.ndarray:
    """
    Run the model on all adjacent deletions, producing a table of results.

    :param model: The model to run. This should be a ModelToAnalyze object.
    :param limit: The maximum number of pairs of adjacent deletions to run on.
        If None, run on all pairs. This is useful for debugging.
    :param outside: If True, deletions are made on the outer edges of the exons.
        This only exists for checking the asymmetry results.
    :return: An array of results, of shape (N, C, 4), where N is the number of
        pairs of adjacent deletions, and C is the number of conditions.
        The last dimension corresponds to (first.acceptor, first.donor,
        second.acceptor, second.donor). The array is of type boolean, indicating
        either above/below the classification threshold.
    """
    res = []
    for first, second in tqdm.tqdm(close_consecutive_coding_exons()[:limit], delay=1):
        res.append(
            run_on_adjacent_deletions(
                model.model,
                first,
                second,
                model_cl=model.model_cl,
                cl_model_clipped=model.cl_model_clipped,
                outside=outside,
            )
            > model.thresholds[[0, 1, 0, 1]]
        )
    return np.array(res)


def run_on_all_adjacent_deletions_for_multiple_series(
    mods: Dict[str, List[ModelToAnalyze]], outside=False
) -> Dict[str, np.ndarray]:
    """
    Like run_on_all_adjacent_deletions, but for multiple model series.
    """
    return {
        name: np.array(
            [
                run_on_all_adjacent_deletions(model, outside=outside)
                for model in tqdm.tqdm(ms, delay=1, desc=name)
            ]
        )
        for name, ms in mods.items()
    }


@permacache(
    "modular_splicing/frame_alignment/deletion_experiment_4",
    key_function=dict(model=stable_hash_cached, outside=drop_if_equal(False)),
)
def run_on_adjacent_deletions(
    model: nn.Module,
    first: CodingExon,
    second: CodingExon,
    *,
    cl_model_clipped,
    model_cl,
    outside=False,
):
    def run_model(inp):
        x, yi = inp["x"], inp["yi"]
        cond_idx = np.arange(yi.shape[0])[:, None]
        seq_idx = yi - cl_model_clipped // 2
        site_idx = np.array([1, 2, 1, 2])[None]
        res = model(x).softmax(-1)
        return res[cond_idx, seq_idx, site_idx]

    x_mut, y_idxs = perform_adjacent_deletions(first, second, model_cl, outside=outside)

    return run_batched(
        run_model,
        dict(x=x_mut.astype(np.float32), yi=y_idxs),
        32,
        device=device_of(model),
    )


def multiple_deletions(x, yidxs, deletions):
    """
    Compute multiple deletions on a sequence x, with corresponding yidxs.

    :param x: The sequence to delete from. Numpy array of shape (n, 4).
    :param yidxs: The corresponding yidxs to x. List of integers.
    :param deletions: List of tuples of integers, where the first element is the
        start index of the deletion, and the second element is the count of
        elements to delete.
    :return: Tuple of the new x and yidxs.
        The length of the new x will be the length of the original x, padded with
           zeros corresponding to the deletions.
        The yidxs will be updated to reflect the deletions; so
            out_x[out_yidxs[i]] == x[yidxs[i]].
    """
    x = list(x)
    yidxs = list(yidxs)
    cant_touch_past = float("inf")
    deletions = sorted(deletions, key=lambda x: -x[0])
    for start, count in deletions:
        assert start + count < cant_touch_past
        del x[start : start + count]
        # This is important because we are mutating the list in place, it would be
        # confusing to use enumerate here
        # pylint: disable=consider-using-enumerate
        for i in range(len(yidxs)):
            if yidxs[i] < start:
                continue
            assert yidxs[i] > start + count
            yidxs[i] -= count
        x += [[0] * 4] * count
        cant_touch_past = start
    return np.array(x), yidxs


def plot_adjacent_deletion_results(results: Dict[str, np.ndarray], h=3, w=3):
    """
    Plots the results of the adjacent deletion experiment, with each
    subplot corresponding to a different model series.

    :param results: A dictionary of results, output of
        run_on_all_adjacent_deletions_for_multiple_series.
    """
    _, axs = plt.subplots(
        1,
        len(results),
        figsize=(len(results) * w, h),
        sharey=True,
        dpi=400,
        tight_layout=True,
    )
    ax2 = None
    for ax, k in zip(axs, results):
        res = np.array(results[k]) * 100
        base = res[:, :, conditions.index((0, 0))]
        for i, condition in enumerate(conditions[1:]):
            for_this = res[:, :, conditions.index(condition)]
            delta = (for_this - base.astype(np.float32)).mean(1)
            mu = delta.mean(0)
            lo, hi = bootstrap_series(delta)
            ax.plot(
                mu,
                label=f"X={condition[0]};Y={condition[1]}",
                color=line_color(i % 3),
                linestyle=["-", "--"][i // 3],
            )
            ax.fill_between(
                np.arange(len(mu)), lo, hi, color=bar_color(i % 3), alpha=0.25
            )
        ax.set_xticks(range(4), ["3'", "5'", "3'", "5'"])
        ax2 = ax.twiny()
        ax2.xaxis.set_ticks_position("bottom")
        ax2.spines["bottom"].set_position(("outward", 15))
        ax2.set_xticks([0.5, 2.5], ["Exon 1", "Exon 2"])
        ax2.set_xlim(ax.get_xlim())
        ax2.spines["bottom"].set_color("none")
        ax.set_title(k)
    assert ax2 is not None
    axs[0].set_ylabel("Change in accuracy [%]")
    legend = axs[-1].legend(
        ncol=2,
        loc="lower right",
        facecolor="white",
        framealpha=1,
        edgecolor="black",
    )
    legend.remove()
    ax2.add_artist(legend)
