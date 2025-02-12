from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from ..bootstrap import bootstrap
from ..compute_stop_codons import is_stop
from ..plotting.colors import bar_color, line_color
from ..utils import all_3mers, bootstrap_series, draw_bases
from .stop_codon_replacement import Replace3MerResult


def plot_by_codon(result: Replace3MerResult, mask: np.ndarray, *, ax=None):
    """
    Plot the accuracy drop per codon. This is a detailed plot that shows the accuracy drop
    for each codon, averaged across seeds. The plot is split by phase. The x-axis is the codon
    and the y-axis is the accuracy drop.

    :param result: The result of the experiment
    :param mask: You can pass ``result.no_undesired_changes_mask``, this is useful if you want to exclude
      some mutations that cause undesired changes.
    :param ax: The axis to plot on. If None, a new figure is created.
    """
    drawn, series = compute_series(result, mask)
    series_mean = series.mean(0)
    lo, hi = bootstrap_series(series)
    mean_errorbar = (hi + lo) / 2
    errorbar = (hi - lo) / 2
    gap = 0.5
    xs, line_pos = compute_codon_locations(gap)
    if ax is None:
        plt.figure(figsize=(12, 5))
        ax = plt.gca()
    # stdev of the mean estimate
    for phase in (-1, 0, 1):
        ax.scatter(
            xs,
            series_mean[phase + 1],
            label=f"Phase: {phase}",
            color=line_color(phase + 1),
            marker="_",
        )
        ax.errorbar(
            xs,
            mean_errorbar[phase + 1],
            yerr=errorbar[phase + 1],
            color=line_color(phase + 1),
            linestyle="None",
        )
    ax.set_xticks(xs, drawn, rotation=90)
    # set xticks to monospace font
    for tick in ax.get_xticklabels():
        tick.set_fontname("monospace")
    ax.set_ylabel("Accuracy drop [%]")
    ax.set_ylim(-30, 4)
    ax.axhline(0, color="black")
    ax.legend()
    for x in line_pos:
        ax.axvline(x, color="black", lw=0.5)
    ax.set_xlim(-gap * 2, xs[-1] + gap * 2)


def plot_effect_grouped(
    results: Dict[str, Replace3MerResult], mask: np.ndarray, distance_out: int, **kwargs
):
    """
    Plot the summary of the accuracy drop by codon, with all non-stop codons grouped together.
    This is a summary plot that shows the accuracy drop for each codon, averaged across seeds;
    and with all models placed on a single plot. The plot is split by model, phase, and then
    codon.

    :param acc_delta: The accuracy drop per codon. Second output of ```fac.replace_3mer.experiments```
    :param mask: The mask of experiment results to be used. First output of ```fac.replace_3mer.experiments``` or
      all ones if you want to include all results.
    :param distance_out: The distance out from the splice site.
    :param kwargs: Additional arguments to pass to plt.figure.
    """
    stops_mask = is_stop(all_3mers().argmax(-1))
    codon_masks = [
        (stops_mask == 0, "not stop", dict()),
        ((all_3mers().argmax(-1) == [3, 0, 2]).all(-1), "TAG", dict(hatch="+++")),
        ((all_3mers().argmax(-1) == [3, 0, 0]).all(-1), "TAA", dict(hatch="ooo")),
        ((all_3mers().argmax(-1) == [3, 2, 0]).all(-1), "TGA", dict(hatch="///")),
    ]
    plt.figure(dpi=400, tight_layout=True, **kwargs)
    centers_for_models = []
    centers_for_phases = []
    for i, k in enumerate(results):
        center_for_model = i * 3.5
        centers_for_models.append(center_for_model)
        for phase in (-1, 0, 1):
            center_for_phase = center_for_model + phase
            centers_for_phases.append(center_for_phase)
            means_summary = {}
            for idx, (codon_mask, label, _) in enumerate(codon_masks):
                arr = results[k].acc_delta[:, :, :, phase + 1, codon_mask]
                arr_mask = mask[:, :, phase + 1, codon_mask]
                mean_by_model = (arr * arr_mask).sum((1, 2, 3)) / arr_mask.sum()
                # stdev of the mean estimate
                width = 0.7
                x = (
                    center_for_phase
                    + (idx - (len(codon_masks) - 1) / 2) / len(codon_masks) * width
                )
                means_summary[label] = mean_by_model.mean()
                plt.bar(
                    [x],
                    mean_by_model.mean(),
                    color=bar_color(idx),
                    width=1 / len(codon_masks) * width,
                    label=label if i == 0 and phase == 0 else None,
                )
                lo, hi = bootstrap(mean_by_model)
                mean_of_error = (hi + lo) / 2
                error = (hi - lo) / 2
                plt.errorbar(x, mean_of_error, error, color="black")
            if phase == 0:
                tag, taa, tga = [
                    means_summary[label] for label in ["TAG", "TAA", "TGA"]
                ]
                ratio = tga / ((tag + taa) / 2)
                print(f"Model: {k}, TGA[0]/mean(TAG[0], TAA[0]) = {ratio:.2%}")
    ax_models = plt.gca()
    # Second X-axis
    ax_phases = ax_models.twiny()
    ax_phases.spines["top"].set_position(("axes", 1.13))

    # put both X-axes on the top
    ax_models.xaxis.set_ticks_position("top")
    ax_phases.xaxis.set_label_position("top")

    # ensure that ax2 has the same scale as ax1
    ax_phases.set_xlim(ax_models.get_xlim())

    ax_phases.set_xticks(centers_for_models, list(results))
    ax_models.set_xticks(centers_for_phases, ["−1", "0", "+1"] * len(results))
    # 3 columns, lower right, small font
    ax_models.legend(loc="lower right", fontsize="small")
    ax_models.set_ylabel(
        f"Drop in acc. when codon is\nplaced {distance_out}nt away from splice site [%]"
    )
    ax_models.grid(axis="y")


def plot_by_codon_table(
    results: Dict[str, Replace3MerResult], no_undesired_changes: np.ndarray
):
    """
    Plots the same information as plot_by_codon, but in a table format.

    :param results: The accuracy drop per codon. Second output of ```fac.replace_3mer.experiments```.
    :param no_undesired_changes: The mask of experiment results to be used.
      First output of ```fac.replace_3mer.experiments```.
    """
    # pylint: disable=no-member
    cmap = plt.cm.viridis
    cmin, cmax = -30, 4
    overall_gap = 0.25
    x_width, y_gap = 3, 0.25
    x_start = 0
    ys, _ = compute_codon_locations(y_gap)
    ys = ys * (1 + overall_gap)
    xmids_all = []
    x = x_start
    for k in results:
        names, values = compute_series(results[k], no_undesired_changes)
        values = values.mean(0).T
        for y, vs in zip(ys, values):
            xmids = []
            for x, v in zip(np.arange(len(vs)) * (x_width + overall_gap) + x_start, vs):
                color = cmap((v - cmin) / (cmax - cmin))
                plt.fill_between([x, x + x_width], [y, y], [y + 1, y + 1], color=color)
                hsv_value = plt.cm.colors.rgb_to_hsv(color[:3])[-1]
                # text in the middle of the bar
                v_str = f"{v:.1f}"
                while len(v_str) < 5:
                    v_str = " " + v_str
                plt.text(
                    x + x_width / 2,
                    y + 0.5,
                    v_str,
                    ha="center",
                    va="center",
                    fontfamily="monospace",
                    color="black" if hsv_value > 0.75 else "white",
                )
                xmids.append(x + x_width / 2)
        xmids_all.append(xmids)
        x_start = x + x_width + 1
    plt.ylim(ys[-1] + 1 + 2 * y_gap, ys[0] - 2 * y_gap)
    plt.xlim(-1, x_start)
    plt.yticks(ys + 0.5, names)
    # two axes, one for the models and one for the phases
    ax_models = plt.gca()
    ax_phases = ax_models.twiny()
    ax_models.spines["top"].set_position(("axes", 1.02))
    # turn off line for ax_models
    ax_models.spines["top"].set_visible(False)
    # turn off tickmarks for ax_models
    ax_models.xaxis.set_ticks_position("top")
    ax_phases.xaxis.set_label_position("top")

    ax_phases.set_xlim(ax_models.get_xlim())
    ax_phases.set_xticks(
        np.array(xmids_all).flatten(), ["−1", "0", "+1"] * len(results)
    )
    ax_models.set_xticks(np.mean(xmids_all, 1), list(results))
    ax_models.tick_params(axis="x", top=False)

    plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap),
        label="Drop in acc. [%]",
        values=np.linspace(cmin, cmax, 100),
        # small width
        fraction=0.02,
        ax=ax_phases,
    )
    # monospace font for yticks
    for tick in ax_models.get_yticklabels():
        tick.set_fontname("monospace")


def compute_codon_locations(gap):
    xs = np.arange(16)[:, None] * (4 + gap) + np.arange(4)
    line_pos = xs[1:, 0] - (1 + gap) / 2
    xs = xs.flatten()
    return xs, line_pos


def compute_series(result, mask):
    acc_delta = result.acc_delta
    drawn = draw_bases(all_3mers())
    reorder = np.argsort(drawn)
    drawn = np.array(drawn)[reorder]
    acc_delta = acc_delta[..., reorder]
    mask = mask[..., reorder]
    series = (acc_delta * mask).sum((1, 2)) / mask.sum((0, 1))
    return drawn, series
