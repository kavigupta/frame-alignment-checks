import matplotlib.pyplot as plt
import numpy as np

from ..bootstrap import bootstrap
from ..utils import all_3mers, boostrap_series, draw_bases


def plot_stop_codon_acc_delta_per_codon(acc_delta, mask, *, ax=None, color_fn):
    """
    Plot the accuracy drop per codon, with the option to plot ranges of stop codons
    and codons that are close to stop codons.

    :param acc_delta: The accuracy drop per codon.
    :param mask: The mask of codons to be used
    """
    drawn, series = compute_series(acc_delta, mask)
    series_mean = series.mean(0)
    lo, hi = boostrap_series(series)
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
            color=color_fn(phase + 1),
            marker="_",
        )
        ax.errorbar(
            xs,
            mean_errorbar[phase + 1],
            yerr=errorbar[phase + 1],
            color=color_fn(phase + 1),
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


def plot_stop_codon_acc_delta_summary(
    acc_delta, mask, distance_out, codon_masks, *, color_fn, **kwargs
):
    """
    Plot the summary of the accuracy drop per codon.
    """
    plt.figure(dpi=400, tight_layout=True, **kwargs)
    centers_for_models = []
    centers_for_phases = []
    for i, k in enumerate(acc_delta):
        center_for_model = i * 3.5
        centers_for_models.append(center_for_model)
        for phase in (-1, 0, 1):
            center_for_phase = center_for_model + phase
            centers_for_phases.append(center_for_phase)
            means_summary = {}
            for idx, (codon_mask, label, _) in enumerate(codon_masks):
                arr = acc_delta[k][:, :, :, phase + 1, codon_mask]
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
                    color=color_fn(idx),
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

    ax_phases.set_xticks(centers_for_models, list(acc_delta))
    ax_models.set_xticks(centers_for_phases, ["−1", "0", "+1"] * len(acc_delta))
    # 3 columns, lower right, small font
    ax_models.legend(loc="lower right", fontsize="small")
    ax_models.set_ylabel(
        f"Drop in acc. when codon is\nplaced {distance_out}nt away from splice site [%]"
    )
    ax_models.grid(axis="y")


def plot_stop_codon_acc_delta_summary_as_image(acc_delta, no_undesired_changes):
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
    for k in acc_delta:
        names, values = compute_series(acc_delta[k], no_undesired_changes)
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
        np.array(xmids_all).flatten(), ["−1", "0", "+1"] * len(acc_delta)
    )
    ax_models.set_xticks(np.mean(xmids_all, 1), list(acc_delta))
    ax_models.tick_params(axis="x", top=False)

    plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap),
        label="Drop in acc. [%]",
        values=np.linspace(cmin, cmax, 100),
        # small width
        fraction=0.02,
    )
    # monospace font for yticks
    for tick in ax_models.get_yticklabels():
        tick.set_fontname("monospace")


def compute_codon_locations(gap):
    xs = np.arange(16)[:, None] * (4 + gap) + np.arange(4)
    line_pos = xs[1:, 0] - (1 + gap) / 2
    xs = xs.flatten()
    return xs, line_pos


def compute_series(acc_delta, mask):
    drawn = draw_bases(all_3mers())
    reorder = np.argsort(drawn)
    drawn = np.array(drawn)[reorder]
    acc_delta = acc_delta[..., reorder]
    mask = mask[..., reorder]
    series = (acc_delta * mask).sum((1, 2)) / mask.sum((0, 1))
    return drawn, series
