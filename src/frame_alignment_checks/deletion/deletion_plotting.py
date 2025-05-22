from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from ..plotting.colors import bar_color, line_color
from ..utils import bootstrap_series
from .delete import (
    DeletionAccuracyDeltaResult,
    affected_splice_sites,
    mutation_locations,
)
from .deletion_num_stops import num_open_reading_frames


def plot_by_deletion_loc_and_affected_site(
    deltas_by_model: Dict[str, DeletionAccuracyDeltaResult], distance_out: int
):
    """
    Plot the effect of deletions on the accuracy of the model. This is plotted
    for 3'SS and 5'SS separately, for all 4 deletion locations, and for all
    deletion lengths.

    :param deltas_by_model: The deltas by model.
    :param distance_out: The distance out.
    """
    _, axs = plt.subplots(
        1,
        len(deltas_by_model),
        figsize=(2.5 * len(deltas_by_model), 4),
        sharey=True,
        dpi=400,
    )

    for name, ax in zip(deltas_by_model, axs):
        delta = deltas_by_model[name]
        # delta = 100 * delta.mean(1)
        # delta = delta[:, :, :, [1, 2]]  # only the A and D
        xs = 1 + np.arange(9)
        is_in_exon = []
        for i, dl in enumerate(mutation_locations):
            for j, loc in enumerate(["3'SS", "5'SS"]):
                is_in_exon.append(
                    {
                        "u.s. of 3'SS": 0,
                        "d.s. of 3'SS": 1,
                        "u.s. of 5'SS": 1,
                        "d.s. of 5'SS": 0,
                    }[dl]
                )
                ys = 100 * delta.mean_effect_series(dl, loc)
                ax.plot(
                    xs,
                    ys.mean(0),
                    label=f"{loc}; del. {dl}",
                    color=line_color(i),
                    marker=".",
                    linestyle=["-", "--"][j],
                )
                lo, hi = bootstrap_series(ys)
                ax.fill_between(xs, lo, hi, alpha=0.5, color=bar_color(i))
        ax.axhline(0, color="black")
        ax.set_xticks([3, 6, 9])
        ax.set_title(name)
        ax.grid()
        ax.set_xlabel("Deletion length")
    axs[0].set_ylabel("Drop in accuracy when deleting")
    setup_legend(axs[-1], is_in_exon)
    plt.suptitle(f"Starting at {distance_out}")


def setup_legend(ax, is_in_exon):
    handles, labels = ax.get_legend_handles_labels()
    ordering = np.argsort(
        is_in_exon + np.arange(len(is_in_exon)) / len(is_in_exon) * 0.5
    )
    ax.legend(
        [handles[i] for i in ordering],
        [labels[i] for i in ordering],
        ncol=2,
        loc="lower right",
    )


def plot_exon_effects_by_orf(
    deltas_by_model: Dict[str, DeletionAccuracyDeltaResult],
    distance_out: int,
    *,
    axs=None,
):
    """
    Plot the effect of deletions on the accuracy of the model. This is plotted
    for 3'SS and 5'SS combined into a single effect, for both the exonic deletions
    (all 4 combinations, averaged).

    :param deltas_by_model: The deltas by model.
    :param distance_out: The distance out.
    :param axs: The axes to plot on.
    """
    num_frames_open = num_open_reading_frames(
        distance_out, limit=list(deltas_by_model.values())[0].num_exons
    )
    if axs is None:
        _, axs = plt.subplots(
            1,
            len(deltas_by_model),
            figsize=(2.5 * len(deltas_by_model), 4),
            sharey=True,
            dpi=400,
        )
    for name, ax in zip(deltas_by_model, axs):
        delta = deltas_by_model[name]
        conditions = {
            "all": np.ones_like(num_frames_open, dtype=bool),
            "at least one r.f. open": num_frames_open > 0,
            "all r.f.'s closed": num_frames_open == 0,
        }
        for color_idx, condition in enumerate(conditions):
            mask = conditions[condition]
            xs = np.arange(9) + 1
            frac = 100 * delta.mean_effect_masked(mask)
            ax.plot(
                xs,
                frac.mean(0),
                color=line_color(color_idx),
                marker="*",
                label=condition,
            )
            lo, hi = bootstrap_series(frac)
            ax.fill_between(xs, lo, hi, alpha=0.5, color=bar_color(color_idx))
        ax.axhline(0, color="black")
        ax.set_xticks([3, 6, 9])
        ax.set_title(name)
        ax.grid()
    axs[-1].legend()
    axs[0].set_ylabel("Drop in accuracy when deleting")


def plot_matrix_at_site(
    deltas: Dict[str, DeletionAccuracyDeltaResult],
    distance_out: int,
    num_deletions: int,
    height=4,
):
    """
    Plot a matrix of effects for each model. This is a 4x4 matrix where the rows are
    the deletions in each region (u.s. of 3'SS, d.s. of 3'SS, u.s. of 5'SS, d.s. of 5'SS) and
    the columns are the affected splice sites (P5'SS, 3'SS, 5'SS, N3'SS).

    The values are the drop in accuracy when deleting the given region and splice site.

    The values are in percentage points.

    :param deltas: The deltas by model.
    :param distance_out: The distance out.
    :param num_deletions: The number of deletions.
    :param height: The height of the figure, in inches.
    """
    _, axs = plt.subplots(
        1,
        len(deltas),
        figsize=(height * 0.8 * len(deltas), height),
        sharey=True,
        tight_layout=True,
    )

    delta_matr = {
        name: delta.mean_effect_matrix(num_deletions) for name, delta in deltas.items()
    }
    min_clim = -0.1
    for name, ax in zip(deltas, axs):
        im = ax.imshow(delta_matr[name] * 100, vmin=min_clim * 100, vmax=0)
        # text in each box
        for i in range(4):
            for j in range(4):
                limits = np.min(delta_matr[name]), np.max(delta_matr[name])
                value_relative_to_limits = (delta_matr[name][i, j] - limits[0]) / (
                    limits[1] - limits[0]
                )
                ax.text(
                    j,
                    i,
                    f"{delta_matr[name][i, j] * 100:.1f}",
                    ha="center",
                    va="center",
                    color="black" if value_relative_to_limits > 0.5 else "white",
                )
        ax.set_xticks(np.arange(4), affected_splice_sites)
        ax.set_yticks(np.arange(4), mutation_locations)
        ax.set_title(name)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(f"Starting at {distance_out}; {num_deletions} deletions")
