import matplotlib.pyplot as plt
import numpy as np

from ..deletion.delete import affected_splice_sites, mutation_locations
from ..deletion.deletion_num_stops import num_open_reading_frames
from ..utils import bootstrap_series
from .colors import bar_color, line_color


def plot_deletion_effect_matrix(deltas, distance_out, num_deletions):
    """
    Plot a matrix of effects for each model. This is a 4x4 matrix where the rows are
    the deletions in each region (left of A, right of A, left of D, right of D) and
    the columns are the affected splice sites (PD, A, D, NA).
    """
    _, axs = plt.subplots(
        1,
        len(deltas),
        figsize=(3.25 * len(deltas), 4),
        sharey=True,
    )

    delta_matr = {
        name: delta.mean_effect_matrix(num_deletions) for name, delta in deltas.items()
    }
    for name, ax in zip(deltas, axs):
        im = ax.imshow(delta_matr[name] * 100)
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
        plt.colorbar(im, ax=ax)
    plt.suptitle(f"Starting at {distance_out}; {num_deletions} deletions")


def plot_deletion_effect(deltas_by_model, distance_out):
    """
    Plot the deletions for all models.
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
        for i, dl in enumerate(mutation_locations):
            for j, loc in enumerate("AD"):
                ys = 100 * delta.mean_effect_series(dl, loc)
                ax.plot(
                    xs,
                    ys.mean(0),
                    label=f"{loc}; deleted {dl}",
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
    axs[-1].legend()
    plt.suptitle(f"Starting at {distance_out}")


def plot_deletion_effect_by_whether_stop_codon(
    deltas_by_model,
    distance_out,
    *,
    axs=None,
):
    num_frames_open = num_open_reading_frames(distance_out)
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
            "at least one phase open": num_frames_open > 0,
            "all phases closed": num_frames_open == 0,
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
