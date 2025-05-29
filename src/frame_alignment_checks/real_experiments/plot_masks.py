import itertools

import numpy as np

from ..plotting.colors import line_color
from .experiment_results import FullRealExperimentResult


def compute_binned_predictor(actual, predicted):
    """
    Compute a binned predictor for the given actual and predicted values. Computes
    the average and standard deviation of the predicted values in each bin.
    """
    bins = np.linspace(actual.min(), actual.max(), 25)
    count_by_bin = np.zeros(bins.shape[0])
    sum_by_bin = np.zeros(bins.shape[0])
    sum_sq_by_bin = np.zeros(bins.shape[0])
    bin_idxs = (
        ((actual - actual.min()) / (actual.max() - actual.min()) * (bins.shape[0] - 1))
        .round()
        .astype(np.int64)
    )
    np.add.at(count_by_bin, bin_idxs, 1)
    np.add.at(sum_by_bin, bin_idxs, predicted)
    np.add.at(sum_sq_by_bin, bin_idxs, predicted**2)
    avg_by_bin = np.zeros(bins.shape[0]) + np.nan
    avg_by_bin[count_by_bin > 0] = (
        sum_by_bin[count_by_bin > 0] / count_by_bin[count_by_bin > 0]
    )
    std_by_bin = np.zeros(bins.shape[0]) + np.nan
    std_by_bin[count_by_bin > 0] = np.sqrt(
        np.clip(
            sum_sq_by_bin[count_by_bin > 0] / count_by_bin[count_by_bin > 0]
            - avg_by_bin[count_by_bin > 0] ** 2,
            0,
            None,
        )
    ) / np.sqrt(count_by_bin[count_by_bin > 0])

    expected = avg_by_bin[bin_idxs]

    return bins, avg_by_bin, std_by_bin, expected


def plot_for_masks(
    ax, title, xlabel, result, masks, mean_decrease_prob_by_mask, *, color_for_idx
):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    predicted = result.predicteds[0]
    bins, avg_by_bin, std_by_bin, _ = compute_binned_predictor(result.actual, predicted)
    for idx, (mask, label), mean_decrease_prob_this in zip(
        itertools.count(), masks, mean_decrease_prob_by_mask
    ):
        ax.scatter(
            result.actual[mask],
            predicted[mask],
            alpha=min(1, 1 / mask.sum() ** 0.5 * 5),
            label=f"{label}; c.d.p.: {mean_decrease_prob_this:.2%}",
            marker=".",
            color=color_for_idx(idx),
        )
    ax.plot(bins, avg_by_bin, color="black", lw=0.5)
    ax.fill_between(
        bins,
        avg_by_bin - 2 * std_by_bin,
        avg_by_bin + 2 * std_by_bin,
        alpha=0.10,
        color="black",
        label=r"Expected $\log_2(\hat P(\mathrm{splice}))$; 95% CI",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\log_2(\hat P(\mathrm{splice}))$")
    ax.set_xlim(*np.percentile(result.actual, [2, 98]))
    ax.set_ylim(*np.percentile(predicted, [2, 98]))
    ax.legend()
    ax.set_title(title)


def plot_raw_real_experiment_results(
    title,
    *,
    er_by_model: FullRealExperimentResult,
    xlabel,
    axs,
    k,
):
    """
    Plots the raw real experiment results for each model and mask. By "raw" we mean
    all the data points, rather than computing decrease probabilities.

    Each model is plotted on its own axis, as provided in ``axs``.

    We also provide a binned estimator of the expected prediction value given the actual PSI.
    We furthermore label each mask with the controlled mean percentile of the predicted PSI
    given th eactual PSI.

    :param title: The title of the plot.
    :param er_by_model: The experiment results by model.
    :param xlabel: The label for the x-axis (the actual values)
    :param axs: The axes to plot on.
    """
    assert len(axs.flatten()) == len(er_by_model.er_by_model)
    mean_decrease_probabilities = er_by_model.mean_decrease_probability_each(k=k)
    for ax, name in zip(axs.flatten(), er_by_model.er_by_model):
        plot_for_masks(
            ax,
            f"{title} - {name}",
            xlabel,
            er_by_model.er_by_model[name],
            er_by_model.masks_each,
            mean_decrease_prob_by_mask=mean_decrease_probabilities[name][0],
            color_for_idx=line_color,
        )
