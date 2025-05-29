from functools import lru_cache
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from ..load_data import load_nve_descriptors
from ..plotting.colors import bar_color, line_color
from ..real_experiments import plot_summary
from ..real_experiments.math import mean_decrease_probability


@lru_cache(None)
def load_ef5():
    return np.array([nve["ef_5"] for nve in load_nve_descriptors()])


@lru_cache(None)
def load_all_closed():
    return np.array([nve["all_closed"] for nve in load_nve_descriptors()])


def scatterplot(ax, results, title):
    ef_5 = load_ef5()[: len(results)]
    all_closed = load_all_closed()[: len(results)]

    def plot_w_corr(x, y, color, label):
        (m, b), *_ = np.linalg.lstsq(np.array([x, np.ones_like(x)]).T, y, rcond=None)
        ax.scatter(
            x,
            y,
            label=f"{label}: r={np.corrcoef(x, y)[0,1]:.3f}",
            marker=".",
            alpha=0.25,
            color=color,
        )
        xmin, xmax = x.min(), x.max()
        xrange = xmax - xmin
        xmin -= xrange / 10
        xmax += xrange / 10
        xline = np.array([xmin, xmax])
        ax.plot(xline, m * xline + b, color=color)

    plot_w_corr(
        ef_5[~all_closed],
        results[~all_closed],
        line_color(0),
        "Open",
    )
    plot_w_corr(
        ef_5[all_closed],
        results[all_closed],
        line_color(1),
        "Closed",
    )
    ax.set_xlim(ef_5.min(), ef_5.max())
    ax.set_ylabel("$\\log_{10}$(predicted exon probability)")
    ax.set_xlabel("EF 5%")
    ax.set_title(title)


def poison_exon_scatterplots(results: Dict[str, np.ndarray]):
    """
    Plot scatterplots of the results of the poison exon analysis. Will plot
    the results of the analysis for each model in the results dictionary, and
    provide line fits to the data, as well as the correlation coefficient, for
    the open and closed exons separately.

    :param results: A dictionary of results, where the keys are the names of the
        models and the values are the results of the poison exon analysis.
    """
    _, axs = plt.subplots(
        1, len(results), figsize=(3 * len(results), 4), tight_layout=True, dpi=200
    )
    for ax, k in zip(axs, results):
        scatterplot(ax, results[k][0], k)
        ax.legend()


def mean_decrease_probability_pe(results, *, k):
    return mean_decrease_probability(
        load_ef5(),
        np.array(results),
        np.array(
            [
                # ~np.array(table.all_closed),
                # np.array(table.all_closed),
                ~load_all_closed(),
                load_all_closed(),
            ]
        ),
        k=k,
    )


def mean_decrease_probability_pe_each(results: Dict[str, np.ndarray], *, k):
    return {
        m: np.array([mean_decrease_probability_pe(r, k=k) for r in results[m]])
        for m in results
    }


def poison_exons_summary_plot(results: Dict[str, np.ndarray], ax=None, *, k, **kwargs):
    """
    Plot the summary of the poison exon analysis. This will plot the mean
    decrease probability for each model in the results dictionary, and
    provide a bar plot of the results.

    :param results: A dictionary of results, where the keys are the names of the
        models and the values are the results of the poison exon analysis.
    """
    if ax is None:
        plt.figure(dpi=400, tight_layout=True, figsize=(6, 4))
        ax = plt.gca()
    summary = mean_decrease_probability_pe_each(results, k=k)
    style_kwargs = dict(
        line_style=lambda i: dict(color=line_color(i)),
        bar_style=lambda i: dict(color=bar_color(i), alpha=0.5),
    )
    style_kwargs.update(kwargs)
    plot_summary(ax, summary, "", **style_kwargs)
