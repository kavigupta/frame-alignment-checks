import numpy as np
from frozendict import frozendict

from ..utils import permutation_test


def plot_multi_seed_experiment(
    results,
    ylabel,
    ax,
    *,
    name_remapping=frozendict(),
    line_style,
    bar_style,
):
    """
    Plot the swapping experiment results. The results should be a dictionary where the keys are the names
    of the models and the values are the results for each seed.

    :param results: The results of the experiment.
    :param ylabel: The label for the y axis.

    :returns:
        permutation_result: A dictionary where the keys are pairs of model names and the values are the
            p values of the permutation test comparing the two models.
    """
    width = 0.5
    for i, k in enumerate(results):
        ax.scatter([i] * len(results[k]), results[k], **line_style(i))
        lo, hi = np.percentile(
            np.random.RandomState(0)
            .choice(results[k], size=(len(results[k]), 100_000))
            .mean(0),
            [2.5, 97.5],
        )
        ax.fill_between(
            [i - width / 2, i + width / 2],
            [lo, lo],
            [hi, hi],
            **bar_style(i),
        )
    ax.set_ylabel(ylabel)
    ax.set_xticks(
        np.arange(len(results)),
        [name_remapping.get(k, k) for k in results],
        rotation=45,
    )
    ax.grid(axis="y")
    permutation_result = {}
    for k1 in results:
        for k2 in results:
            if k1 <= k2:
                continue
            permutation_result[(k1, k2)] = permutation_test(results[k1], results[k2])
    return permutation_result
