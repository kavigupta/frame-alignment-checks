import numpy as np
from frozendict import frozendict

from ..plotting.multi_seed_experiment import plot_multi_seed_experiment
from ..utils import display_permutation_test_p_values


def plot_real_experiment_summary(
    ax,
    summaries,
    title,
    *,
    name_remapping=frozendict(),
    **kwargs,
):
    """
    Plot a summary of each experiment, showing the mean percentile of closed frames.
    Also, print the p-values of the permutation tests.

    :param ax: the axis to plot on.
    :param summaries: a dictionary mapping model names to lists of tuples. Each tuple contains the mean
        percentile of closed frames and the mean percentile of closed frames for the control.
    :param title: the title of the plot.
    :param name_remapping: a dictionary mapping model names to new names.
    """
    pval = plot_multi_seed_experiment(
        {k: 100 * np.array(v)[:, 1] for k, v in summaries.items()},
        "Controlled mean percentile of closed frames",
        ax,
        name_remapping=name_remapping,
        **kwargs,
    )
    ax.set_title(title)
    display_permutation_test_p_values(pval, "")
