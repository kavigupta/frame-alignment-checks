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
    pval = plot_multi_seed_experiment(
        {k: 100 * np.array(v)[:, 1] for k, v in summaries.items()},
        "Controlled mean percentile of closed frames",
        ax,
        name_remapping=name_remapping,
        **kwargs,
    )
    ax.set_title(title)
    display_permutation_test_p_values(pval, "")
