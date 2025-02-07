import numpy as np

from ..load_data import load_train_counts_by_phase


def get_phase_specific_9mers():
    """
    Computes the 5 most phase-specific 9-mers for each phase. This is done by the formula
        score(phase, x) = count(phase, x) * log((count(phase, x) + 1) / (total_count(x) - count(phase, x) + 1))
    """
    # pylint: disable=unsubscriptable-object,no-member
    counts_by_phase = load_train_counts_by_phase()
    result = {}
    for phase in range(3):
        score = counts_by_phase[phase] * np.log(
            (counts_by_phase[phase] + 1)
            / (counts_by_phase.sum(0) - counts_by_phase[phase] + 1)
        )
        idxs = np.argsort(score)[::-1][:5]
        result[phase] = idxs
    return result
