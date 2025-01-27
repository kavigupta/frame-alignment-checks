import numpy as np

from .deletion import (
    ModelForDeletion,
    accuracy_given_deletion_experiment,
)


def num_stops_by_phase(distance_out):
    """
    Compute the number of stops in each phase category.

    Outputs a matrix num_stops: (3, N, 9, 2) where it can be indexed as
    num_stops[phase_wrt_start, which_exon, num_deletions - 1, A/D].

    What phase_wrt_start is is the phase of the stop codon relative to the
    exon. E.g., if an exon is CCTAGCTGACTAA... then the TAG is counted in phase 2,
    the TGAC in phase 1, and the TAA in phase 0. Note that this is *not* the phase
    of the codon with respect to the reading frame!!

    :param distance_out: The distance out to compute the stops.

    :return: num_stops
    """
    num_stops = [
        accuracy_given_deletion_experiment(
            ModelForDeletion(None, 0),
            dict(type="RemoveStopCodons", phase_wrt_start=i),
            distance_out=distance_out,
        )[2][..., [1, 2]]
        for i in range(3)
    ]
    num_stops = np.array(num_stops)
    return num_stops
