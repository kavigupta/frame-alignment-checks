import numpy as np

from ..load_data import load_long_canonical_internal_coding_exons
from ..models import ModelToAnalyze
from .delete import accuracy_given_deletion_experiment


def get_phases():
    """
    Get the phases of each exon, i.e., the phase of the start codon
    relative to the reading frame.
    """
    return np.array(
        [x.phase_start for x in load_long_canonical_internal_coding_exons()]
    )


def num_stops_by_phase(distance_out, *, limit=None):
    """
    Compute the number of stops in each phase category.

    Outputs a matrix num_stops: (3, N, 9, 2) where it can be indexed as
    num_stops[phase_wrt_start, which_exon, num_deletions - 1, A/D].

    What phase_wrt_start is is the phase of the stop codon relative to the
    exon. E.g., if an exon is CCTAGCTGACTAA... then the TAG is counted in phase 2,
    the TGAC in phase 1, and the TAA in phase 0. Note that this is *not* the phase
    of the codon with respect to the reading frame!!

    :param distance_out: The distance out to compute the stops.
    :param limit: The limit to use for the experiment.

    :return: num_stops
    """
    num_stops = [
        accuracy_given_deletion_experiment(
            ModelToAnalyze(None, 0, 0, 0),
            dict(type="RemoveStopCodons", phase_wrt_start=i),
            distance_out=distance_out,
            limit=limit,
        )[2][..., [1, 2]]
        for i in range(3)
    ]
    num_stops = np.array(num_stops)
    return num_stops


def phase_to_pull_each_start():
    """
    Returns an array of size (N, 9) where each entry is the phase to pull
    from num_stops_by_phase to get the reading frame consistent with
    the start of the exon.

    E.g.,
        [CCT][AGC][CGC][AAA]... -> [CCT][AGC][CGC][AAA]...
    """
    phases = get_phases()
    # phase = 1 means that the exon is in phase 1 relative to the reading frame, so the reading frame
    # is in phase 2 relative to the exon
    phases = (-phases) % 3
    return phases[:, None].repeat(9, axis=1)


def phase_to_pull_each_end():
    """
    Returns an array of size (N, 9) where each entry is the phase to pull
    from num_stops_by_phase to get the reading frame consistent with
    the end of the exon.

    Technique: to get the phase to align with the end, subtract the deletion amount. E.g., if
    the exon is in phase 0 at the start (meaning the reading frame is in phase 0),
    and we delete 1 nucleotide, the frame we want to pull to be consistent with the
    reading frame is phase 2. E.g.,
        [CCT][AGC][CGC][AAA]... -> CC[TAG][CGC][AAA]...
    the TAG is in frame, and is in phase 2 relative to the exon. So phase 2 is now the
    reading frame, rather than phase 0.
    """
    phases = (phase_to_pull_each_start() - np.arange(1, 1 + 9)) % 3
    return phases


def phase_to_pull_from_each():
    """
    Returns an array of shape (N, 9, 2) where each entry is the phase to pull
    from num_stops_by_phase to get the reading frame consistent with
    the most nucleotides.

    We use the end phase for A and the start phase for D, because the A deletion
    disrupts near the start (so we want to align with the end) and vice versa.
    """
    return np.stack([phase_to_pull_each_end(), phase_to_pull_each_start()], axis=-1)


def num_in_frame_stops(distance_out):
    """
    Returns an array of shape (N, 9, 2), where each entry is the number of in-frame
    stops for the exon, given the distance out. The "frame" is defined for deletions
    as the reading frame of the larger unedited chunk, i.e., for A deletions, the
    reading frame is the one consistent with the end of the exon, and for D deletions,
    the reading frame is the one consistent with the start of the exon.

    :param distance_out: The distance out to compute the stops.

    :return: num_in_frame_stops
    """
    num_stops = num_stops_by_phase(distance_out)
    one_hot_phase = np.eye(3, dtype=int)[phase_to_pull_from_each()]
    return (one_hot_phase.transpose(3, 0, 1, 2) * num_stops).sum(0)


def num_open_reading_frames(distance_out, *, limit=None):
    return (num_stops_by_phase(distance_out, limit=limit) == 0).sum(0)
