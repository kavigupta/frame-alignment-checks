import numpy as np

from ..compute_stop_codons import is_stop
from ..utils import all_3mers


def no_undesired_changes_mask(o_seq):
    """
    Compute a mask of whether a 3mer mutation induces a change to the stop codon landscape
    in any of the three phases that wasn't the one that was mutated.

    E.g., TAGCCC -> TAGTGA
    is a desired change because it adds a stop codon in phase 0, which was the mutation, but
    ATACCC -> ATAGCC is an undesired change because it adds a stop codon in phase 1, which
    is not the phase that was mutated.

    :param o_seq: The original sequence. Shape (N, 2, 9) where N is the number of exons,
        2 is the number of locations, and 9 is the length of the original sequence at
        the location.

    :return: A mask of shape (N, 2, 3, 64) where N is the number of exons, 2 is the number
        of locations, 3 is the number of phases, and 64 is the number of codons. The mask is True
        when the codon mutation does not introduce or remove any out of frame stop codons.
    """
    mutated_seqs = o_seq[..., None, None, :]
    mutated_seqs = np.repeat(mutated_seqs, 64, axis=-2)
    mutated_seqs = np.repeat(mutated_seqs, 3, axis=-3)
    for phase in (-1, 0, 1):
        mutated_seqs[..., phase + 1, :, 3 + phase : 6 + phase] = all_3mers().argmax(-1)
    # mutated_seqs_flat = mutated_seqs.reshape(-1, mutated_seqs.shape[-1])
    added_stop = (
        num_stop_codons_at_phases_batched(mutated_seqs)
        - num_stop_codons_at_phases_batched(o_seq)[..., None, None, :]
    )
    # ignore desired changes
    added_stop[:, :, [0, 1, 2], :, [0, 1, 2]] = 0

    no_undesired_changes = (added_stop == 0).all(-1)
    return no_undesired_changes


def num_stop_codons_at_phases_batched(x):
    """
    Compute the number of stop codons at each phase in a batched sequence.

    The sequence should be of shape (*batch, L). Returns a tensor of shape (*batch, 3),
        where index 0 is phase -1, index 1 is phase 0, and index 2 is phase 1.
    """
    *batch_axes, final_axis = x.shape
    x = x.reshape(-1, final_axis)
    result = np.array(
        [stop_codons_at_phase_batched(x, phase) for phase in (-1, 0, 1)]
    ).T
    return result.reshape(*batch_axes, 3)


def stop_codons_at_phase_batched(x, phase):
    """
    Compute the number of stop codons at a given phase in a batched sequence.

    The sequence should be of shape (batch, L).
    """
    start = phase % 3
    end = x.shape[1]
    end -= (end - start) % 3
    x = x[:, start:end]
    codons = x.reshape(x.shape[0], -1, 3)
    mask = is_stop(codons)
    mask = mask.sum(-1)
    return mask
