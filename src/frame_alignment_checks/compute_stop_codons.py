import numpy as np


def sequence_to_codons(sequence, off=0):
    """
    Convert a sequence to codons. If the sequence is a one-hot encoding, it will be converted
    to integers [0, 1, 2, 3] by taking the argmax along the last axis. The sequence will be
    sliced from `off` to the end of the sequence, and then truncated to the nearest multiple of 3.

    :param sequence: np.ndarray. The sequence to convert to codons. Should be of shape (N, 4) or (N,).
    :param off: int. The offset to start the frame from. Things before this offset will be ignored.
    :return: np.ndarray. The sequence as codons. Will be of shape ((N - off) // 3, 3).
    """
    if off not in (0, 1, 2):
        raise ValueError("off must be 0, 1, or 2")
    if len(sequence.shape) > 1:
        assert len(sequence.shape) == 2
        assert sequence.shape[-1] == 4
        sequence = sequence.argmax(-1)
    sequence = sequence[off:]
    sequence = sequence[: len(sequence) - len(sequence) % 3]
    return sequence.reshape(-1, 3)


def is_stop(codons):
    """
    Compute whether a sequence of codons is a stop codon.

    :param codons: np.ndarray. The codons to check. Should be of shape (N, 3).

    :return: np.ndarray. A boolean array of shape (N,) where True indicates a stop codon.
    """
    TAG = [3, 0, 2]
    TAA = [3, 0, 0]
    TGA = [3, 2, 0]
    stops = TAG, TAA, TGA
    return np.any([(codons == np.array(stop)).all(-1) for stop in stops], axis=0)


def all_frames_closed(exon_sequences):
    """
    Compute whether all frames are closed for a set of exon sequences.

    :param exon_sequences: List[np.ndarray]. The exon sequences to check. Each sequence should be of shape (N, 4).
    :return: np.ndarray. A boolean array of shape (N,) where True indicates that all frames are closed.
    """
    return np.array(
        [
            np.array(
                [is_stop(sequence_to_codons(x, off)).any(-1) for x in exon_sequences]
            )
            for off in range(3)
        ]
    ).all(0)
