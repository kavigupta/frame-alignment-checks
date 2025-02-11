from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from ..compute_stop_codons import is_stop, sequence_to_codons


class RepairStrategy(ABC):
    """
    Represents a strategy for repairing an exonic sequence (i.e., removing stops)
    """

    @abstractmethod
    def repair(self, sequence: np.ndarray) -> Tuple[np.ndarray, object]:
        """
        Repair an exonic sequence (i.e., remove stops somehow). Return metadata
        """


class NoRepair(RepairStrategy):
    """
    Does not repair the sequence, just returns it as is. This is useful as
    a baseline.
    """

    def repair(self, sequence: np.ndarray) -> Tuple[np.ndarray, object]:
        return sequence, None


class RemoveStopCodons(RepairStrategy):
    """
    Removes stop codons by setting the first T -> C. This is done because
    it's the cleanest way to remove stop codons without creating an alternate
    frame stop codon.

    This returns the repaired sequence and the number of stop codons removed.
    """

    def __init__(self, phase_wrt_start: int):
        self.phase_wrt_start = phase_wrt_start

    def manipulate_stop_codons(self, codons: np.ndarray) -> np.ndarray:
        """
        This manipulates the stop codons in the given codons. This sets the first
        T to C.
        """
        stop_mask = is_stop(codons)
        codons[stop_mask, 0] = 1  # T -> C
        return codons, stop_mask.sum()

    def repair(self, sequence: np.ndarray) -> Tuple[np.ndarray, object]:
        codons = sequence_to_codons(sequence, self.phase_wrt_start)
        codons, meta = self.manipulate_stop_codons(codons)
        codon_sequence = codons.reshape(-1)
        relevant_slice = slice(
            self.phase_wrt_start, self.phase_wrt_start + len(codon_sequence)
        )
        sequence[relevant_slice] = np.eye(4)[codon_sequence]
        return sequence, meta


class RemoveStopCodonsAtoT(RemoveStopCodons):
    """
    Removes stop codons by setting

        TAG -> TTG
        TGA -> TGT
        TAA -> TAT
    """

    old_to_new_str = [
        ("TAG", "TTG"),
        ("TGA", "TGT"),
        ("TAA", "TAT"),
    ]

    old_to_new = [
        [["ACGT".index(nt) for nt in codon] for codon in codons]
        for codons in old_to_new_str
    ]

    def manipulate_stop_codons(self, codons: np.ndarray) -> np.ndarray:
        count_original = is_stop(codons).sum()
        count = 0
        for old, new in self.old_to_new:
            mask = np.all(codons == old, axis=1)
            count += mask.sum()
            codons[mask] = new
        assert count == count_original
        assert is_stop(codons).sum() == 0
        return codons, count


def repair_strategy_types():
    return dict(
        NoRepair=NoRepair,
        RemoveStopCodons=RemoveStopCodons,
        RemoveStopCodonsAtoT=RemoveStopCodonsAtoT,
    )
