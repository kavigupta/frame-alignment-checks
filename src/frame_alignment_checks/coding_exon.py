from dataclasses import dataclass
from typing import Tuple


@dataclass
class CodingExon:
    """
    Represents a coding exon in a gene. Can be constitutive or alternative.

    To load these, see ```load_long_canonical_internal_coding_exons```.

    :param gene_idx: Index of the gene in the validation set
    :param acceptor: Position of the acceptor site. Start of the exon
    :param donor: Position of the donor site. Note: not the same as the "end" of the exon since that's exclusive
    :param prev_donor: Position of the previous donor site
    :param next_acceptor: Position of the next acceptor site
    :param phase_start: The phase of the start of the exon, 0, 1, or 2. This is relative to the coding frame, i.e.,
        a phase of 1 indicates that the exon starts with a 2mer followed by a codon.
    """

    gene_idx: int  # Index of the gene in the validation set
    acceptor: int  # Position of the acceptor site
    donor: int  # Position of the donor site. Note: not the same as the "end" of the exon since that's exclusive
    prev_donor: int
    next_acceptor: int
    phase_start: int  # The phase of the start of the exon, 0, 1, or 2

    @property
    def text(self):
        """
        Compute the text of the exon. This is the sequence of the exon in the gene.

        :return: np.ndarray. The sequence of the exon. Will be of shape (L, 4) where L is the length of the exon.
        """
        # pylint: disable=cyclic-import,no-member
        from .load_data import load_validation_gene

        x, _ = load_validation_gene(self.gene_idx)
        return x.argmax(-1)[self.acceptor : self.donor + 1]

    @property
    def all_locations(self) -> Tuple[int, int, int, int]:
        """
        Returns a tuple of the locations of the exon. This is a tuple of the
        (prev_donor, acceptor, donor, next_acceptor) positions.

        :return: Tuple[int, int, int, int]. The locations of the exon.
        """
        return self.prev_donor, self.acceptor, self.donor, self.next_acceptor

    @property
    def length(self) -> int:
        """
        Return the length of the exon. This is the number of bases in the exon.

        :return: int. The length of the exon.
        """
        return self.donor - self.acceptor + 1

    def to_dict(self):
        """
        Convert the exon to a dictionary. This is useful for saving the exon to a file.

        Invariant: ```CodingExon(**exon.to_dict()) == exon```

        :return: Dict[str, Any]. The dictionary representation of the exon.
        """
        return {
            "gene_idx": self.gene_idx,
            "acceptor": self.acceptor,
            "donor": self.donor,
            "prev_donor": self.prev_donor,
            "next_acceptor": self.next_acceptor,
            "phase_start": self.phase_start,
        }
