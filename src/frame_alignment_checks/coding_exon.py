from dataclasses import dataclass


@dataclass
class CodingExon:
    """
    Represents a coding exon in a gene. Can be constitutive or alternative.
    """

    gene_idx: int  # Index of the gene in the validation set
    acceptor: int  # Position of the acceptor site
    donor: int  # Position of the donor site. Note: not the same as the "end" of the exon since that's exclusive
    prev_donor: int
    next_acceptor: int
    phase_start: int  # The phase of the start of the exon, 0, 1, or 2

    @property
    def text(self):
        # pylint: disable=cyclic-import,no-member
        from .data.load import load_validation_gene

        x, _ = load_validation_gene(self.gene_idx)
        return x.argmax(-1)[self.acceptor : self.donor + 1]

    @property
    def all_locations(self):
        return self.prev_donor, self.acceptor, self.donor, self.next_acceptor

    @property
    def length(self):
        return self.donor - self.acceptor + 1

    def to_dict(self):
        return {
            "gene_idx": self.gene_idx,
            "acceptor": self.acceptor,
            "donor": self.donor,
            "prev_donor": self.prev_donor,
            "next_acceptor": self.next_acceptor,
            "phase_start": self.phase_start,
        }
