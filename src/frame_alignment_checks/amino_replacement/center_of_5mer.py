import itertools
from collections import defaultdict
from functools import lru_cache
from typing import List, Tuple

from ..coding_exon import CodingExon
from ..utils import draw_bases
from .amino_classification import amino_classification
from ..codon_table import codon_to_amino_acid


def same_functional_group(a, b):
    a = amino_classification[codon_to_amino_acid[a]]
    b = amino_classification[codon_to_amino_acid[b]]
    return a == b


@lru_cache(None)
def center_of_5mer_experiments():
    fivemers_to_mutations = defaultdict(list)
    for fivemer in list(itertools.product("ACGT", repeat=5)):
        fivemer = "".join(fivemer)
        for center_mutation in "ACGT":
            if fivemer[2] == center_mutation:
                continue
            post_mutation = fivemer[:2] + center_mutation + fivemer[3:]
            if same_functional_group(fivemer[1:4], post_mutation[1:4]):
                continue
            if not same_functional_group(fivemer[:3], post_mutation[:3]):
                continue
            if not same_functional_group(fivemer[2:], post_mutation[2:]):
                continue
            fivemers_to_mutations[fivemer].append(center_mutation)
    return {fivemer: mutations for fivemer, mutations in fivemers_to_mutations.items()}


def compute_relevant_mutations_for_exon(exon: CodingExon) -> List[Tuple[int, str]]:
    """
    Compute the relevant mutations for the given exon. A relevant mutation
    is (offset, kmer, mutation) where start_offset is the offset of the base
    of the 1nt mutation (relative to the start of the exon) and mutation is
    the new base to replace the old base with. kmer is the 5mer that the
    mutation is in the center of, provided for convenience.
    """
    k = 5
    co5mer = center_of_5mer_experiments()
    text_str = draw_bases(exon.text)
    mutations = []
    for start_off in range(len(text_str) - k):
        kmer = text_str[start_off : start_off + k]
        if kmer not in co5mer:
            continue
        for mutation in co5mer[kmer]:
            mutations.append((start_off + 2, kmer, mutation))
    return mutations
