import numpy as np
import pickle
import pkg_resources

from ..coding_exon import CodingExon


def load_validation_gene(idx):
    path = pkg_resources.resource_filename(
        "frame_alignment_checks", "data/relevant_validation_genes.npz"
    )
    with np.load(path) as data:
        return data[f"x{idx}"], data[f"y{idx}"]


def load_long_canonical_internal_coding_exons():
    path = pkg_resources.resource_filename(
        "frame_alignment_checks", "data/long_canonical_internal_coding_exons.pkl"
    )
    with open(path, "rb") as f:
        return [CodingExon(**d) for d in pickle.load(f)]


def load_minigene(gene, exon):
    path = pkg_resources.resource_filename(
        "frame_alignment_checks", f"data/minigene_{gene}_{exon}.pkl"
    )
    with open(path, "rb") as f:
        return pickle.load(f)
