import json
import numpy as np
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
        "frame_alignment_checks", "data/long_canonical_internal_coding_exons.json"
    )
    with open(path) as f:
        return [CodingExon(**d) for d in json.load(f)]
