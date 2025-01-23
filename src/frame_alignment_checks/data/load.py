import numpy as np
import pkg_resources


def load_validation_gene(idx):
    path = pkg_resources.resource_filename(
        "frame_alignment_checks", "data/relevant_validation_genes.npz"
    )
    with np.load(path) as data:
        return data[f"x{idx}"], data[f"y{idx}"]
