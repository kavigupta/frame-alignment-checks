import gzip
import pickle

import numpy as np
import pandas as pd
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


def load_saturation_mutagenesis_table():
    path = pkg_resources.resource_filename(
        "frame_alignment_checks",
        "data/saturation_mutagenesis_Supplemental_Table_S2.xlsx",
    )
    return pd.read_excel(path)


def load_train_counts_by_phase():
    path = pkg_resources.resource_filename(
        "frame_alignment_checks", "data/train_handedness_counts.npz"
    )
    with np.load(path) as data:
        return data["arr_0"]


def load_non_stop_donor_windows():
    path = pkg_resources.resource_filename(
        "frame_alignment_checks", "data/phase_handedness_test_set.pkl.gz"
    )
    with gzip.open(path, "rb") as f:
        return pickle.load(f)
