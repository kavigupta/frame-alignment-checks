import gzip
import pickle
from importlib.resources import as_file, files
from typing import Tuple

import numpy as np
import pandas as pd

import frame_alignment_checks

from ..coding_exon import CodingExon


def load_validation_gene(idx) -> Tuple[np.ndarray, np.ndarray]:
    with as_file(
        files(frame_alignment_checks.data).joinpath("relevant_validation_genes.npz")
    ) as path:
        with np.load(path) as data:
            return data[f"x{idx}"], data[f"y{idx}"]


def load_long_canonical_internal_coding_exons():
    source = files(frame_alignment_checks.data).joinpath(
        "long_canonical_internal_coding_exons.pkl"
    )
    with as_file(source) as path:
        with open(path, "rb") as f:
            return [CodingExon(**d) for d in pickle.load(f)]


def load_minigene(gene, exon):
    with as_file(
        files(frame_alignment_checks.data).joinpath(f"minigene_{gene}_{exon}.pkl")
    ) as path:
        with open(path, "rb") as f:
            return pickle.load(f)


def load_saturation_mutagenesis_table():
    with as_file(
        files(frame_alignment_checks.data).joinpath(
            "saturation_mutagenesis_Supplemental_Table_S2.xlsx"
        )
    ) as path:
        return pd.read_excel(path)


def load_train_counts_by_phase() -> np.ndarray:
    with as_file(
        files(frame_alignment_checks.data).joinpath("train_handedness_counts.npz")
    ) as path:
        with np.load(path) as data:
            return data["arr_0"]


def load_non_stop_donor_windows():
    with as_file(
        files(frame_alignment_checks.data).joinpath("phase_handedness_test_set.pkl.gz")
    ) as path:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
