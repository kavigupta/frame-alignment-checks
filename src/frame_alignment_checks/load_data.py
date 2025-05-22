import gzip
import pickle
from functools import lru_cache
from importlib.resources import as_file, files
from typing import Tuple

import numpy as np
import pandas as pd

from . import data
from .coding_exon import CodingExon


def load_validation_gene(idx) -> Tuple[np.ndarray, np.ndarray]:
    with as_file(files(data).joinpath("relevant_validation_genes.npz")) as path:
        with np.load(path) as d:
            return d[f"x{idx}"], d[f"y{idx}"]


def load_canonical_internal_coding_exons():
    source = files(data).joinpath("canonical_internal_coding_exons.pkl")
    with as_file(source) as path:
        with open(path, "rb") as f:
            return [CodingExon(**d) for d in pickle.load(f)]


@lru_cache(None)
def load_long_canonical_internal_coding_exons():
    return [
        e for e in load_canonical_internal_coding_exons() if e.donor - e.acceptor > 100
    ]


def load_minigene(gene, exon):
    with as_file(files(data).joinpath(f"minigene_{gene}_{exon}.pkl")) as path:
        with open(path, "rb") as f:
            return pickle.load(f)


def load_saturation_mutagenesis_table():
    with as_file(
        files(data).joinpath("saturation_mutagenesis_Supplemental_Table_S2.xlsx")
    ) as path:
        return pd.read_excel(path)


def load_train_counts_by_phase() -> np.ndarray:
    with as_file(files(data).joinpath("train_handedness_counts.npz")) as path:
        with np.load(path) as d:
            return d["arr_0"]


def load_non_stop_donor_windows():
    with as_file(files(data).joinpath("phase_handedness_test_set.pkl.gz")) as path:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)


@lru_cache(None)
def load_poison_exon_data():
    with as_file(files(data).joinpath("poison_exon_genes.pkl.gz")) as path:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)


def load_poison_exon_sequence(gene_spec, acc, don, model_cl):
    pe_data = load_poison_exon_data()
    remove = (pe_data["cl_max"] - model_cl) // 2
    text = pe_data["gene_sequences"][gene_spec, acc, don]
    text = text[remove : text.shape[0] - remove]
    return text


def load_nve_descriptors():
    with as_file(files(data).joinpath("nve_descriptors.pkl")) as path:
        with open(path, "rb") as f:
            return pickle.load(f)
