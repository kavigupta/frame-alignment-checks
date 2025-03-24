from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import tqdm.auto as tqdm
from permacache import permacache
from run_batched import run_batched

from frame_alignment_checks.load_data import load_long_canonical_internal_coding_exons
from frame_alignment_checks.models import ModelToAnalyze

from ..amino_replacement import amino_classification, codon_to_amino_acid
from ..coding_exon import CodingExon
from ..replace_3mer.stop_codon_replacement import extract_window_around_center
from ..utils import device_of, stable_hash_cached
from .center_of_5mer import compute_relevant_mutations_for_exon


def data_for_experimentation(ex: CodingExon, *, model_cl):
    x, acc, don, _ = extract_window_around_center(
        ex, loc=np.nan, model_cl=model_cl, pad_to_cl=True
    )
    x_orig = x.copy()
    relevant_mutations = compute_relevant_mutations_for_exon(ex)
    x = x[None].repeat(len(relevant_mutations), axis=0)
    positions, _, target = zip(*relevant_mutations)
    positions = acc + np.array(positions)
    target = np.array(["ACGT".index(nt) for nt in target])
    x[np.arange(len(positions)), positions] = np.eye(4, dtype=x.dtype)[target]
    return acc, don, x_orig, x


@dataclass
class AminoAcidExperimentResultOnExon:
    yps_base: np.ndarray
    yps_mut: np.ndarray
    mutations: List[Tuple[int, str, str]]
    exon: CodingExon

    @property
    def phases(self):
        mut_loc_rel_start = np.array([p for p, _, _ in self.mutations])
        return (mut_loc_rel_start - 1 + self.exon.phase_start) % 3

    def accuracy_drop(self, thresholds):
        return (self.yps_mut > thresholds).mean(-1) - (self.yps_base > thresholds).mean(
            -1
        )

    @property
    def source_amino_acids(self):
        return [codon_to_amino_acid[kmer[1:4]] for _, kmer, _ in self.mutations]

    @property
    def dest_amino_acids(self):
        return [
            codon_to_amino_acid[kmer[1] + mutation + kmer[3]]
            for _, kmer, mutation in self.mutations
        ]


@permacache(
    "frame_alignment_checks/amino_replacement/amino_replacement_experiment/amino_replacement_experiment_for_exon_7",
    key_function=dict(ex=lambda x: x.__dict__, model=stable_hash_cached),
)
def amino_replacement_experiment_for_exon(
    ex: CodingExon, model, model_cl, cl_model_clipped
) -> AminoAcidExperimentResultOnExon:
    acc, don, x_orig, x = data_for_experimentation(ex, model_cl=model_cl)
    yps = run_batched(
        lambda x: model(x).softmax(-1)[
            :, [acc - cl_model_clipped // 2, don - cl_model_clipped // 2], [1, 2]
        ],
        np.concatenate([[x_orig], x]).astype(np.float32),
        128,
        device=device_of(model),
    )
    return AminoAcidExperimentResultOnExon(
        yps_base=yps[0],
        yps_mut=yps[1:],
        mutations=compute_relevant_mutations_for_exon(ex),
        exon=ex,
    )


@permacache(
    "frame_alignment_checks/amino_replacement/amino_replacement_experiment/amino_replacement_experiment_for_exons_7",
    key_function=dict(
        exs=lambda xs: [x.__dict__ for x in xs], model=stable_hash_cached
    ),
)
def amino_replacement_experiment_for_exons(
    exons: List[CodingExon], model, model_cl, cl_model_clipped
) -> List[AminoAcidExperimentResultOnExon]:
    return [
        amino_replacement_experiment_for_exon(ex, model, model_cl, cl_model_clipped)
        for ex in tqdm.tqdm(exons)
    ]


@dataclass
class AminoAcidExperimentResultSummary:
    deltas: np.ndarray
    phase_of_sensitive_3mer: np.ndarray
    kmers: np.ndarray
    mutations: np.ndarray

    @classmethod
    def of(cls, exp_results: List[AminoAcidExperimentResultOnExon], thresholds):
        return cls(
            deltas=np.concatenate(
                [exp.accuracy_drop(thresholds) for exp in exp_results]
            ),
            phase_of_sensitive_3mer=np.concatenate([exp.phases for exp in exp_results]),
            kmers=np.array(
                [kmer for res in exp_results for _, kmer, _ in res.mutations]
            ),
            mutations=np.array(
                [mutation for res in exp_results for _, _, mutation in res.mutations]
            ),
        )

    def controlled_mean_by_phase(self):
        kmers_all = sorted(set(self.kmers))
        mutations = "ACGT"
        kmer_to_idx = {k: i for i, k in enumerate(kmers_all)}
        kmer_idxs = np.array([kmer_to_idx[kmer] for kmer in self.kmers])
        mutation_idxs = np.array([mutations.index(mut) for mut in self.mutations])
        stop_relevant = np.array(
            [
                [
                    "*" in kmer_mutation_to_before_after_codon(kmer, mut)
                    for kmer in kmers_all
                ]
                for mut in mutations
            ]
        ).T
        counts, sums = [np.zeros((len(kmers_all), 4, 3)) for _ in range(2)]
        key = kmer_idxs, mutation_idxs, self.phase_of_sensitive_3mer
        np.add.at(counts, key, 1)
        np.add.at(sums, key, self.deltas)
        mask = (counts != 0).all(-1) & ~stop_relevant
        return (sums[mask] / counts[mask]).mean(0)

    def controlled_mean_by_in_frame(self):
        phase_0, phase_1, phase_2 = self.controlled_mean_by_phase()
        return phase_0, (phase_1 + phase_2) / 2


# def __getitem__(self, index_like):
#     return AminoAcidExperimentResultSummary(
#         deltas=self.deltas[index_like],
#         phase_of_sensitive_3mer=self.phase_of_sensitive_3mer[index_like],
#         source_amino_acids=self.source_amino_acids[index_like],
#         dest_amino_acids=self.dest_amino_acids[index_like],
#     )

# def without_stops(self):
#     return self[(self.source_amino_acids != "*") & (self.dest_amino_acids != "*")]

# def deltas_by_whether_aligned(self):
#     return [
#         self.deltas[self.phase_of_sensitive_3mer == 0].mean(),
#         self.deltas[self.phase_of_sensitive_3mer != 0].mean(),
#     ]


def amino_replacement_experiment_single(
    model: ModelToAnalyze,
) -> AminoAcidExperimentResultSummary:
    exons = list(load_long_canonical_internal_coding_exons())
    exp_results = amino_replacement_experiment_for_exons(
        exons, model.model, model.model_cl, model.cl_model_clipped
    )
    # return exp_results
    return AminoAcidExperimentResultSummary.of(exp_results, model.thresholds)


def kmer_mutation_to_before_after_codon(kmer, mut):
    assert len(kmer) == 5
    after = kmer[:2] + mut + kmer[3:]
    return [amino_classification[codon_to_amino_acid[x[1:4]]] for x in [kmer, after]]
