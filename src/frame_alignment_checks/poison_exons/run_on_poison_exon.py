from typing import Dict, List

import numpy as np
import torch
import tqdm.auto as tqdm
from permacache import permacache, stable_hash

from ..load_data import load_nve_descriptors, load_poison_exon_sequence
from ..models import ModelToAnalyze
from ..utils import device_of, stable_hash_cached


@permacache(
    "frame_alignment_checks/poison_exons/experiment/run_model_on_exon",
    key_function=dict(model=stable_hash_cached),
)
def run_model_on_exon(
    *,
    gene_spec: tuple,
    model: torch.nn.Module,
    model_cl: int,
    cl_model_clipped: int,
    acc: int,
    don: int
):
    """
    Run the given model on the specified gene exon.

    :param gene_spec: Tuple containing the gene information.
    :param model: The model to be run.
    :param model_cl: The model's context length.
    :param cl_model_clipped: The clipped model context length.
    :param acc: The acceptor position, relative to the gene.
    :param don: The donor position, relative to the gene.

    :returns: The model's predictions for the acceptor and donor positions, as
    probabilities (not logits).
    """

    text = load_poison_exon_sequence(gene_spec, acc, don, model_cl)

    text = np.concatenate([np.eye(4, dtype=np.float32), [[0, 0, 0, 0]]]).astype(
        np.float32
    )[text]

    with torch.no_grad():
        yp = model(torch.tensor(text[None]).to(device_of(model))).softmax(-1)
        # coordinates will be offset by the clip // 2 because, as the entire sequence has
        # clip // 2 removed from the front.
        off = (model_cl // 2) - (cl_model_clipped // 2)
        yp = yp[0, [off, off + (don - acc)], [1, 2]]
        return yp.cpu().numpy()


@permacache(
    "frame_alignment_checks/poison_exons/experiment/poison_exon_scores",
    key_function=dict(model_to_analyze=stable_hash),
)
def poison_exon_scores(model_to_analyze: ModelToAnalyze, limit=None) -> np.ndarray:
    """
    Run the model on poison exons and return the log10 of the probabilities.

    :param model_to_analyze: The model to be analyzed.
    :param limit: The maximum number of poison exons to analyze. Useful for testing.
    """
    return np.array(
        [
            np.log10(
                run_model_on_exon(
                    gene_spec=nve_descriptor["gene_spec"],
                    model=model_to_analyze.model,
                    model_cl=model_to_analyze.model_cl,
                    cl_model_clipped=model_to_analyze.cl_model_clipped,
                    acc=nve_descriptor["acc"],
                    don=nve_descriptor["don"],
                ).mean()
            )
            for nve_descriptor in tqdm.tqdm(load_nve_descriptors()[:limit], delay=3)
        ]
    )


def poison_exon_scores_for_model_series(
    mods: Dict[str, List[ModelToAnalyze]], limit=None
) -> Dict[str, List[np.ndarray]]:
    """
    Run the model on poison exons and return the log10 of the probabilities.

    :param mods: A dictionary of models to be analyzed.
    :param limit: The maximum number of poison exons to analyze. Useful for testing.
    """
    return {
        name: [
            poison_exon_scores(model, limit=limit)
            for model in tqdm.tqdm(models, desc=name, delay=3)
        ]
        for name, models in tqdm.tqdm(
            mods.items(), desc="Running models on poison exons"
        )
    }
