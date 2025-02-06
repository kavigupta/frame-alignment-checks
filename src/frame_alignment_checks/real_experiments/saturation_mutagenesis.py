import numpy as np
import tqdm.auto as tqdm
from permacache import permacache, stable_hash
from run_batched import run_batched

from ..compute_stop_codons import all_frames_closed
from ..load_data import load_minigene, load_saturation_mutagenesis_table
from ..utils import device_of, extract_center, parse_sequence_as_one_hot
from .experiment_results import FullRealExperimentResult, RealExperimentResultForModel

SEQUENCE_PADDING_LEFT = 23
SEQUENCE_PADDING_RIGHT = 16
chunk_size = 100


@permacache(
    "modular_splicing/frame_alignment/saturation_mutagenesis/load_mutagenesis_table",
)
def load_mutagenesis_table():
    """
    Load the table from the saturation mutagenesis paper.
    """
    data = load_saturation_mutagenesis_table()
    data.columns = [
        "id",
        "hexmut_number",
        "posn_number",
        # Sequence (90 nt) from -23 to +16 relative to exon (51 nt) ends;
        # sorted by Hexmut#, then position#, then alphabetically.
        # Exon is in red, Variable Hexmut 6mer position is underlined.
        # Position 0 is the relative WT.
        # The mini-gene library (~3 kb PCR products) is available for experimentation. (email lac2@columbia.edu)
        "sequence",
        "EI",
        "rel_EI",
        "PUP",
        "LEI",
        "LEIsc",
        "Dot Bracket Structure",
        "MFE dGo",
        "MFE structure link",
        "base change(s)",
        "SBS / DBS",
    ]
    data = data[data.id == data.id]
    assert (data.id == np.arange(len(data)) + 1).all()
    data = data.reset_index(drop=True)
    del data["id"]
    return data


def mutagenesis_sequences():
    """
    Get the sequences from the saturation mutagenesis paper, one-hot encoded.
    """
    data = load_mutagenesis_table()
    mut_sequences = np.array(list(data.sequence.apply(parse_sequence_as_one_hot)))
    return mut_sequences


def mutagenesis_sequences_exons():
    """
    Like mutagenesis_sequences, but only the exon part.
    """
    mut_sequences = mutagenesis_sequences()
    return mut_sequences[:, SEQUENCE_PADDING_LEFT:-SEQUENCE_PADDING_RIGHT]


def mutagenesis_sequence_reading_frame_closed():
    """
    Compute whether the reading frame is closed in the exon sequences,
    for all three possible offsets.
    """
    exon_sequences = mutagenesis_sequences_exons()
    return all_frames_closed(exon_sequences)


@permacache(
    "modular_splicing/frame_alignment/saturation_mutagenesis/load_mutagenesis_minigene",
)
def load_mutagenesis_minigene():
    """
    Get the full minigene as used in the saturation mutagenesis paper.
    """
    minigene = load_minigene("WT1", 5)
    x = minigene["x"]
    y = minigene["y"]
    main_exon_start = minigene["main_exon_start"]
    main_exon_end = minigene["main_exon_end"]
    mut_range = slice(
        main_exon_start - SEQUENCE_PADDING_LEFT,
        main_exon_end + SEQUENCE_PADDING_RIGHT + 1,
    )

    return dict(
        x=x,
        y=y,
        exon_5_start=main_exon_start,
        exon_5_end=main_exon_end,
        mut_range=mut_range,
    )


@permacache(
    "modular_splicing/frame_alignment/saturation_mutagenesis/run_on_saturation_mutagenesis_data_2",
    key_function=dict(m=stable_hash),
)
def run_on_saturation_mutagenesis_data(m, cl):
    """
    Run the model on the saturation mutagenesis data, producing the predicted splice probabilities.
    """
    mut_sequences = mutagenesis_sequences()
    minigene = load_mutagenesis_minigene()
    x, exon_5_start, exon_5_end, mut_range = (
        minigene["x"],
        minigene["exon_5_start"],
        minigene["exon_5_end"],
        minigene["mut_range"],
    )
    results = []
    pad = cl // 2
    for i in tqdm.trange(0, mut_sequences.shape[0], chunk_size):
        mut_sequences_chunk = mut_sequences[i : i + chunk_size]
        x_batch = np.repeat(x[None], mut_sequences_chunk.shape[0], axis=0)
        x_batch[:, mut_range] = mut_sequences_chunk
        x_batch_padded = np.concatenate(
            [
                np.zeros((x_batch.shape[0], pad, 4), dtype=np.float32),
                x_batch,
                np.zeros((x_batch.shape[0], pad, 4), dtype=np.float32),
            ],
            axis=1,
        )
        acc, don = [
            x_batch_padded[:, loc : 1 + loc + cl] for loc in (exon_5_start, exon_5_end)
        ]
        res = run_batched(
            lambda x: extract_center(m, x),
            np.concatenate([acc, don]).astype(np.float32),
            32,
            device=device_of(m),
        )
        res = res.reshape(2, acc.shape[0], 3)
        results.append(res[[0, 1], :, [1, 2]])
    return np.concatenate(results, axis=1)


def run_on_saturation_mutagenesis_data_all(ms, cl):
    results = [run_on_saturation_mutagenesis_data(m, cl) for m in ms]
    return np.array(results)


def saturation_mutagenesis_experiment(mod):
    data = load_mutagenesis_table()

    ms = [mod.model_series[seed_idx].model for seed_idx in range(len(mod.model_series))]
    res = run_on_saturation_mutagenesis_data_all(ms, mod.needed_context_model).mean(1)
    actual = np.array(data.LEI)
    predicted = np.array(np.log2(res))
    return RealExperimentResultForModel(actual, predicted)


def saturation_mutagenesis_experiment_all(models):
    reading_frames_closed = mutagenesis_sequence_reading_frame_closed()
    return FullRealExperimentResult(
        {mod.name: saturation_mutagenesis_experiment(mod) for mod in models},
        [
            (~reading_frames_closed, "open reading frame"),
            (
                reading_frames_closed,
                "closed reading frame",
            ),
        ],
    )
