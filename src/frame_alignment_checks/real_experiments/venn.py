import numpy as np
from matplotlib_venn import venn3

from .saturation_mutagenesis import (
    SEQUENCE_PADDING_LEFT,
    SEQUENCE_PADDING_RIGHT,
    load_mutagenesis_table,
    mutagenesis_sequence_reading_frame_closed,
)


def closed_reading_frames_venn(ax, title, reading_frames_closed, tag, taa, tga):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    venn3(
        [set(np.where((w != 0) & reading_frames_closed)[0]) for w in (tag, taa, tga)],
        ("TAG", "TAA", "TGA"),
        ax=ax,
    )
    ax.set_title(f"{title}\nSequences where all reading frames are closed contain")


def closed_reading_frames_venn_sm(ax):
    table = load_mutagenesis_table()
    tag, taa, tga = [
        table.sequence.apply(
            lambda x, trimer=trimer: x[
                SEQUENCE_PADDING_LEFT:-SEQUENCE_PADDING_RIGHT
            ].count(trimer)
        )
        for trimer in ["TAG", "TAA", "TGA"]
    ]
    closed_reading_frames_venn(
        ax,
        "Saturation Mutagenesis",
        mutagenesis_sequence_reading_frame_closed(),
        tag,
        taa,
        tga,
    )
