import numpy as np
from matplotlib_venn import venn3


def closed_reading_frames_venn(ax, title, reading_frames_closed, tag, taa, tga):
    venn3(
        [set(np.where((w != 0) & reading_frames_closed)[0]) for w in (tag, taa, tga)],
        ("TAG", "TAA", "TGA"),
        ax=ax,
    )
    ax.set_title(f"{title}\nSequences where all reading frames are closed contain")
