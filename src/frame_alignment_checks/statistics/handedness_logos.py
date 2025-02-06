from render_psam import render_psams

from ..load_data import load_train_counts_by_phase
from ..phase_handedness.best_5mers_each import get_phase_specific_9mers
from ..phase_handedness.compute_self_agreement import all_9mers
from ..plotting.colors import line_color
from ..utils import draw_bases


def relative_logos_by_phase():
    # pylint: disable=unsubscriptable-object,no-member
    counts_by_phase = load_train_counts_by_phase()

    logo_overall = (counts_by_phase.sum(0)[..., None, None] * all_9mers()).sum(
        0
    ) / counts_by_phase.sum()
    logo_by_phase = (counts_by_phase[..., None, None] * all_9mers()).sum(
        1
    ) / counts_by_phase.sum(1)[:, None, None]
    relative_logo = logo_by_phase - logo_overall
    return relative_logo


def phase_handedness_plot_relative_logos(**kwargs):
    rlp = relative_logos_by_phase()
    render_psams(
        [rlp[-1], rlp[0], rlp[1]],
        psam_mode="raw",
        names=[""] * 3,
        axes_mode="just_y",
        figure_kwargs=dict(dpi=400),
        color_scheme={
            "A": line_color(2),
            "C": line_color(0),
            "G": line_color(1),
            "T": line_color(3),
        },
        **kwargs,
    )


def phase_handedness_print_statistics_by_phase():
    # pylint: disable=unsubscriptable-object,no-member
    counts_by_phase = load_train_counts_by_phase()
    phase_specific_9mers = get_phase_specific_9mers()

    print(f"Overall: {counts_by_phase.sum()}")
    print()
    for phase in range(3):
        print(f"Phase {phase}: {counts_by_phase[phase].sum(0)}")
        for idx in phase_specific_9mers[phase]:
            print(
                draw_bases(all_9mers()[idx]),
                *[f"{x:5d}" for x in counts_by_phase[:, idx]],
            )
