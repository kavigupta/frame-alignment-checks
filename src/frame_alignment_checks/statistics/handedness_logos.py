from frame_alignment_checks.data.load import load_train_counts_by_phase
from frame_alignment_checks.phase_handedness.compute_self_agreement import all_9mers

from render_psam import render_psams


def relative_logos_by_phase():
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
        **kwargs,
    )
