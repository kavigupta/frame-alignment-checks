"""
Sanity checks on AlphaGenome's splice-site signal, used by the deletion
experiment. Per exon: that each annotated site lands on the model's predicted
peak, and that the alternate track is still left-shifted by del_len (alphagenome
issue #23) so the readout's ``idx - del_len`` correction stays valid. At run
level: the frequency-bar verdict on splice-site disagreements and the collected
per-exon failure reporting.
"""

import numpy as np

from .delete import affected_splice_sites

# check_splice_site_signals: a site passes as the max within ±WINDOW bp or by
# scoring >= FLOOR absolutely.
SPLICE_SITE_PEAK_WINDOW = 50
SPLICE_SITE_PEAK_FLOOR = 0.5
# max fraction of placeable exons that may disagree with the model's peak before
# the run is failed as a systematic coordinate bug.
MAX_SPLICE_SITE_FAILURE_RATE = 0.05
# Frameshift guard (alt track left-shifted by del_len, alphagenome issue #23):
# each sharp reference peak past the deletion votes on whether the shifted lookup
# still beats the un-shifted one; the run fails if >= MAX_FLIP_RATE flip.
FRAMESHIFT_PEAK_REL_FLOOR = 0.5  # peak >= this * track max
FRAMESHIFT_PEAK_SHARPNESS = 2  # ref >= this * neighbor at ±del_len
FRAMESHIFT_PEAK_SURVIVAL = 0.5  # surviving alt >= this * ref
FRAMESHIFT_MAX_FLIP_RATE = 0.05


def check_splice_site_signals(ref_track, site_genomic, site_track_idx):
    """
    Check each known splice site lands on AlphaGenome's predicted peak on its
    donor/acceptor track: it passes as the local max within ±WINDOW bp (ties ok)
    or by scoring >= FLOOR absolutely. Returns a description per failing in-window
    site (empty if all pass).
    """
    track_start = ref_track.interval.start
    W = ref_track.values.shape[0]
    failures = []
    for sg, ti, label in zip(site_genomic, site_track_idx, affected_splice_sites):
        idx = sg - 1 - track_start
        if not 0 <= idx < W:
            continue
        lo = max(0, idx - SPLICE_SITE_PEAK_WINDOW)
        hi = min(W, idx + SPLICE_SITE_PEAK_WINDOW + 1)
        site_val = float(ref_track.values[idx, ti])
        nb = np.concatenate(
            [ref_track.values[lo:idx, ti], ref_track.values[idx + 1 : hi, ti]]
        )
        nb_max = float(nb.max())
        # ``>=`` (not ``>``) so a site tied with a base of its own peak passes.
        if not (site_val >= nb_max or site_val >= SPLICE_SITE_PEAK_FLOOR):
            failures.append(
                f"{label}: value {site_val:.4f} not >= neighbor max {nb_max:.4f} "
                f"in window ±{SPLICE_SITE_PEAK_WINDOW} and below floor "
                f"{SPLICE_SITE_PEAK_FLOOR} (track {ti})"
            )
    return failures


def frameshift_votes(ref_col, alt_col, del_len, *, track_start, del_end_0based, ti):
    """
    Vote, over every sharp reference peak past the deletion, on whether ``alt`` is
    still left-shifted by ``del_len`` (issue #23). A peak passes when the shifted
    lookup ``alt[i-del_len]`` matches ref at least as well as ``alt[i]``, and flips
    otherwise. Returns ``(n_total, n_fail, flipped_details)``.
    """
    W = ref_col.shape[0]
    i = np.arange(W)
    room = (i >= del_len) & (i < W - del_len)
    past_del = (track_start + i) >= del_end_0based
    # clipped for a safe gather; masked out by ``room``.
    im = np.clip(i - del_len, 0, W - 1)
    ip = np.clip(i + del_len, 0, W - 1)
    nb = np.maximum(ref_col[im], ref_col[ip])
    shifted_alt = alt_col[im]
    rmax = float(ref_col.max()) if W else 0.0
    eligible = (
        room
        & past_del
        & (ref_col > 0)
        & (ref_col >= FRAMESHIFT_PEAK_REL_FLOOR * rmax)
        & (ref_col >= FRAMESHIFT_PEAK_SHARPNESS * nb)
        & (np.maximum(shifted_alt, alt_col) >= FRAMESHIFT_PEAK_SURVIVAL * ref_col)
    )
    idxs = np.nonzero(eligible)[0]
    shifted_err = np.abs(shifted_alt[idxs] - ref_col[idxs])
    unshifted_err = np.abs(alt_col[idxs] - ref_col[idxs])
    flip = shifted_err > unshifted_err
    details = [
        f"track={ti} pos0={track_start + int(k)} del_len={del_len}: "
        f"ref={ref_col[k]:.4f} shifted_alt={shifted_alt[k]:.4f} "
        f"unshifted_alt={alt_col[k]:.4f} "
        f"(shifted_err={shifted_err[j]:.4f} vs unshifted_err={unshifted_err[j]:.4f})"
        for j, k in enumerate(idxs)
        if flip[j]
    ]
    return int(idxs.size), int(flip.sum()), details


def assert_alt_tracks_left_shifted(variant_outputs, variants, output_type, track_idxs):
    """
    Fail if AlphaGenome's alt tracks are no longer left-shifted by ``del_len``, the
    behaviour the readout's ``idx - del_len`` correction compensates for. Every
    sharp reference peak past each deletion votes (not just the four annotated
    sites), so the bar below sees hundreds of votes per exon.
    """
    n_total = 0
    n_fail = 0
    flipped = []
    for vo, v in zip(variant_outputs, variants):
        ref_ss = vo.reference.get(output_type)
        alt_ss = vo.alternate.get(output_type)
        del_len = len(v.reference_bases)
        for ti in track_idxs:
            nt, nf, details = frameshift_votes(
                ref_ss.values[:, ti].astype(np.float64),
                alt_ss.values[:, ti].astype(np.float64),
                del_len,
                track_start=ref_ss.interval.start,
                del_end_0based=v.position - 1 + del_len,
                ti=ti,
            )
            n_total += nt
            n_fail += nf
            flipped += details

    # fail only if >= MAX_FLIP_RATE of the peaks flip (a real fix flips ~all).
    assert n_total == 0 or n_fail < FRAMESHIFT_MAX_FLIP_RATE * n_total, (
        "AlphaGenome alt track no longer appears left-shifted by del_len: the "
        "shifted readout failed to match the reference splice peak better than "
        f"the un-shifted readout in {n_fail}/{n_total} sharp peaks "
        f"(>= {FRAMESHIFT_MAX_FLIP_RATE:.0%}). If a release fixed the deletion "
        "frameshift (google-deepmind/alphagenome issue #23), drop the "
        "idx-del_len correction in deltas_for_exon. Flipped peak(s):"
        + "".join(f"\n    - {d}" for d in flipped[:10])
    )


# --- run-level checks ---


def report_splice_site_disagreements(ss_disagreements, n_placeable):
    """
    Print the consolidated disagreement listing and decide whether the rate is
    high enough to look like a systematic coordinate bug rather than noise.
    Returns ``(rate, fatal)``.
    """
    rate = len(ss_disagreements) / max(n_placeable, 1)
    fatal = bool(ss_disagreements) and rate >= MAX_SPLICE_SITE_FAILURE_RATE
    if ss_disagreements:
        verdict = (
            "EXCEEDED, failing run" if fatal else "within bar, kept (deltas retained)"
        )
        # consolidated list (inline prints get buried in the progress bar).
        listing = "\n".join(
            f"  exon {i} (gene_idx={gene_idx}): {descs}"
            for i, gene_idx, descs in ss_disagreements
        )
        print(
            f"  splice-site sanity: {len(ss_disagreements)}/{n_placeable} placeable "
            f"exon(s) disagreed (rate {rate:.3%}, bar "
            f"{MAX_SPLICE_SITE_FAILURE_RATE:.1%}) -- {verdict}:\n{listing}"
        )
    return rate, fatal


def raise_for_run_failures(failures, ss_disagreements, ss_rate, ss_fatal):
    """Raise every per-exon failure together, chained to the first one."""
    if not (failures or ss_fatal):
        return
    parts = [
        f"  exon {i} (gene_idx={gene_idx}): {type(e).__name__}: {e}"
        for i, gene_idx, e in failures
    ]
    if ss_fatal:
        parts += [
            f"  exon {i} (gene_idx={gene_idx}): splice-site disagreement: {descs}"
            for i, gene_idx, descs in ss_disagreements
        ]
    reason = f"{len(failures)} hard failure(s)" + (
        f" and splice-site disagreement rate {ss_rate:.3%} >= "
        f"{MAX_SPLICE_SITE_FAILURE_RATE:.1%}"
        if ss_fatal
        else ""
    )
    cause = failures[0][2] if failures else None
    raise RuntimeError(
        f"AlphaGenome deletion experiment failed ({reason}):\n" + "\n".join(parts)
    ) from cause
