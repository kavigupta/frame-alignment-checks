"""
Run small deletions around the splice sites of coding exons through the
AlphaGenome batch variant interface and return the per-site delta table as
a ``DeletionAccuracyDeltaResult``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np
import tqdm
from permacache import permacache, stable_hash

from ..alphagenome_api import find_strand_track, predict_variants_with_retry
from ..alphagenome_calibration import (
    _exon_centered_interval,
    _seq_pos_to_genomic_1based,
    alphagenome_calibration_thresholds,
)
from ..coding_exon import CodingExon
from ..load_data import (
    load_long_canonical_internal_coding_exons,
    load_transcript_coords,
    load_validation_gene,
)
from .delete import (
    DeletionAccuracyDeltaResult,
    affected_splice_sites,
    deletion_ranges_for_exon,
    mutation_locations,
)

if TYPE_CHECKING:
    # alphagenome is an optional extra (Python >=3.10), imported lazily at runtime.
    from alphagenome.models import dna_client
    from alphagenome.models.dna_output import OutputType

# donor/acceptor track per affected_splice_site ["P5'SS", "3'SS", "5'SS", "N3'SS"].
_SITE_TRACK_TYPES = ("donor", "acceptor", "donor", "acceptor")

_COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}
_NTS = np.array(list("ACGT"))

# --- tuning constants ---
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

# Package-local cache dirs (shipped via package_data) so results travel with the
# install. `_refalt` versions the cache: deltas_for_exon now stores ref/alt, not a
# pre-differenced delta.
_CACHE_DIR_REFALT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "alphagenome_cache_refalt",
)


def run_alphagenome_deletion_experiment(
    exons: List[CodingExon],
    model: dna_client.DnaClient,
    output_type: OutputType,
    *,
    distance_out: int,
    delete_up_to: int,
    binary_metric: bool = True,
    thresholds: Optional[dict] = None,
    interval_len: int = 131072,
    ontology_terms: Sequence[str] = ("UBERON:0001157",),
    progress: bool = True,
) -> DeletionAccuracyDeltaResult:
    """
    Run :func:`deltas_for_exon` over ``exons``, returning a
    :class:`DeletionAccuracyDeltaResult` with ``raw_data`` shape
    ``(1, num_exons, delete_up_to, 4, 4)`` in input order.

    Each cell is the deletion's change in splice readout. ``binary_metric=True``
    (default, mirroring the CNN path) thresholds ref/alt at the calibrated
    donor/acceptor threshold and reports ``call(alt) - call(ref)`` in ``{-1,0,1}``;
    ``binary_metric=False`` reports continuous ``alt - ref``. ``thresholds``
    overrides the calibration (computed once when None).

    ``output_type`` must be ``SPLICE_SITES``: the readout and the calibration
    both index a donor/acceptor-typed track, which ``SPLICE_SITE_USAGE`` (tracks
    per assay) does not have.

    Exons with no transcript coords yield an all-NaN block (skipped by the
    NaN-aware aggregation). Splice-site disagreements keep their deltas and only
    fail the run if their rate reaches ``MAX_SPLICE_SITE_FAILURE_RATE``; any
    per-exon error is fatal. Issues are collected and raised together as a
    ``RuntimeError``. Per-exon results are cached; errors are not.
    """
    tc = load_transcript_coords()
    # per-site thresholds donor/acceptor/donor/acceptor (via _SITE_TRACK_TYPES).
    thr_vec = None
    if binary_metric:
        if thresholds is None:
            thresholds = alphagenome_calibration_thresholds(
                model,
                output_type,
                interval_len=interval_len,
                ontology_terms=ontology_terms,
                progress=progress,
            )
        thr_vec = np.array([thresholds[t] for t in _SITE_TRACK_TYPES])

    def metric(res):
        """Continuous or calibrated-binary delta of shape (delete_up_to, 4, 4)."""
        ref = np.asarray(res["ref"])
        alt = np.asarray(res["alt"])
        if not binary_metric:
            return alt - ref
        # preserve NaN (out-of-bounds sites) instead of nan > thr -> False -> 0.
        delta = (alt > thr_vec).astype(np.float64) - (ref > thr_vec).astype(np.float64)
        return np.where(np.isfinite(ref) & np.isfinite(alt), delta, np.nan)

    nan_block = np.full(
        (delete_up_to, len(mutation_locations), len(affected_splice_sites)), np.nan
    )
    per_exon = []
    failures = []
    ss_disagreements = []
    n_placeable = 0
    iterator = tqdm.tqdm(exons, desc="exons") if progress else exons
    for i, ex in enumerate(iterator):
        if ex.gene_idx not in tc:
            print(f"  exon {i} (gene_idx={ex.gene_idx}): no transcript coords; NaN")
            per_exon.append(nan_block)
            continue
        n_placeable += 1
        try:
            x_seq, _ = load_validation_gene(ex.gene_idx)
            res = deltas_for_exon(
                ex,
                tc[ex.gene_idx],
                # astroid mis-infers x_seq; argmax is valid on the ndarray.
                x_seq.argmax(-1),  # pylint: disable=no-member
                model,
                output_type,
                distance_out=distance_out,
                delete_up_to=delete_up_to,
                interval_len=interval_len,
                ontology_terms=ontology_terms,
            )
        except Exception as e:  # pylint: disable=broad-except
            print(f"  exon {i} (gene_idx={ex.gene_idx}): FAILED - {e}")
            failures.append((i, ex.gene_idx, e))
            continue
        # keep deltas regardless of disagreement; just record it below.
        per_exon.append(metric(res))
        if res["splice_site_failures"]:
            descs = "; ".join(res["splice_site_failures"])
            print(
                f"  exon {i} (gene_idx={ex.gene_idx}): splice-site disagreement - {descs}"
            )
            ss_disagreements.append((i, ex.gene_idx, descs))

    ss_rate, ss_fatal = _report_splice_site_disagreements(ss_disagreements, n_placeable)
    _raise_for_run_failures(failures, ss_disagreements, ss_rate, ss_fatal)

    raw_data = np.stack(per_exon)[None]  # (1, num_exons, delete_up_to, 4, 4)
    return DeletionAccuracyDeltaResult(raw_data=raw_data)


def alphagenome_deletion_experiment(
    model: dna_client.DnaClient,
    output_type: OutputType,
    *,
    distance_out: int,
    delete_up_to: int,
    binary_metric: bool = True,
    thresholds: Optional[dict] = None,
    limit: Optional[int] = None,
    interval_len: int = 131072,
    ontology_terms: Sequence[str] = ("UBERON:0001157",),
    progress: bool = True,
) -> DeletionAccuracyDeltaResult:
    """
    Full experiment: load all canonical internal coding exons and run every
    ``1..delete_up_to`` nt deletion through the model (the analogue of
    ``fac.deletion.experiment``). ``exp1`` is a capped smoke-test driver.

    :param model: AlphaGenome client (needs an explicit ``model_version``).
    :param output_type: ``SPLICE_SITES``; see
        :func:`run_alphagenome_deletion_experiment`.
    :param distance_out: nt from each splice site to place deletions.
    :param delete_up_to: longest deletion length (runs ``1..delete_up_to``).
    :param binary_metric: threshold to a binary call-delta (default) or return
        continuous ``alt - ref``. See :func:`run_alphagenome_deletion_experiment`.
    :param thresholds: optional ``{"donor","acceptor"}`` override.
    :param limit: run only the first ``limit`` exons; default all.
    :returns: :class:`DeletionAccuracyDeltaResult`, ``raw_data`` shape
        ``(1, num_exons, delete_up_to, 4, 4)``.
    """
    exons = load_long_canonical_internal_coding_exons()[:limit]
    return run_alphagenome_deletion_experiment(
        exons,
        model,
        output_type,
        distance_out=distance_out,
        delete_up_to=delete_up_to,
        binary_metric=binary_metric,
        thresholds=thresholds,
        interval_len=interval_len,
        ontology_terms=ontology_terms,
        progress=progress,
    )


@permacache(
    _CACHE_DIR_REFALT,
    key_function=dict(
        exon=lambda e: e.__dict__,
        # derived from exon.gene_idx, but hashed too so the key reflects them.
        gene_info=stable_hash,
        seq_idx=stable_hash,
        # served model identity, so cached folds don't collide under a shared None.
        model=lambda m: m._model_version,  # pylint: disable=protected-access
        output_type=str,
        ontology_terms=list,
    ),
    shelf_type="individual-file",
    driver="json",
)
def deltas_for_exon(
    exon: CodingExon,
    gene_info: dict,
    seq_idx: np.ndarray,
    model: dna_client.DnaClient,
    output_type: OutputType,
    *,
    distance_out: int,
    delete_up_to: int,
    interval_len: int = 131072,
    ontology_terms: Sequence[str] = ("UBERON:0001157",),
) -> dict:
    """
    Run every ``1..delete_up_to`` nt deletion ``distance_out`` nt from ``exon``'s
    acceptor and donor, on all four sides (``fac.deletion.mutation_locations``).

    :param exon: exon to perturb.
    :param gene_info: ``load_transcript_coords()`` entry for ``exon.gene_idx``.
    :param seq_idx: integer base indices for the gene, shape ``(L,)``.
    :param model: AlphaGenome client.
    :param output_type: ``SPLICE_SITES``; see
        :func:`run_alphagenome_deletion_experiment`.
    :returns: JSON-serializable dict (cached one file per exon):
        - ``"ref"``/``"alt"``: ``(delete_up_to, 4, 4)`` nested lists indexed by
          ``[deletion-1, mutation_location, affected_splice_site]``, the readouts
          at each annotated site (stored separately so either metric applies
          without re-querying). NaN where a site is outside the track interval.
        - ``"splice_site_failures"``: descriptions from check_splice_site_signals.
    """
    from alphagenome.data import genome

    _assert_explicit_model_version(model)
    _assert_deletion_fits_exon(
        exon, distance_out=distance_out, delete_up_to=delete_up_to
    )
    strand = gene_info["strand"]

    def seq_slice_to_ref_bases(start, end):
        return _seq_slice_to_ref_bases(gene_info, seq_idx, start, end)

    def seq_pos_to_genomic_1based(pos):
        return _seq_pos_to_genomic_1based(gene_info, pos)

    interval = _exon_centered_interval(gene_info, exon, interval_len)
    _assert_interval_on_chromosome(interval, exon, gene_info, interval_len)

    variants = []
    for seq_start, seq_end in deletion_ranges_for_exon(
        exon, distance_out=distance_out, delete_up_to=delete_up_to
    ):
        ref_bases = seq_slice_to_ref_bases(seq_start, seq_end)
        # leftmost genomic coordinate of the half-open deleted span [start, end)
        pos = seq_pos_to_genomic_1based(seq_start if strand == "+" else seq_end - 1)
        variants.append(
            genome.Variant(
                chromosome=gene_info["chrom"],
                position=pos,
                reference_bases=ref_bases,
                alternate_bases="",
            )
        )

    variant_outputs = predict_variants_with_retry(
        model,
        intervals=interval,
        variants=variants,
        ontology_terms=list(ontology_terms),
        requested_outputs=[output_type],
        progress_bar=False,
    )

    ref_ss_0 = variant_outputs[0].reference.get(output_type)
    site_track_idx = [
        find_strand_track(ref_ss_0, st_type, strand) for st_type in _SITE_TRACK_TYPES
    ]

    site_seq_positions = [
        exon.prev_donor,
        exon.acceptor,
        exon.donor,
        exon.next_acceptor,
    ]
    site_genomic = [seq_pos_to_genomic_1based(p) for p in site_seq_positions]

    # reported, not fatal: keep the deltas; the run-level rate bar decides.
    splice_site_failures = check_splice_site_signals(
        ref_ss_0,
        site_genomic,
        site_track_idx,
    )

    _assert_alt_tracks_left_shifted(
        variant_outputs,
        variants,
        output_type,
        site_track_idx[:2],  # donor, acceptor
    )

    # predict_variants indexes alt tracks by local position in the right-padded
    # alt sequence, so a site past the deletion reads alt at idx - del_len
    # (alphagenome issue #23).
    ref_raw = np.zeros((len(variants), 4))
    alt_raw = np.zeros((len(variants), 4))
    for vi, (vo, v) in enumerate(zip(variant_outputs, variants)):
        ref_ss = vo.reference.get(output_type)
        alt_ss = vo.alternate.get(output_type)
        _assert_ref_alt_tracks_aligned(ref_ss, alt_ss)
        track_start = ref_ss.interval.start
        W = ref_ss.values.shape[0]

        del_len = len(v.reference_bases)
        del_end_0based = v.position - 1 + del_len

        for si, (sg, ti) in enumerate(zip(site_genomic, site_track_idx)):
            idx = sg - 1 - track_start
            shifted = (sg - 1) >= del_end_0based
            alt_idx = idx - del_len if shifted else idx
            if not (0 <= idx < W and 0 <= alt_idx < W):
                ref_raw[vi, si] = np.nan
                alt_raw[vi, si] = np.nan
                continue
            ref_raw[vi, si] = float(ref_ss.values[idx, ti])
            alt_raw[vi, si] = float(alt_ss.values[alt_idx, ti])

    shape = (delete_up_to, len(mutation_locations), len(affected_splice_sites))
    return {
        "ref": ref_raw.reshape(shape).tolist(),
        "alt": alt_raw.reshape(shape).tolist(),
        "splice_site_failures": splice_site_failures,
    }


def _seq_slice_to_ref_bases(gene_info, seq_idx, start, end):
    """
    Forward-strand reference bases for seq slice ``[start, end)``. On ``-`` the seq
    letters are the reverse complement of the forward genome, undone here.
    """
    bases = "".join(_NTS[seq_idx[start:end]])
    if gene_info["strand"] == "+":
        return bases
    return "".join(_COMP[b] for b in reversed(bases))


# --- preconditions ---


def _assert_explicit_model_version(model):
    """The cache is keyed on the fold, so a ``None`` version would collide them."""
    assert model._model_version is not None, (  # pylint: disable=protected-access
        "model was created without an explicit model_version; pass "
        "model_version=... to dna_client.create() so cached results don't "
        "collide across folds"
    )


def _assert_deletion_fits_exon(exon, *, distance_out, delete_up_to):
    """Deletions must stay clear of the exon center (parity with the CNN path)."""
    assert (distance_out + delete_up_to) * 2 < exon.donor - exon.acceptor, (
        f"This deletion experiment (distance_out={distance_out}, "
        f"delete_up_to={delete_up_to}) is too large for the exon {exon}"
    )


def _assert_interval_on_chromosome(interval, exon, gene_info, interval_len):
    assert interval.start >= 0, (
        f"interval start {interval.start} < 0 for exon {exon}: the "
        f"{interval_len} nt window runs off the start of {gene_info['chrom']}"
    )


def _assert_ref_alt_tracks_aligned(ref_ss, alt_ss):
    """Both readouts are indexed by the same local position, so they must agree."""
    assert alt_ss.values.shape == ref_ss.values.shape
    assert alt_ss.interval.start == ref_ss.interval.start


# --- signal checks ---


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


def _frameshift_votes(ref_col, alt_col, del_len, *, track_start, del_end_0based, ti):
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


def _assert_alt_tracks_left_shifted(variant_outputs, variants, output_type, track_idxs):
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
            nt, nf, details = _frameshift_votes(
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


def _report_splice_site_disagreements(ss_disagreements, n_placeable):
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


def _raise_for_run_failures(failures, ss_disagreements, ss_rate, ss_fatal):
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
