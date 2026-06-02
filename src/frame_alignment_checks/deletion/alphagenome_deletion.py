"""
Run small deletions around the splice sites of coding exons through the
AlphaGenome batch variant interface and return the per-site delta table as
a ``DeletionAccuracyDeltaResult``.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, List, Sequence

import numpy as np
import tqdm
from permacache import permacache, stable_hash

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
    # alphagenome requires Python >=3.10 and is an optional extra; these imports
    # are only needed for type annotations (kept lazy via ``from __future__
    # import annotations``). The runtime imports live inside the functions that
    # actually use alphagenome, so the module imports fine without it installed.
    from alphagenome.models import dna_client
    from alphagenome.models.dna_output import OutputType

# donor track for donor sites, acceptor track for acceptor sites; aligned with
# ``affected_splice_sites`` = ["P5'SS", "3'SS", "5'SS", "N3'SS"].
_SITE_TRACK_TYPES = ("donor", "acceptor", "donor", "acceptor")

_COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}
_NTS = np.array(list("ACGT"))

# --- tuning constants ---
# ``_predict_variants_with_retry``: attempts before giving up on transient
# grpc RpcErrors.
PREDICT_MAX_ATTEMPTS = 5
# ``check_splice_site_signals``: a site passes if it is the maximum within
# ±WINDOW bp of its annotated position, or scores at least FLOOR in absolute
# terms regardless of the window.
SPLICE_SITE_PEAK_WINDOW = 50
SPLICE_SITE_PEAK_FLOOR = 0.5
# ``run_alphagenome_deletion_experiment``: fraction of placeable exons that may
# disagree with the model's predicted peak before the run is treated as having a
# systematic coordinate bug and failed (rather than scattered discrepancies).
MAX_SPLICE_SITE_FAILURE_RATE = 0.05
# Frameshift guard in ``deltas_for_exon`` (alt track left-shifted by del_len,
# google-deepmind/alphagenome issue #23). A central peak only votes if it is
# sharp at the del_len scale (ref >= SHARPNESS * neighbor) and largely survived
# the deletion (max(shifted, unshifted) alt >= SURVIVAL * ref); the guard fails
# only if at least MAX_FLIP_RATE of the voting peaks prefer the un-shifted
# readout (which would mean a release fixed the frameshift).
FRAMESHIFT_PEAK_SHARPNESS = 2
FRAMESHIFT_PEAK_SURVIVAL = 0.5
FRAMESHIFT_MAX_FLIP_RATE = 0.25

# Package-local cache directory (shipped with the package via package_data), so
# precomputed AlphaGenome results travel with the install instead of living in
# the user's global permacache. Resolved from __file__ so it works wherever the
# package is installed; an absolute path overrides permacache's default base.
# ``_refalt`` versions the cache: ``deltas_for_exon`` now stores ref/alt readouts
# rather than a pre-differenced delta, so old entries are intentionally bypassed.
_CACHE_DIR_REFALT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "alphagenome_cache_refalt",
)
# Calibration thresholds (one tiny dict per model_version/output_type/ontology),
# cached package-locally so they ship with the install like ``_CACHE_DIR``.
_CACHE_DIR_CALIB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "alphagenome_calibration_cache",
)


def _with_rpc_retry(call, what):
    """
    Call ``call()`` with exponential backoff on ``grpc.RpcError``. Re-raises
    after ``PREDICT_MAX_ATTEMPTS`` failures. ``what`` is used only in the
    retry log line.
    """
    import grpc

    for attempt in range(1, PREDICT_MAX_ATTEMPTS + 1):
        try:
            return call()
        except grpc.RpcError as e:
            if attempt == PREDICT_MAX_ATTEMPTS:
                raise
            print(
                f"  {what} RpcError (attempt {attempt}/{PREDICT_MAX_ATTEMPTS}): "
                f"{e.code() if hasattr(e, 'code') else e}; retrying"
            )
            time.sleep(2 ** (attempt - 1))
    # Unreachable: the final attempt above either returns or re-raises.
    raise AssertionError(f"{what} retry loop exited without returning")


def _predict_variants_with_retry(model, **kwargs):
    """``model.predict_variants(**kwargs)`` with grpc retry (see :func:`_with_rpc_retry`)."""
    return _with_rpc_retry(lambda: model.predict_variants(**kwargs), "predict_variants")


def _predict_interval_with_retry(model, **kwargs):
    """``model.predict_interval(**kwargs)`` with grpc retry (see :func:`_with_rpc_retry`)."""
    return _with_rpc_retry(lambda: model.predict_interval(**kwargs), "predict_interval")


def _seq_pos_to_genomic_1based(gene_info, pos):
    """
    Map a sequence/transcript coordinate ``pos`` (0-based index into the
    validation gene, oriented 5'->3') to its 1-based genomic (hg38) coordinate.
    The gene sequence spans exactly ``[hg38_start, hg38_end]``, so seq position 0
    is the genomically-leftmost base on ``+`` strand and the rightmost on ``-``.
    """
    if gene_info["strand"] == "+":
        return gene_info["hg38_start"] + pos
    return gene_info["hg38_end"] - pos


def _seq_slice_to_ref_bases(gene_info, seq_idx, start, end):
    """
    The forward-genomic-strand reference bases for the half-open seq slice
    ``[start, end)``. On ``-`` strand the seq letters are the reverse complement
    of the forward genome, so we undo that to get VCF-style forward bases.
    """
    bases = "".join(_NTS[seq_idx[start:end]])
    if gene_info["strand"] == "+":
        return bases
    return "".join(_COMP[b] for b in reversed(bases))


def _find_strand_track(ss, st_type, strand):
    """
    Index of the ``st_type`` ("donor"/"acceptor") track on ``strand`` within an
    AlphaGenome splice-site output ``ss``. Raises if none is found.
    """
    track_names = list(ss.metadata["name"])
    track_strands = list(ss.metadata["strand"])
    for t, (tn, ts) in enumerate(zip(track_names, track_strands)):
        if ts == strand and st_type in tn.lower():
            return t
    raise ValueError(
        f"No {st_type} track found for strand {strand}; "
        f"tracks={list(zip(track_names, track_strands))}"
    )


def check_splice_site_signals(ref_track, site_genomic, site_track_idx):
    """
    Sanity-check that each known splice site maps onto AlphaGenome's predicted
    splice peak on its matched donor/acceptor track. A correctly-mapped site
    sits on the (sharp, ~1-valued) peak while bases even a few bp away score
    ~0, so the site passes if it is either the local maximum within
    ``±SPLICE_SITE_PEAK_WINDOW`` bp (ties pass) **or** still a strong peak in
    absolute terms (``>= SPLICE_SITE_PEAK_FLOOR``) -- the latter tolerates a
    distinct, real neighbouring
    splice site that happens to score slightly higher in the wide window. A
    genuinely mis-mapped (or model-disagreed) site fails both (it sits on the ~0
    background).

    Returns a list of human-readable descriptions, one per failing in-window
    site (empty if all pass). This is a *disagreement* report, not an error: a
    failure means AlphaGenome puts the peak a few bp off the annotation (a real,
    scattered model/annotation discrepancy, not a coordinate bug), so the caller
    keeps the exon's computed deltas and only treats the disagreement as fatal
    if it turns out to be systematic (see the frequency bar in
    :func:`run_alphagenome_deletion_experiment`). Sites whose genomic position
    falls outside the track interval are skipped (those are handled as
    out-of-bounds NaNs in the delta readout). Uses the reference (un-variant)
    prediction.
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
        # ``>=`` so a site that is itself the (tied) maximum passes; a strict
        # ``>`` spuriously failed sites tied with an adjacent base of their own
        # peak.
        if not (site_val >= nb_max or site_val >= SPLICE_SITE_PEAK_FLOOR):
            failures.append(
                f"{label}: value {site_val:.4f} not >= neighbor max {nb_max:.4f} "
                f"in window ±{SPLICE_SITE_PEAK_WINDOW} and below floor "
                f"{SPLICE_SITE_PEAK_FLOOR} (track {ti})"
            )
    return failures


@permacache(
    _CACHE_DIR_REFALT,
    key_function=dict(
        exon=lambda e: e.__dict__,
        # gene_info and seq_idx are derived from exon.gene_idx, but hash them
        # anyway so the key faithfully reflects the actual inputs.
        gene_info=stable_hash,
        seq_idx=stable_hash,
        # the served model identity (e.g. "ALL_FOLDS"/"FOLD_0"); the client must
        # have been created with an explicit model_version so the cache key
        # distinguishes folds rather than collapsing them onto a shared None.
        model=lambda m: m._model_version,  # pylint: disable=protected-access
        output_type=str,
        ontology_terms=list,
    ),
    shelf_type="individual-file",
    driver="json",
)
def deltas_for_exon(  # pylint: disable=too-many-statements
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
    Run all deletions of length ``1..delete_up_to`` placed ``distance_out`` nt
    away (in seq/transcript coordinates) from the acceptor and donor of
    ``exon``, on each of the four sides defined by
    ``fac.deletion.mutation_locations``.

    :param exon: the exon to perturb.
    :param gene_info: dict for ``exon.gene_idx`` from ``load_transcript_coords()``.
    :param seq_idx: integer base indices for the gene (shape ``(L,)``).
    :param model: AlphaGenome client.
    :param output_type: ``OutputType.SPLICE_SITES`` or ``OutputType.SPLICE_SITE_USAGE``.
    :returns: a JSON-serializable dict (so it can be cached one-file-per-exon).
        ``"ref"`` and ``"alt"`` are stored separately (rather than a single
        pre-differenced delta) so the caller can apply either the continuous
        ``alt - ref`` metric or the calibrated binary metric without re-querying
        the model:

        - ``"ref"`` / ``"alt"``: nested lists of shape ``(delete_up_to, 4, 4)``
          indexed by ``[deletion - 1, mutation_location, affected_splice_site]``
          (wrap in ``np.asarray``), holding the reference and alternate (post-
          deletion) splice-site readouts at each annotated site. Individual
          entries are NaN only where a site falls out of the track interval.
        - ``"splice_site_failures"``: list of descriptions of any annotated
          splice sites that didn't land on AlphaGenome's predicted peak (see
          :func:`check_splice_site_signals`). A *report*, not an error -- the
          readouts are still valid; the caller applies a frequency bar.
    """
    from alphagenome.data import genome

    # The model identity is part of the cache key, so the client must declare an
    # explicit model_version -- otherwise results from different folds would
    # collide under a shared ``None`` key.
    assert model._model_version is not None, (  # pylint: disable=protected-access
        "model was created without an explicit model_version; pass "
        "model_version=... to dna_client.create() so cached results don't "
        "collide across folds"
    )
    # The deletions reach out ``distance_out + delete_up_to`` nt on each side of
    # both splice sites; require they stay clear of the exon center (and of each
    # other) just as ``basic_deletion_experiment`` does for the CNN path.
    assert (distance_out + delete_up_to) * 2 < exon.donor - exon.acceptor, (
        f"This deletion experiment (distance_out={distance_out}, "
        f"delete_up_to={delete_up_to}) is too large for the exon {exon}"
    )
    strand = gene_info["strand"]

    def seq_slice_to_ref_bases(start, end):
        return _seq_slice_to_ref_bases(gene_info, seq_idx, start, end)

    def seq_pos_to_genomic_1based(pos):
        return _seq_pos_to_genomic_1based(gene_info, pos)

    exon_mid_0based = seq_pos_to_genomic_1based((exon.acceptor + exon.donor) // 2) - 1
    interval = genome.Interval(
        chromosome=gene_info["chrom"],
        start=exon_mid_0based - interval_len // 2,
        end=exon_mid_0based + interval_len // 2,
    )
    assert interval.start >= 0, (
        f"interval start {interval.start} < 0 for exon {exon}: the "
        f"{interval_len} nt window runs off the start of {gene_info['chrom']}"
    )

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

    variant_outputs = _predict_variants_with_retry(
        model,
        intervals=interval,
        variants=variants,
        ontology_terms=list(ontology_terms),
        requested_outputs=[output_type],
        progress_bar=False,
    )

    ref_ss_0 = variant_outputs[0].reference.get(output_type)
    site_track_idx = [
        _find_strand_track(ref_ss_0, st_type, strand) for st_type in _SITE_TRACK_TYPES
    ]

    site_seq_positions = [
        exon.prev_donor,
        exon.acceptor,
        exon.donor,
        exon.next_acceptor,
    ]
    site_genomic = [seq_pos_to_genomic_1based(p) for p in site_seq_positions]

    # A disagreement here (the model puts a peak a few bp off the annotation) is
    # reported, not fatal: we still compute and return this exon's deltas. The
    # run-level frequency bar decides whether the *rate* of disagreement looks
    # systematic enough to abort.
    splice_site_failures = check_splice_site_signals(
        variant_outputs[0].reference.get(output_type),
        site_genomic,
        site_track_idx,
    )

    # raw_per_variant[variant_index, site] then reshape to (deletion, location, site)
    # `predict_variants` returns alt tracks that span the same array shape as
    # ref but are indexed by **local position in the right-padded alt sequence**
    # (the server pulls del_len extra reference bases from beyond the interval
    # so the alt sequence is still W bp). Local index i in alt therefore maps
    # to genomic position start+i before the deletion and start+i+del_len after
    # it -- so for sites past the deletion we look up alt at idx - del_len.
    # (See google-deepmind/alphagenome issue #23.)
    ref_raw = np.zeros((len(variants), 4))
    alt_raw = np.zeros((len(variants), 4))
    # Evidence that AlphaGenome's alt track is still left-shifted by del_len for
    # deletions -- the un-fixed behavior the idx-del_len readout corrects for
    # (issue #23, confirmed open by a maintainer). For each sharp, surviving
    # central splice peak read on the shifted branch, the shifted lookup
    # (idx-del_len) must match the reference peak better than the un-shifted
    # lookup (idx). If a future release fixes the frameshift this flips, and we
    # fail loudly here rather than silently double-correcting.
    frameshift_checks = []
    frameshift_diag = []  # one (passed, detail-string) per voting peak
    for vi, (vo, v) in enumerate(zip(variant_outputs, variants)):
        ref_ss = vo.reference.get(output_type)
        alt_ss = vo.alternate.get(output_type)
        track_start = ref_ss.interval.start
        W = ref_ss.values.shape[0]
        assert alt_ss.values.shape == ref_ss.values.shape
        assert alt_ss.interval.start == track_start

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
            rv = float(ref_ss.values[idx, ti])
            av = float(alt_ss.values[alt_idx, ti])
            ref_raw[vi, si] = rv
            alt_raw[vi, si] = av

            # Frameshift guard. Only the central acceptor/donor (si 1, 2) are
            # validated sharp peaks. Require a peak that is sharp at the del_len
            # scale in the reference (rv >= SHARPNESS*neighbor) and that largely
            # survived the deletion, so the shifted-vs-unshifted comparison is
            # well-defined; otherwise this (vi, si) just doesn't vote.
            if shifted and si in (1, 2) and rv > 0 and idx + del_len < W:
                nb = max(
                    float(ref_ss.values[idx - del_len, ti]),
                    float(ref_ss.values[idx + del_len, ti]),
                )
                unshifted = float(alt_ss.values[idx, ti])
                if (
                    rv >= FRAMESHIFT_PEAK_SHARPNESS * nb
                    and max(av, unshifted) >= FRAMESHIFT_PEAK_SURVIVAL * rv
                ):
                    shifted_err = abs(av - rv)
                    unshifted_err = abs(unshifted - rv)
                    passed = shifted_err <= unshifted_err
                    frameshift_checks.append(passed)
                    frameshift_diag.append(
                        (
                            passed,
                            f"del_len={del_len} loc={mutation_locations[vi % 4]!r} "
                            f"site={affected_splice_sites[si]!r}: "
                            f"ref={rv:.4f} shifted_alt={av:.4f} unshifted_alt={unshifted:.4f} "
                            f"-> shifted_err={shifted_err:.4f} vs unshifted_err={unshifted_err:.4f} "
                            f"(margin={unshifted_err - shifted_err:+.4f})",
                        )
                    )

    # No qualifying peak is vacuously fine (others vote). A real upstream fix
    # flips essentially every qualifying peak, so only error when a substantial
    # fraction (>= FRAMESHIFT_MAX_FLIP_RATE) flip; isolated flips on weak peaks
    # are noise.
    flipped = "".join(f"\n    - {d}" for ok, d in frameshift_diag if not ok)
    n_fail = frameshift_checks.count(False)
    n_total = len(frameshift_checks)
    assert n_total == 0 or n_fail < FRAMESHIFT_MAX_FLIP_RATE * n_total, (
        "AlphaGenome alt track no longer appears left-shifted by del_len: the "
        "shifted readout failed to match the reference splice peak better than "
        f"the un-shifted readout in {n_fail}/{n_total} checks "
        f"(>= {FRAMESHIFT_MAX_FLIP_RATE:.0%}). If a release fixed the deletion "
        "frameshift (google-deepmind/alphagenome issue #23), drop the "
        "idx-del_len correction in deltas_for_exon. "
        f"Flipped peak(s):{flipped}"
    )

    # (delete_up_to, len(mutation_locations), len(affected_splice_sites))
    shape = (delete_up_to, len(mutation_locations), len(affected_splice_sites))
    return {
        "ref": ref_raw.reshape(shape).tolist(),
        "alt": alt_raw.reshape(shape).tolist(),
        "splice_site_failures": splice_site_failures,
    }


# Track types whose calibrated thresholds the binary metric applies to each of
# the four affected splice sites, aligned with ``_SITE_TRACK_TYPES`` /
# ``affected_splice_sites``.
_CALIB_TRACK_TYPES = ("donor", "acceptor")


@permacache(
    _CACHE_DIR_CALIB,
    key_function=dict(
        # served model identity (e.g. "ALL_FOLDS"/"FOLD_0"); see deltas_for_exon.
        model=lambda m: m._model_version,  # pylint: disable=protected-access
        output_type=str,
        ontology_terms=list,
    ),
    shelf_type="individual-file",
    driver="json",
)
def alphagenome_calibration_thresholds(
    model: dna_client.DnaClient,
    output_type: OutputType,
    *,
    interval_len: int = 131072,
    ontology_terms: Sequence[str] = ("UBERON:0001157",),
    limit: int = None,
    progress: bool = True,
) -> dict:
    """
    Calibrate per-track-type decision thresholds for AlphaGenome's splice-site
    readout -- the AlphaGenome analogue of
    :func:`fac.models.calibration_accuracy_and_thresholds`.

    Mirrors the CNN calibration: over every validation gene that has a canonical
    internal coding exon (and transcript coords), it gathers the model's
    predicted donor/acceptor value at every annotated position and picks, per
    track type, the threshold ``quantile(values, 1 - base_rate)`` so the model
    calls the correct *number* of donor/acceptor sites. The reference
    (un-variant) prediction is read from a single :meth:`predict_interval` call
    per gene.

    Each gene is scored in **one fixed ``interval_len`` window centred on the
    gene** -- the same window length the deletion experiment uses -- so every
    position is read under the same generous, uniform context the experiment
    reads its splice sites in. This deliberately avoids sizing the window to the
    gene: a window only marginally larger than the gene would push its splice
    sites to the window edges, where AlphaGenome's predictions are least
    reliable, biasing the calibrated distribution. Genes longer than
    ``interval_len`` (a small minority) contribute only their central
    ``interval_len`` of positions; the dropped tails are exactly the positions
    that would otherwise sit near a window edge, so the distributional threshold
    is unaffected.

    Thresholds are output-type-specific (``SPLICE_SITES`` and
    ``SPLICE_SITE_USAGE`` live on different scales), ontology-specific, and
    window-specific, so the cache is keyed on ``output_type``, ``ontology_terms``
    and ``interval_len`` alongside the model fold. The same ``ontology_terms``
    and ``interval_len`` must be used here and in the deletion experiment for the
    thresholds to match the values they're applied to (the experiment wires this
    up automatically).

    :returns: a JSON-serializable dict with float ``"donor"`` / ``"acceptor"``
        thresholds, the observed base rates ``"frac_donor"`` / ``"frac_acceptor"``,
        and the resulting recall ``"recall_donor"`` / ``"recall_acceptor"`` (the
        fraction of true sites called at threshold) for reference.
    """
    from alphagenome.data import genome

    assert model._model_version is not None, (  # pylint: disable=protected-access
        "model was created without an explicit model_version; pass "
        "model_version=... to dna_client.create() so cached thresholds don't "
        "collide across folds"
    )

    tc = load_transcript_coords()
    gene_idxs = sorted({ex.gene_idx for ex in load_long_canonical_internal_coding_exons()})
    gene_idxs = [g for g in gene_idxs if g in tc][:limit]

    # values[type] collects predicted readouts at every labelled position;
    # truth[type] the matching 0/1 label. "donor" <-> label channel 2 (y[:, 2]),
    # "acceptor" <-> channel 1, matching load_validation_gene's (null, acc, don).
    label_channel = {"acceptor": 1, "donor": 2}
    values = {t: [] for t in _CALIB_TRACK_TYPES}
    truth = {t: [] for t in _CALIB_TRACK_TYPES}

    iterator = tqdm.tqdm(gene_idxs, desc="calibration genes") if progress else gene_idxs
    for gene_idx in iterator:
        gene_info = tc[gene_idx]
        _, y = load_validation_gene(gene_idx)
        gene_len = y.shape[0]

        mid_genomic_0based = (gene_info["hg38_start"] + gene_info["hg38_end"]) // 2
        start = max(0, mid_genomic_0based - interval_len // 2)
        interval = genome.Interval(
            chromosome=gene_info["chrom"], start=start, end=start + interval_len
        )

        pred = _predict_interval_with_retry(
            model,
            interval=interval,
            requested_outputs=[output_type],
            ontology_terms=list(ontology_terms),
        )
        ss = pred.get(output_type)
        track_start = ss.interval.start
        W = ss.values.shape[0]
        ti = {t: _find_strand_track(ss, t, gene_info["strand"]) for t in _CALIB_TRACK_TYPES}

        # genomic 1-based position of each seq index, vectorised
        positions = np.arange(gene_len)
        if gene_info["strand"] == "+":
            genomic_1based = gene_info["hg38_start"] + positions
        else:
            genomic_1based = gene_info["hg38_end"] - positions
        idx = genomic_1based - 1 - track_start
        in_bounds = (idx >= 0) & (idx < W)
        idx_ib = idx[in_bounds]
        for t in _CALIB_TRACK_TYPES:
            values[t].append(ss.values[idx_ib, ti[t]])
            truth[t].append((y[in_bounds, label_channel[t]] > 0.5).astype(np.float64))

    result = {}
    for t in _CALIB_TRACK_TYPES:
        vals = np.concatenate(values[t])
        tru = np.concatenate(truth[t])
        frac = float(tru.mean())
        thr = float(np.quantile(vals, 1 - frac))
        called = (vals > thr) & (tru > 0.5)
        recall = float(called.sum() / max((tru > 0.5).sum(), 1))
        result[t] = thr
        result[f"frac_{t}"] = frac
        result[f"recall_{t}"] = recall
    return result


def run_alphagenome_deletion_experiment(
    exons: List[CodingExon],
    model: dna_client.DnaClient,
    output_type: OutputType,
    *,
    distance_out: int,
    delete_up_to: int,
    binary_metric: bool = True,
    thresholds: dict = None,
    interval_len: int = 131072,
    ontology_terms: Sequence[str] = ("UBERON:0001157",),
    progress: bool = True,
) -> DeletionAccuracyDeltaResult:
    """
    Run :func:`deltas_for_exon` across a list of exons and return a
    :class:`DeletionAccuracyDeltaResult` whose ``raw_data`` has shape
    ``(1, num_exons, delete_up_to, 4, 4)`` (single seed), with the exons in the
    same order as ``exons`` (no rows dropped or reordered).

    The per-cell value is the change in splice-site readout at each annotated
    site caused by the deletion. With ``binary_metric=True`` (the default, to
    mirror the CNN path's :func:`fac.deletion.experiment`) each site's reference
    and alternate readout is first thresholded at its calibrated decision
    threshold (donor vs. acceptor; see :func:`alphagenome_calibration_thresholds`)
    and the delta is ``call(alt) - call(ref)`` in ``{-1, 0, +1}`` -- the analogue
    of the CNN binary accuracy-delta. With ``binary_metric=False`` the cell is
    the raw continuous ``alt - ref`` readout difference (the analogue of the CNN
    ``binary_metric=False`` delta); the two are *not* directly comparable.
    ``thresholds`` (a dict with ``"donor"``/``"acceptor"`` floats) overrides the
    calibrated thresholds; when ``None`` and ``binary_metric`` is set they are
    computed once via :func:`alphagenome_calibration_thresholds` with the same
    ``ontology_terms``.

    Exons whose gene has no entry in ``load_transcript_coords()`` cannot be
    placed genomically, so they yield an all-NaN block (skipped by the
    NaN-aware aggregation on :class:`DeletionAccuracyDeltaResult`) rather than
    failing the run. NaN means "couldn't place this", nothing else -- within a
    placed exon, only sites that fall outside the track interval are NaN.

    Exons whose splice-site sanity check reports a disagreement (AlphaGenome
    places a peak a few bp off the annotated coordinate for a small minority of
    real sites) **keep their computed deltas** -- this is a model/annotation
    discrepancy, not a placement failure, so the data is retained rather than
    NaN'd. The disagreements are tallied and, only if their rate across
    placeable exons meets ``MAX_SPLICE_SITE_FAILURE_RATE`` (which would signal a
    *systematic* coordinate bug rather than scattered discrepancies), the run is
    failed. Any genuine per-exon error is always fatal. All issues are recorded
    and processing continues to the end so they're reported together; if the run
    fails, a ``RuntimeError`` is raised and no result is returned. (Successful
    per-exon predictions are cached individually by :func:`deltas_for_exon`, and
    errors are never cached, so re-running after fixing the cause only recomputes
    the failures.)
    """
    tc = load_transcript_coords()
    # Per-site thresholds aligned with affected_splice_sites (via _SITE_TRACK_TYPES):
    # donor/acceptor/donor/acceptor. Computed once (cached) for the whole run.
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
        # Threshold per site (last axis); preserve NaN where a site was out of
        # bounds rather than letting (nan > thr) -> False fabricate a 0 delta.
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
                # pylint (astroid numpy brain) mis-infers x_seq as np.array
                # itself; argmax is valid on the actual ndarray.
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
        # Keep the deltas regardless of any splice-site disagreement; only record
        # the disagreement for the frequency bar / reporting below.
        per_exon.append(metric(res))
        if res["splice_site_failures"]:
            descs = "; ".join(res["splice_site_failures"])
            print(
                f"  exon {i} (gene_idx={ex.gene_idx}): splice-site disagreement - {descs}"
            )
            ss_disagreements.append((i, ex.gene_idx, descs))

    # The splice-site disagreements only sink the run if they're frequent enough
    # to look systematic; a scattered minority is reported but kept.
    ss_rate = ss_disagreements and len(ss_disagreements) / max(n_placeable, 1)
    ss_fatal = ss_disagreements and ss_rate >= MAX_SPLICE_SITE_FAILURE_RATE

    if ss_disagreements:
        verdict = (
            "EXCEEDED, failing run"
            if ss_fatal
            else "within bar, kept (deltas retained)"
        )
        # Print the full consolidated list (the inline lines above get buried in
        # the progress bar) so every disagreeing exon is reported together.
        listing = "\n".join(
            f"  exon {i} (gene_idx={gene_idx}): {descs}"
            for i, gene_idx, descs in ss_disagreements
        )
        print(
            f"  splice-site sanity: {len(ss_disagreements)}/{n_placeable} placeable "
            f"exon(s) disagreed (rate {ss_rate:.3%}, bar "
            f"{MAX_SPLICE_SITE_FAILURE_RATE:.1%}) -- {verdict}:\n{listing}"
        )

    if failures or ss_fatal:
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

    raw_data = np.stack(per_exon)[None]  # (1, num_exons, delete_up_to, 4, 4)
    return DeletionAccuracyDeltaResult(raw_data=raw_data)


def alphagenome_deletion_experiment(
    model: dna_client.DnaClient,
    output_type: OutputType,
    *,
    distance_out: int,
    delete_up_to: int,
    binary_metric: bool = True,
    thresholds: dict = None,
    limit: int = None,
    interval_len: int = 131072,
    ontology_terms: Sequence[str] = ("UBERON:0001157",),
    progress: bool = True,
) -> DeletionAccuracyDeltaResult:
    """
    Full AlphaGenome deletion experiment: the AlphaGenome analogue of the CNN
    path's :func:`fac.deletion.experiment`. Loads the canonical internal coding
    exons itself (rather than taking an explicit list, as
    :func:`run_alphagenome_deletion_experiment` does) and runs every
    ``1..delete_up_to`` nt deletion around their splice sites through the model.

    The ``exp1``/``exp2`` scripts are smoke-test drivers that inline this same
    data loading but cap at a handful of exons; this runs the whole set.

    :param model: AlphaGenome client. Must be created with an explicit
        ``model_version`` so per-exon cached results don't collide across folds.
    :param output_type: ``OutputType.SPLICE_SITES`` or
        ``OutputType.SPLICE_SITE_USAGE``.
    :param distance_out: distance (nt) from each splice site at which deletions
        are placed.
    :param delete_up_to: longest deletion length to run (lengths ``1..delete_up_to``).
    :param binary_metric: if True (default), threshold each site's readout at its
        calibrated decision threshold and report the binary call-delta (mirrors
        the CNN path); if False, report the continuous ``alt - ref`` readout
        delta. See :func:`run_alphagenome_deletion_experiment`.
    :param thresholds: optional ``{"donor", "acceptor"}`` threshold override;
        when None and ``binary_metric`` is set, calibrated via
        :func:`alphagenome_calibration_thresholds`.
    :param limit: if given, run only the first ``limit`` exons (e.g. for a quick
        check); defaults to all canonical internal coding exons.
    :returns: a :class:`DeletionAccuracyDeltaResult` with ``raw_data`` of shape
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
