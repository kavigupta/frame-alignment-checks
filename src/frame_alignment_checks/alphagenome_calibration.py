"""
Calibrate per-track-type decision thresholds for AlphaGenome's splice-site
predictions -- the AlphaGenome analogue of
``fac.models.calibration_accuracy_and_thresholds``.

alphagenome is an optional extra, so its imports are deferred and this module
loads without it.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np
import tqdm
from permacache import permacache

from .alphagenome_api import find_strand_track, predict_interval_with_retry
from .load_data import (
    load_long_canonical_internal_coding_exons,
    load_transcript_coords,
    load_validation_gene,
)

if TYPE_CHECKING:
    from alphagenome.models import dna_client
    from alphagenome.models.dna_output import OutputType

_CALIB_TRACK_TYPES = ("donor", "acceptor")

# shipped with the package, so thresholds travel with the install rather than
# living in the user's global permacache
_CACHE_DIR_CALIB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "alphagenome_calibration_cache",
)


def seq_pos_to_genomic_1based(gene_info, pos):
    """Map seq coord ``pos`` (0-based, 5'->3') to 1-based hg38."""
    if gene_info["strand"] == "+":
        return gene_info["hg38_start"] + pos
    return gene_info["hg38_end"] - pos


def _exon_mid_seq(exon):
    """Seq coord every window centred on ``exon`` is placed on."""
    return (exon.acceptor + exon.donor) // 2


def exon_centered_interval(gene_info, exon, interval_len):
    """Length-``interval_len`` interval centred on ``exon``'s midpoint."""
    from alphagenome.data import genome

    mid_0based = seq_pos_to_genomic_1based(gene_info, _exon_mid_seq(exon)) - 1
    return genome.Interval(
        chromosome=gene_info["chrom"],
        start=mid_0based - interval_len // 2,
        end=mid_0based + interval_len // 2,
    )


@permacache(
    _CACHE_DIR_CALIB,
    key_function=dict(
        # keyed on the fold so folds don't collapse onto a shared None
        model=lambda m: m._model_version,  # pylint: disable=protected-access
        output_type=str,
        ontology_terms=list,
        # affects only the tqdm bar, so keep it out of the key
        progress=lambda _: None,
    ),
    shelf_type="individual-file",
    driver="json",
)
def alphagenome_calibration_thresholds(
    model: dna_client.DnaClient,
    output_type: OutputType,
    *,
    interval_len: int = 131072,
    harvest_radius: int = 4096,
    ontology_terms: Sequence[str] = ("UBERON:0001157",),
    limit: Optional[int] = None,
    progress: bool = True,
) -> dict:
    """
    Calibrate per-track decision thresholds for AlphaGenome's splice readout (the
    analogue of :func:`fac.models.calibration_accuracy_and_thresholds`).

    Over every canonical internal coding exon, runs one ``predict_interval`` on an
    ``interval_len`` window centred on that exon and picks
    ``quantile(values, 1 - base_rate)`` per track type.

    Only positions within ``harvest_radius`` of the window centre are read out.
    AlphaGenome's per-base output depends on where in the window the base sits, so
    a threshold is only valid at the offsets it was calibrated at.

    Keyed on model fold, ``output_type``, ``ontology_terms``, ``interval_len`` and
    ``harvest_radius``, which must match wherever the thresholds are applied.

    :param model: AlphaGenome client (needs an explicit ``model_version`` so
        cached thresholds don't collide across folds).
    :param output_type: ``SPLICE_SITES``. ``SPLICE_SITE_USAGE`` is not supported:
        its tracks are per-assay, so there is no donor/acceptor track to threshold
        and track lookup raises.
    :param interval_len: AlphaGenome input window length (a supported size).
    :param harvest_radius: how far either side of the window centre to read
        positions from. Trades offset fidelity against sample size. Must be at
        least half the longest exon and at most ``interval_len // 2``; both are
        asserted.
    :param ontology_terms: ontology terms requested from the model.
    :param limit: if given, calibrate on only the first ``limit`` exons.
    :returns: dict with float ``"donor"``/``"acceptor"`` thresholds, base rates
        ``"frac_*"`` and recalls ``"recall_*"``.
    """
    assert model._model_version is not None, (  # pylint: disable=protected-access
        "model was created without an explicit model_version; pass "
        "model_version=... to dna_client.create() so cached thresholds don't "
        "collide across folds"
    )
    assert harvest_radius <= interval_len // 2, (
        f"harvest_radius={harvest_radius} exceeds half of interval_len="
        f"{interval_len}; positions past the window edge would be dropped"
    )

    tc = load_transcript_coords()
    exons = [
        ex for ex in load_long_canonical_internal_coding_exons() if ex.gene_idx in tc
    ][:limit]

    # "donor" <-> y channel 2, "acceptor" <-> channel 1 (null, acc, don).
    label_channel = {"acceptor": 1, "donor": 2}
    values = {t: [] for t in _CALIB_TRACK_TYPES}
    truth = {t: [] for t in _CALIB_TRACK_TYPES}

    y_by_gene = {}
    iterator = tqdm.tqdm(exons, desc="calibration exons") if progress else exons
    for ex in iterator:
        gene_info = tc[ex.gene_idx]
        if ex.gene_idx not in y_by_gene:
            y_by_gene[ex.gene_idx] = load_validation_gene(ex.gene_idx)[1]
        y = y_by_gene[ex.gene_idx]
        gene_len = y.shape[0]

        assert harvest_radius >= (ex.donor - ex.acceptor) // 2, (
            f"harvest_radius={harvest_radius} is under half the length of exon "
            f"{ex.acceptor}-{ex.donor} in gene {ex.gene_idx}, so that exon's own "
            f"sites would fall outside its harvest"
        )

        interval = exon_centered_interval(gene_info, ex, interval_len)
        # off the chromosome start, so unplaceable
        if interval.start < 0:
            continue

        pred = predict_interval_with_retry(
            model,
            interval=interval,
            requested_outputs=[output_type],
            ontology_terms=list(ontology_terms),
        )
        ss = pred.get(output_type)
        track_start = ss.interval.start
        W = ss.values.shape[0]
        ti = {
            t: find_strand_track(ss, t, gene_info["strand"]) for t in _CALIB_TRACK_TYPES
        }

        # the midpoint the window was placed on, so these are the offsets the
        # threshold gets applied at
        positions = np.arange(gene_len)
        positions = positions[np.abs(positions - _exon_mid_seq(ex)) <= harvest_radius]

        genomic_1based = seq_pos_to_genomic_1based(gene_info, positions)
        idx = genomic_1based - 1 - track_start
        in_bounds = (idx >= 0) & (idx < W)
        idx_ib = idx[in_bounds]
        seq_ib = positions[in_bounds]
        for t in _CALIB_TRACK_TYPES:
            values[t].append(ss.values[idx_ib, ti[t]])
            truth[t].append((y[seq_ib, label_channel[t]] > 0.5).astype(np.float64))

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


# y's label channels are (null, acceptor, donor), matching
# ModelToAnalyze.thresholds; the dict's own order is donor-first.
_THRESHOLD_ORDER = ("acceptor", "donor")


def alphagenome_calibration_accuracy_and_thresholds(
    model: dna_client.DnaClient,
    output_type: OutputType,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    AlphaGenome analogue of
    :func:`fac.models.calibration_accuracy_and_thresholds`, returning the same
    ``(acc, thresholds)`` shape so AlphaGenome calibration drops into the same
    downstream machinery as the CNN models.

    This is the intended public entry point: it wraps the cached
    :func:`alphagenome_calibration_thresholds` and reshapes its dict into two
    length-2 arrays ordered ``[acceptor, donor]``.

    :param model: AlphaGenome client (see
        :func:`alphagenome_calibration_thresholds`).
    :param output_type: ``OutputType.SPLICE_SITES``; see
        :func:`alphagenome_calibration_thresholds` for why
        ``SPLICE_SITE_USAGE`` is not supported.
    :param kwargs: forwarded verbatim to
        :func:`alphagenome_calibration_thresholds` (``interval_len``,
        ``harvest_radius``, ``ontology_terms``, ``limit``, ``progress``).
    :returns: ``(acc, thresholds)``, each an ``np.ndarray`` of shape ``(2,)`` in
        ``[acceptor, donor]`` order. ``acc`` is the per-channel recall at the
        chosen threshold; ``thresholds`` are in the raw ``output_type`` value
        scale (not softmax probabilities, unlike the CNN thresholds), so they
        apply only to that same ``output_type``.
    """
    raw = alphagenome_calibration_thresholds(model, output_type, **kwargs)
    acc = np.array([raw[f"recall_{t}"] for t in _THRESHOLD_ORDER])
    thresholds = np.array([raw[t] for t in _THRESHOLD_ORDER])
    return acc, thresholds
