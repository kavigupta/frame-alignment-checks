"""
Calibrate per-track-type decision thresholds for AlphaGenome's splice-site
predictions -- the AlphaGenome analogue of
``fac.models.calibration_accuracy_and_thresholds``.

AlphaGenome requires Python >=3.10 and is an optional extra; the runtime imports
live inside the function that uses it (and ``from __future__ import annotations``
keeps the type-only imports lazy), so this module imports fine without it.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Sequence, Tuple

import numpy as np
import tqdm
from permacache import permacache

from .load_data import (
    load_long_canonical_internal_coding_exons,
    load_transcript_coords,
    load_validation_gene,
)

if TYPE_CHECKING:
    from alphagenome.models import dna_client
    from alphagenome.models.dna_output import OutputType

# Attempts before giving up on transient grpc RpcErrors in predict_interval.
PREDICT_MAX_ATTEMPTS = 5

# Splice-site track types calibrated, aligned with load_validation_gene's label
# channels: "acceptor" <-> y[:, 1], "donor" <-> y[:, 2].
_CALIB_TRACK_TYPES = ("donor", "acceptor")

# Package-local cache directory (shipped with the package via package_data), so
# precomputed thresholds travel with the install instead of living in the user's
# global permacache. Resolved from __file__ so it works wherever installed; an
# absolute path overrides permacache's default base.
_CACHE_DIR_CALIB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
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


def _predict_interval_with_retry(model, **kwargs):
    """``model.predict_interval(**kwargs)`` with grpc retry (see :func:`_with_rpc_retry`)."""
    return _with_rpc_retry(lambda: model.predict_interval(**kwargs), "predict_interval")


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


@permacache(
    _CACHE_DIR_CALIB,
    key_function=dict(
        # the served model identity (e.g. "ALL_FOLDS"/"FOLD_0"); the client must
        # have been created with an explicit model_version so the cache key
        # distinguishes folds rather than collapsing them onto a shared None.
        model=lambda m: m._model_version,  # pylint: disable=protected-access
        output_type=str,
        ontology_terms=list,
        # progress only controls the tqdm bar, not the result, so keep it out of
        # the cache key (collapse to None) to avoid fragmenting the cache across
        # progress=True/False calls.
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
    gene**. This deliberately avoids sizing the window to the gene: a window only
    marginally larger than the gene would push its splice sites to the window
    edges, where AlphaGenome's predictions are least reliable, biasing the
    calibrated distribution. Genes longer than ``interval_len`` (a small
    minority) contribute only their central ``interval_len`` of positions; the
    dropped tails are exactly the positions that would otherwise sit near a
    window edge, so the distributional threshold is unaffected.

    Thresholds are output-type-specific, ontology-specific, and window-specific,
    so the cache is keyed on ``output_type``, ``ontology_terms`` and
    ``interval_len`` alongside the model fold. The same ``ontology_terms`` and
    ``interval_len`` must be used here and wherever the thresholds are applied,
    for them to match the values they're thresholding.

    :param model: AlphaGenome client. Must be created with an explicit
        ``model_version`` so cached thresholds don't collide across folds.
    :param output_type: ``OutputType.SPLICE_SITES`` -- the per-base donor/
        acceptor probability tracks. ``SPLICE_SITE_USAGE`` is not supported: its
        tracks are per-assay, not split into donor/acceptor, so there is no
        donor/acceptor track to threshold (calibration would raise looking for
        one).
    :param interval_len: AlphaGenome input window length (a supported size).
    :param ontology_terms: ontology terms requested from the model.
    :param limit: if given, calibrate on only the first ``limit`` genes.
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
    gene_idxs = sorted(
        {ex.gene_idx for ex in load_long_canonical_internal_coding_exons()}
    )
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
        # pylint (astroid numpy brain) mis-infers y as np.array itself, so it
        # flags .shape (no-member) and y[...] (unsubscriptable-object) below; y
        # is the actual label ndarray from load_validation_gene.
        _, y = load_validation_gene(gene_idx)
        gene_len = y.shape[0]  # pylint: disable=no-member

        # 1-based genomic midpoint of the gene, matching the seq->genomic
        # mapping below (hg38_start/hg38_end are the gene's first/last base). Only
        # used to centre the window, so a 1-base offset is immaterial here.
        mid_genomic_1based = (gene_info["hg38_start"] + gene_info["hg38_end"]) // 2
        start = max(0, mid_genomic_1based - interval_len // 2)
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
        ti = {
            t: _find_strand_track(ss, t, gene_info["strand"])
            for t in _CALIB_TRACK_TYPES
        }

        # genomic 1-based position of each seq index, vectorised. The gene
        # sequence spans exactly [hg38_start, hg38_end], so seq position 0 is the
        # genomically-leftmost base on + strand and the rightmost on -.
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
            # y is mis-inferred as np.array (see note above); allow the subscript.
            # pylint: disable=unsubscriptable-object
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


# Channel order shared with the CNN framework: y's label channels are
# (null, acceptor, donor), so the per-non-null-channel arrays returned below --
# and ModelToAnalyze.thresholds -- run [acceptor, donor].
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

    This is the intended public entry point: it wraps the lower-level, cached
    :func:`alphagenome_calibration_thresholds` (which returns a richer JSON dict
    keyed by track-type name, plus base rates) and reshapes it into two length-2
    arrays ordered ``[acceptor, donor]`` -- matching ``y``'s non-null label
    channels and :attr:`fac.models.ModelToAnalyze.thresholds` (one threshold per
    non-null channel, acceptor then donor). Build the array via this explicit
    order rather than the dict's insertion order, which is donor-first.

    :param model: AlphaGenome client (see
        :func:`alphagenome_calibration_thresholds`).
    :param output_type: ``OutputType.SPLICE_SITES`` or
        ``OutputType.SPLICE_SITE_USAGE``.
    :param kwargs: forwarded verbatim to
        :func:`alphagenome_calibration_thresholds` (``interval_len``,
        ``ontology_terms``, ``limit``, ``progress``).
    :returns: ``(acc, thresholds)``, each an ``np.ndarray`` of shape ``(2,)`` in
        ``[acceptor, donor]`` order. ``acc`` is the per-channel recall at the
        chosen threshold; ``thresholds`` are the per-channel decision thresholds
        in the raw ``output_type`` value scale (not softmax probabilities, unlike
        the CNN thresholds), so they apply only to that same ``output_type``.
    """
    raw = alphagenome_calibration_thresholds(model, output_type, **kwargs)
    acc = np.array([raw[f"recall_{t}"] for t in _THRESHOLD_ORDER])
    thresholds = np.array([raw[t] for t in _THRESHOLD_ORDER])
    return acc, thresholds
