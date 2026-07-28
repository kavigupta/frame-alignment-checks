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
    alphagenome_calibration_thresholds,
    exon_centered_interval,
    seq_pos_to_genomic_1based,
)
from ..coding_exon import CodingExon
from ..load_data import (
    load_long_canonical_internal_coding_exons,
    load_transcript_coords,
    load_validation_gene,
)
from .alphagenome_signal_checks import (
    assert_alt_tracks_left_shifted,
    check_splice_site_signals,
    raise_for_run_failures,
    report_splice_site_disagreements,
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

# Package-local cache dirs (shipped via package_data) so results travel with the
# install. `_refalt` versions the cache: deltas_for_exon now stores ref/alt, not a
# pre-differenced delta.
_CACHE_DIR_REFALT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "alphagenome_cache_refalt",
)


@permacache(
    "frame_alignment_checks/deletion/alphagenome_deletion/run_alphagenome_deletion_experiment",
    key_function=dict(
        exons=stable_hash,
        model=lambda m: m._model_version,  # pylint: disable=protected-access
        output_type=str,
    ),
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

    Caveat: the calibration only harvests within ``harvest_radius`` of the exon
    midpoint, and P5'SS/N3'SS can sit further out than that (~12% of sites do),
    so those two columns are thresholded outside the offsets they were
    calibrated at. The exon's own 3'SS/5'SS always fall inside the radius.

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
    assert (
        binary_metric or thresholds is None
    ), "thresholds are only used by the binary metric; got binary_metric=False"
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
            per_exon.append(nan_block)
            continue
        # keep deltas regardless of disagreement; just record it below.
        per_exon.append(metric(res))
        if res["splice_site_failures"]:
            descs = "; ".join(res["splice_site_failures"])
            print(
                f"  exon {i} (gene_idx={ex.gene_idx}): splice-site disagreement - {descs}"
            )
            ss_disagreements.append((i, ex.gene_idx, descs))

    ss_rate, ss_fatal = report_splice_site_disagreements(ss_disagreements, n_placeable)
    raise_for_run_failures(failures, ss_disagreements, ss_rate, ss_fatal)

    assert per_exon, "no exons to run"
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
    ``fac.deletion.experiment``).

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
    _assert_deletion_clears_sites(
        exon, distance_out=distance_out, delete_up_to=delete_up_to
    )
    strand = gene_info["strand"]

    def seq_slice_to_ref_bases(start, end):
        return _seq_slice_to_ref_bases(gene_info, seq_idx, start, end)

    interval = exon_centered_interval(gene_info, exon, interval_len)
    _assert_interval_on_chromosome(interval, exon, gene_info, interval_len)

    variants = []
    for seq_start, seq_end in deletion_ranges_for_exon(
        exon, distance_out=distance_out, delete_up_to=delete_up_to
    ):
        ref_bases = seq_slice_to_ref_bases(seq_start, seq_end)
        # leftmost genomic coordinate of the half-open deleted span [start, end)
        pos = seq_pos_to_genomic_1based(
            gene_info, seq_start if strand == "+" else seq_end - 1
        )
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
    site_genomic = [seq_pos_to_genomic_1based(gene_info, p) for p in site_seq_positions]

    # reported, not fatal: keep the deltas; the run-level rate bar decides.
    splice_site_failures = check_splice_site_signals(
        ref_ss_0,
        site_genomic,
        site_track_idx,
    )

    assert_alt_tracks_left_shifted(
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


def _assert_deletion_clears_sites(exon, *, distance_out, delete_up_to):
    """
    No deletion may span an annotated site: the readout would index the alt track
    at a position the deletion removed. ``perform_deletion``'s "should not delete
    a boundary" assert, which this path does not go through. The exon-width check
    above only clears 3'SS/5'SS, so this is really about the flanking sites.
    """
    for start, end in deletion_ranges_for_exon(
        exon, distance_out=distance_out, delete_up_to=delete_up_to
    ):
        spanned = [i for i in exon.all_locations if start <= i < end]
        assert not spanned, (
            f"deletion [{start}, {end}) spans splice site(s) {spanned} of exon "
            f"{exon}; the flanking intron is shorter than distance_out="
            f"{distance_out}"
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
