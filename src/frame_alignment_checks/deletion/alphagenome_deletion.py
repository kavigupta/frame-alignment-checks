"""
Run small deletions around the splice sites of coding exons through the
AlphaGenome batch variant interface and return the per-site delta table as
a ``DeletionAccuracyDeltaResult``.
"""

import time
from typing import List, Sequence

import grpc
import numpy as np
import tqdm
from alphagenome.data import genome
from alphagenome.models import dna_client
from alphagenome.models.dna_output import OutputType
from permacache import permacache, stable_hash

from ..coding_exon import CodingExon
from ..load_data import load_transcript_coords, load_validation_gene
from .delete import (
    DeletionAccuracyDeltaResult,
    affected_splice_sites,
    deletion_ranges_for_exon,
    mutation_locations,
)

# donor track for donor sites, acceptor track for acceptor sites; aligned with
# ``affected_splice_sites`` = ["P5'SS", "3'SS", "5'SS", "N3'SS"].
_SITE_TRACK_TYPES = ("donor", "acceptor", "donor", "acceptor")

_COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}
_NTS = np.array(list("ACGT"))


def _predict_variants_with_retry(model, *, max_attempts=5, **kwargs):
    """
    Call ``model.predict_variants(**kwargs)`` with exponential backoff on
    ``grpc.RpcError``. Re-raises after ``max_attempts`` failures.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return model.predict_variants(**kwargs)
        except grpc.RpcError as e:
            if attempt == max_attempts:
                raise
            print(
                f"  predict_variants RpcError (attempt {attempt}/{max_attempts}): "
                f"{e.code() if hasattr(e, 'code') else e}; retrying"
            )
            time.sleep(2 ** (attempt - 1))


def check_splice_site_signals(ref_track, site_genomic, site_track_idx, *, window=50):
    """
    Sanity-check that each known splice site is the local maximum on its
    matched donor/acceptor track within ``±window`` bp. Raises ``AssertionError``
    if any in-window site fails. Sites whose genomic position falls outside the
    track interval are skipped. Uses the reference (un-variant) prediction.
    """
    track_start = ref_track.interval.start
    W = ref_track.values.shape[0]
    for sg, ti, label in zip(site_genomic, site_track_idx, affected_splice_sites):
        idx = sg - 1 - track_start
        if not 0 <= idx < W:
            continue
        lo, hi = max(0, idx - window), min(W, idx + window + 1)
        site_val = float(ref_track.values[idx, ti])
        nb = np.concatenate(
            [ref_track.values[lo:idx, ti], ref_track.values[idx + 1 : hi, ti]]
        )
        assert site_val > float(nb.max()), (
            f"splice-site sanity check failed at {label}: "
            f"value {site_val:.4f} not > neighbor max {float(nb.max()):.4f} "
            f"in window ±{window} (track {ti})"
        )


@permacache(
    "modular_splicing/frame_alignment/alphagenome_deltas_for_exon_1",
    key_function=dict(
        exon=lambda e: e.__dict__,
        # gene_info and seq_idx are derived from exon.gene_idx, but hash them
        # anyway so the key faithfully reflects the actual inputs.
        gene_info=stable_hash,
        seq_idx=stable_hash,
        # the served model identity (e.g. "ALL_FOLDS"/"FOLD_0"), or None if the
        # client was created without an explicit model_version.
        model=lambda m: getattr(m, "_model_version", None),
        output_type=lambda o: str(o),
        ontology_terms=lambda t: list(t),
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
) -> list:
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
    :returns: deltas as nested lists (JSON-serializable, so the result can be
        cached one-file-per-exon) of shape ``(delete_up_to, 4, 4)`` indexed by
        ``[deletion - 1, mutation_location, affected_splice_site]``. Wrap in
        ``np.asarray`` to get an array back.
    """
    strand = gene_info["strand"]

    def seq_slice_to_ref_bases(start, end):
        bases = "".join(_NTS[seq_idx[start:end]])
        if strand == "+":
            return bases
        return "".join(_COMP[b] for b in reversed(bases))

    def seq_pos_to_genomic_1based(pos):
        if strand == "+":
            return gene_info["hg38_start"] + pos
        return gene_info["hg38_end"] - pos

    exon_mid_0based = (
        seq_pos_to_genomic_1based((exon.acceptor + exon.donor) // 2) - 1
    )
    interval = genome.Interval(
        chromosome=gene_info["chrom"],
        start=exon_mid_0based - interval_len // 2,
        end=exon_mid_0based + interval_len // 2,
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
    track_names = list(ref_ss_0.metadata["name"])
    track_strands = list(ref_ss_0.metadata["strand"])

    site_track_idx = []
    for st_type in _SITE_TRACK_TYPES:
        for t, (tn, ts) in enumerate(zip(track_names, track_strands)):
            if ts == strand and st_type in tn.lower():
                site_track_idx.append(t)
                break
        else:
            raise ValueError(
                f"No {st_type} track found for strand {strand}; "
                f"tracks={list(zip(track_names, track_strands))}"
            )

    site_seq_positions = [
        exon.prev_donor,
        exon.acceptor,
        exon.donor,
        exon.next_acceptor,
    ]
    site_genomic = [seq_pos_to_genomic_1based(p) for p in site_seq_positions]

    check_splice_site_signals(
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
    raw = np.zeros((len(variants), 4))
    # Evidence that AlphaGenome's alt track is still left-shifted by del_len for
    # deletions -- the un-fixed behavior the idx-del_len readout corrects for
    # (issue #23, confirmed open by a maintainer). For each sharp, surviving
    # central splice peak read on the shifted branch, the shifted lookup
    # (idx-del_len) must match the reference peak better than the un-shifted
    # lookup (idx). If a future release fixes the frameshift this flips, and we
    # fail loudly here rather than silently double-correcting.
    frameshift_checks = []
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
                raw[vi, si] = np.nan
                continue
            rv = float(ref_ss.values[idx, ti])
            av = float(alt_ss.values[alt_idx, ti])
            raw[vi, si] = av - rv

            # Frameshift guard. Only the central acceptor/donor (si 1, 2) are
            # validated sharp peaks. Require a peak that is sharp at the del_len
            # scale in the reference (rv >= 2*neighbor) and that largely
            # survived the deletion, so the shifted-vs-unshifted comparison is
            # well-defined; otherwise this (vi, si) just doesn't vote.
            if shifted and si in (1, 2) and rv > 0 and idx + del_len < W:
                nb = max(
                    float(ref_ss.values[idx - del_len, ti]),
                    float(ref_ss.values[idx + del_len, ti]),
                )
                unshifted = float(alt_ss.values[idx, ti])
                if rv >= 2 * nb and max(av, unshifted) >= 0.5 * rv:
                    frameshift_checks.append(abs(av - rv) <= abs(unshifted - rv))

    # Vacuously true if no peak qualified for this exon (others will vote); a
    # real upstream fix flips essentially every qualifying peak, so it is caught.
    assert all(frameshift_checks), (
        "AlphaGenome alt track no longer appears left-shifted by del_len: the "
        "shifted readout failed to match the reference splice peak better than "
        f"the un-shifted readout in {frameshift_checks.count(False)}/"
        f"{len(frameshift_checks)} checks. If a release fixed the deletion "
        "frameshift (google-deepmind/alphagenome issue #23), drop the "
        "idx-del_len correction in deltas_for_exon."
    )

    # (delete_up_to, len(mutation_locations), len(affected_splice_sites))
    return raw.reshape(
        delete_up_to, len(mutation_locations), len(affected_splice_sites)
    ).tolist()


def run_alphagenome_deletion_experiment(
    exons: List[CodingExon],
    model: dna_client.DnaClient,
    output_type: OutputType,
    *,
    distance_out: int,
    delete_up_to: int,
    interval_len: int = 131072,
    ontology_terms: Sequence[str] = ("UBERON:0001157",),
    progress: bool = True,
) -> DeletionAccuracyDeltaResult:
    """
    Run :func:`deltas_for_exon` across a list of exons and return a
    :class:`DeletionAccuracyDeltaResult` whose ``raw_data`` has shape
    ``(1, num_exons, delete_up_to, 4, 4)`` (single seed), with the exons in the
    same order as ``exons`` (no rows dropped or reordered).

    Every exon must succeed. Exons whose prediction raises are recorded and
    processing continues to the end so that all per-exon failures are reported
    at once; if any failed, a ``RuntimeError`` is raised and no result is
    returned. (Successful per-exon predictions are cached individually by
    :func:`deltas_for_exon`, and errors are never cached, so re-running after
    fixing the cause only recomputes the failures.)
    """
    tc = load_transcript_coords()
    per_exon = []
    failures = []
    iterator = tqdm.tqdm(exons, desc="exons") if progress else exons
    for i, ex in enumerate(iterator):
        try:
            x_seq, _ = load_validation_gene(ex.gene_idx)
            per_exon.append(
                deltas_for_exon(
                    ex,
                    tc[ex.gene_idx],
                    x_seq.argmax(-1),
                    model,
                    output_type,
                    distance_out=distance_out,
                    delete_up_to=delete_up_to,
                    interval_len=interval_len,
                    ontology_terms=ontology_terms,
                )
            )
        except Exception as e:  # pylint: disable=broad-except
            print(f"  exon {i} (gene_idx={ex.gene_idx}): FAILED - {e}")
            failures.append((i, ex.gene_idx, e))

    if failures:
        summary = "\n".join(
            f"  exon {i} (gene_idx={gene_idx}): {type(e).__name__}: {e}"
            for i, gene_idx, e in failures
        )
        raise RuntimeError(
            f"{len(failures)}/{len(exons)} exon(s) failed in the AlphaGenome "
            f"deletion experiment:\n{summary}"
        ) from failures[0][2]

    raw_data = np.stack(per_exon)[None]  # (1, num_exons, delete_up_to, 4, 4)
    return DeletionAccuracyDeltaResult(raw_data=raw_data)
