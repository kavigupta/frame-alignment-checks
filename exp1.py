"""
Smoke test: run all 1-9nt deletions on canonical internal coding exons
using the AlphaGenome batch variant interface (predict_variants).
Averages the delta table across the first N_EXONS exons.
"""

import os

import numpy as np
import tqdm
from alphagenome.data import genome
from alphagenome.models import dna_client

from frame_alignment_checks.load_data import (
    load_long_canonical_internal_coding_exons,
    load_transcript_coords,
    load_validation_gene,
)

# ---------- config ----------
DISTANCE_OUT = 40  # how far from splice site to place deletions
DELETE_UP_TO = 6  # max deletion length
N_EXONS = 200  # number of exons to average over
INTERVAL_LEN = 131072

# ---------- setup ----------
with open(os.path.expanduser("~/.alphagenome")) as f:
    API_KEY = f.read().strip()

model = dna_client.create(API_KEY)

exons = load_long_canonical_internal_coding_exons()[:N_EXONS]
tc = load_transcript_coords()

COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}
NTS = np.array(list("ACGT"))

location_labels = [
    "u.s. of 3'SS",
    "d.s. of 3'SS",
    "u.s. of 5'SS",
    "d.s. of 5'SS",
]
site_labels = ["P5'SS", "3'SS", "5'SS", "N3'SS"]
# donor track for donor sites, acceptor track for acceptor sites
site_track_types = ["donor", "acceptor", "donor", "acceptor"]

# Variant labels: one per deletion_len × location
variant_labels = [
    f"del{dl} {loc}" for dl in range(1, DELETE_UP_TO + 1) for loc in location_labels
]
n_variants = len(variant_labels)

# Accumulator: (n_variants, 4 sites)
all_deltas = []


def process_exon(ex):
    """Build variants for one exon, run prediction, return deltas (n_variants, 4)."""
    gene_info = tc[ex.gene_idx]
    strand = gene_info["strand"]
    x_seq, _ = load_validation_gene(ex.gene_idx)
    seq_idx = x_seq.argmax(-1)

    # --- coordinate helpers (closed over gene_info, strand, seq_idx) ---
    def seq_slice_to_ref_bases(start, end):
        bases = "".join(NTS[seq_idx[start:end]])
        if strand == "+":
            return bases
        return "".join(COMP[b] for b in reversed(bases))

    def seq_pos_to_genomic_1based(pos):
        if strand == "+":
            return gene_info["hg38_start"] + pos
        return gene_info["hg38_end"] - pos

    def deletion_variant_position_1based(start, end):
        if strand == "+":
            return gene_info["hg38_start"] + start
        return gene_info["hg38_end"] - end + 1

    # --- interval ---
    exon_mid_0based = seq_pos_to_genomic_1based((ex.acceptor + ex.donor) // 2) - 1
    interval = genome.Interval(
        chromosome=gene_info["chrom"],
        start=exon_mid_0based - INTERVAL_LEN // 2,
        end=exon_mid_0based + INTERVAL_LEN // 2,
    )

    # --- build variants ---
    variants = []
    for delete_len in range(1, DELETE_UP_TO + 1):
        deletion_specs = [
            (ex.acceptor - DISTANCE_OUT - delete_len, ex.acceptor - DISTANCE_OUT),
            (
                ex.acceptor + DISTANCE_OUT + 1,
                ex.acceptor + DISTANCE_OUT + 1 + delete_len,
            ),
            (ex.donor - DISTANCE_OUT - delete_len, ex.donor - DISTANCE_OUT),
            (ex.donor + DISTANCE_OUT + 1, ex.donor + DISTANCE_OUT + 1 + delete_len),
        ]
        for seq_start, seq_end in deletion_specs:
            ref_bases = seq_slice_to_ref_bases(seq_start, seq_end)
            pos = deletion_variant_position_1based(seq_start, seq_end)
            variants.append(
                genome.Variant(
                    chromosome=gene_info["chrom"],
                    position=pos,
                    reference_bases=ref_bases,
                    alternate_bases="",
                )
            )

    # --- run batch prediction ---
    variant_outputs = model.predict_variants(
        intervals=interval,
        variants=variants,
        ontology_terms=["UBERON:0001157"],
        requested_outputs=[dna_client.OutputType.SPLICE_SITES],
        progress_bar=False,
    )

    # --- discover track indices from first output ---
    ref_ss_0 = variant_outputs[0].reference.splice_sites
    track_names = list(ref_ss_0.metadata["name"])
    track_strands = list(ref_ss_0.metadata["strand"])
    gene_strand = gene_info["strand"]

    site_track_idx = []
    for st_type in site_track_types:
        for t, (tn, ts) in enumerate(zip(track_names, track_strands)):
            if ts == gene_strand and st_type in tn.lower():
                site_track_idx.append(t)
                break
        else:
            for t, ts in enumerate(track_strands):
                if ts == gene_strand:
                    site_track_idx.append(t)
                    break

    # --- splice site genomic positions ---
    site_seq_positions = [ex.prev_donor, ex.acceptor, ex.donor, ex.next_acceptor]
    site_genomic = [seq_pos_to_genomic_1based(p) for p in site_seq_positions]

    # --- extract deltas ---
    deltas = np.zeros((n_variants, 4))
    for vi, (vo, v) in enumerate(zip(variant_outputs, variants)):
        ref_ss = vo.reference.splice_sites
        alt_ss = vo.alternate.splice_sites
        track_start = ref_ss.interval.start

        del_len = len(v.reference_bases)
        del_end_0based = v.position - 1 + del_len

        for si, (sg, ti) in enumerate(zip(site_genomic, site_track_idx)):
            idx = sg - 1 - track_start
            rv = float(ref_ss.values[idx, ti])
            alt_idx = idx - del_len if (sg - 1) >= del_end_0based else idx
            av = float(alt_ss.values[alt_idx, ti])
            deltas[vi, si] = av - rv

    return deltas


# ---------- main loop ----------
print(
    f"Running deletion experiment on {len(exons)} exons, "
    f"{n_variants} variants each ..."
)

for i, ex in enumerate(tqdm.tqdm(exons, desc="exons")):
    try:
        deltas = process_exon(ex)
        all_deltas.append(deltas)
    except Exception as e:
        print(f"  exon {i} (gene_idx={ex.gene_idx}): FAILED - {e}")

all_deltas = np.stack(all_deltas)  # (n_exons_ok, n_variants, 4)
print(f"\nSuccessfully processed {all_deltas.shape[0]}/{len(exons)} exons")

# ---------- average delta table ----------
mean_deltas = all_deltas.mean(axis=0)  # (n_variants, 4)

print()
print("Mean deltas across exons:")
header = f"{'variant':<20s}"
for sl in site_labels:
    header += f"  {sl:>10s}"
print(header)
print("-" * len(header))

for vi, label in enumerate(variant_labels):
    row = f"{label:<20s}"
    for si in range(4):
        row += f"  {mean_deltas[vi, si]:+10.4f}"
    print(row)

print()
print("Done.")

# Output follows
"""
Mean deltas across exons:
variant                    P5'SS        3'SS        5'SS       N3'SS
--------------------------------------------------------------------
del1 u.s. of 3'SS        +0.0004     +0.0003     +0.0002     +0.0006
del1 d.s. of 3'SS        -0.0003     -0.0018     -0.0021     +0.0001
del1 u.s. of 5'SS        -0.0001     -0.0019     -0.0035     -0.0004
del1 d.s. of 5'SS        +0.0003     +0.0002     +0.0006     +0.0005
del2 u.s. of 3'SS        -0.0001     +0.0000     +0.0002     -0.0001
del2 d.s. of 3'SS        -0.0007     -0.0043     -0.0041     -0.0008
del2 u.s. of 5'SS        -0.0004     -0.0078     -0.0075     -0.0012
del2 d.s. of 5'SS        -0.0001     +0.0006     +0.0004     +0.0000
del3 u.s. of 3'SS        -0.0001     -0.0007     -0.0002     +0.0005
del3 d.s. of 3'SS        -0.0003     -0.0017     -0.0015     +0.0002
del3 u.s. of 5'SS        +0.0001     -0.0017     -0.0023     +0.0005
del3 d.s. of 5'SS        -0.0001     -0.0001     +0.0002     +0.0008
del4 u.s. of 3'SS        -0.0003     -0.0005     -0.0007     -0.0002
del4 d.s. of 3'SS        -0.0010     -0.0063     -0.0070     -0.0009
del4 u.s. of 5'SS        -0.0006     -0.0047     -0.0050     -0.0010
del4 d.s. of 5'SS        -0.0004     -0.0001     +0.0000     -0.0003
del5 u.s. of 3'SS        -0.0001     -0.0004     -0.0005     -0.0001
del5 d.s. of 3'SS        -0.0006     -0.0065     -0.0060     -0.0010
del5 u.s. of 5'SS        -0.0006     -0.0050     -0.0054     -0.0007
del5 d.s. of 5'SS        +0.0000     +0.0003     +0.0006     +0.0003
del6 u.s. of 3'SS        -0.0002     -0.0009     -0.0011     -0.0002
del6 d.s. of 3'SS        -0.0002     -0.0008     -0.0033     -0.0008
del6 u.s. of 5'SS        -0.0002     -0.0014     -0.0070     -0.0002
del6 d.s. of 5'SS        -0.0001     +0.0003     +0.0003     +0.0000
"""
