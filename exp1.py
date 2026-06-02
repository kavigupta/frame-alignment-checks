"""
Smoke test: run all 1-6nt deletions on canonical internal coding exons
using the AlphaGenome batch variant interface (predict_variants).
Averages the delta table across the first N_EXONS exons.
"""

import os

import numpy as np
from alphagenome.models import dna_client
from alphagenome.models.dna_output import OutputType

from frame_alignment_checks.deletion import affected_splice_sites, mutation_locations
from frame_alignment_checks.deletion.alphagenome_deletion import (
    run_alphagenome_deletion_experiment,
)
from frame_alignment_checks.load_data import load_long_canonical_internal_coding_exons

# ---------- config ----------
DISTANCE_OUT = 40
DELETE_UP_TO = 6
N_EXONS = 200
INTERVAL_LEN = 131072
OUTPUT_TYPE = OutputType.SPLICE_SITES

# ---------- setup ----------
with open(os.path.expanduser("~/.alphagenome")) as f:
    API_KEY = f.read().strip()

model = dna_client.create(API_KEY, model_version=dna_client.ModelVersion.ALL_FOLDS)

exons = load_long_canonical_internal_coding_exons()[:N_EXONS]

result = run_alphagenome_deletion_experiment(
    exons,
    model,
    OUTPUT_TYPE,
    distance_out=DISTANCE_OUT,
    delete_up_to=DELETE_UP_TO,
    interval_len=INTERVAL_LEN,
)

print(f"\nSuccessfully processed {result.num_exons}/{len(exons)} exons")

print()
print("Mean deltas across exons:")
header = f"{'variant':<20s}"
for sl in affected_splice_sites:
    header += f"  {sl:>10s}"
print(header)
print("-" * len(header))

for dl in range(1, DELETE_UP_TO + 1):
    # shape (4, 4) = (mutation_location, affected_splice_site)
    m = np.nanmean(result.raw_data[:, :, dl - 1], axis=(0, 1))
    for li, loc in enumerate(mutation_locations):
        row = f"{f'del{dl} {loc}':<20s}"
        for si in range(len(affected_splice_sites)):
            row += f"  {m[li, si]:+10.4f}"
        print(row)

print()
print("Done.")

# Output follows
"""
Mean deltas across exons:
variant                    P5'SS        3'SS        5'SS       N3'SS
--------------------------------------------------------------------
del1 u.s. of 3'SS        +0.0002     +0.0003     +0.0002     +0.0006
del1 d.s. of 3'SS        -0.0003     -0.0018     -0.0021     +0.0000
del1 u.s. of 5'SS        -0.0001     -0.0018     -0.0035     -0.0005
del1 d.s. of 5'SS        +0.0001     +0.0000     +0.0005     +0.0006
del2 u.s. of 3'SS        -0.0001     +0.0001     +0.0003     +0.0001
del2 d.s. of 3'SS        -0.0007     -0.0043     -0.0041     -0.0009
del2 u.s. of 5'SS        -0.0006     -0.0078     -0.0074     -0.0012
del2 d.s. of 5'SS        -0.0003     +0.0006     +0.0006     +0.0002
del3 u.s. of 3'SS        -0.0000     -0.0008     -0.0002     +0.0004
del3 d.s. of 3'SS        -0.0004     -0.0018     -0.0015     +0.0002
del3 u.s. of 5'SS        +0.0001     -0.0017     -0.0023     +0.0005
del3 d.s. of 5'SS        -0.0002     -0.0002     +0.0001     +0.0006
del4 u.s. of 3'SS        -0.0004     -0.0005     -0.0006     -0.0004
del4 d.s. of 3'SS        -0.0012     -0.0062     -0.0069     -0.0008
del4 u.s. of 5'SS        -0.0007     -0.0049     -0.0050     -0.0009
del4 d.s. of 5'SS        -0.0005     -0.0001     -0.0001     -0.0001
del5 u.s. of 3'SS        -0.0001     -0.0006     -0.0005     -0.0001
del5 d.s. of 3'SS        -0.0007     -0.0064     -0.0059     -0.0011
del5 u.s. of 5'SS        -0.0006     -0.0050     -0.0054     -0.0008
del5 d.s. of 5'SS        -0.0000     +0.0003     +0.0007     +0.0002
del6 u.s. of 3'SS        -0.0001     -0.0009     -0.0012     -0.0003
del6 d.s. of 3'SS        -0.0002     -0.0009     -0.0032     -0.0007
del6 u.s. of 5'SS        -0.0003     -0.0013     -0.0070     -0.0001
del6 d.s. of 5'SS        -0.0001     +0.0003     +0.0002     +0.0000
"""
