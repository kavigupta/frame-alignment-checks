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
OUTPUT_TYPE = OutputType.SPLICE_SITE_USAGE
# When True, threshold each site's ref/alt readout at its calibrated
# donor/acceptor decision threshold and report the binary call-delta in
# {-1, 0, +1}; the thresholds are calibrated once (and cached) via
# alphagenome_calibration_thresholds. Set False for the raw continuous
# alt - ref readout delta.
BINARY_METRIC = True

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
    binary_metric=BINARY_METRIC,
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
    m = np.nanmean(result.raw_data[:, :, dl - 1], axis=(0, 1))
    for li, loc in enumerate(mutation_locations):
        row = f"{f'del{dl} {loc}':<20s}"
        for si in range(len(affected_splice_sites)):
            row += f"  {m[li, si]:+10.4f}"
        print(row)

print()
print("Done.")

# Output follows (captured with BINARY_METRIC = False, the continuous
# alt - ref readout delta; the binary call-delta will differ).
"""
Mean deltas across exons:
variant                    P5'SS        3'SS        5'SS       N3'SS
--------------------------------------------------------------------
del1 u.s. of 3'SS        -0.0001     -0.0020     -0.0004     -0.0001
del1 d.s. of 3'SS        -0.0011     -0.0047     -0.0044     -0.0010
del1 u.s. of 5'SS        -0.0014     -0.0066     -0.0074     -0.0018
del1 d.s. of 5'SS        -0.0001     -0.0016     -0.0016     -0.0001
del2 u.s. of 3'SS        -0.0007     -0.0038     -0.0010     -0.0014
del2 d.s. of 3'SS        -0.0013     -0.0060     -0.0045     -0.0018
del2 u.s. of 5'SS        -0.0012     -0.0130     -0.0166     -0.0024
del2 d.s. of 5'SS        -0.0002     -0.0005     -0.0004     -0.0014
del3 u.s. of 3'SS        +0.0001     -0.0039     -0.0014     -0.0001
del3 d.s. of 3'SS        -0.0002     -0.0030     -0.0030     +0.0002
del3 u.s. of 5'SS        -0.0006     -0.0054     -0.0099     -0.0002
del3 d.s. of 5'SS        +0.0005     -0.0012     -0.0013     +0.0004
del4 u.s. of 3'SS        -0.0003     -0.0026     -0.0007     -0.0004
del4 d.s. of 3'SS        -0.0009     -0.0079     -0.0086     -0.0007
del4 u.s. of 5'SS        -0.0020     -0.0081     -0.0108     -0.0017
del4 d.s. of 5'SS        +0.0001     -0.0006     -0.0007     +0.0002
del5 u.s. of 3'SS        -0.0001     -0.0024     -0.0011     +0.0009
del5 d.s. of 3'SS        -0.0002     -0.0093     -0.0074     +0.0001
del5 u.s. of 5'SS        -0.0012     -0.0076     -0.0109     -0.0003
del5 d.s. of 5'SS        +0.0001     -0.0006     -0.0002     +0.0013
del6 u.s. of 3'SS        -0.0009     -0.0031     -0.0011     -0.0003
del6 d.s. of 3'SS        -0.0008     -0.0022     -0.0035     -0.0002
del6 u.s. of 5'SS        -0.0025     -0.0032     -0.0092     -0.0007
del6 d.s. of 5'SS        -0.0006     +0.0001     +0.0004     +0.0003
"""
