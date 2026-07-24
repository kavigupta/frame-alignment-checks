"""
Full AlphaGenome deletion experiment: run all 1-6nt deletions around the splice
sites of every canonical internal coding exon and print the mean delta table.

Unlike the exp1 smoke test, this loads the whole canonical exon set (via
``fac.deletion.alphagenome_deletion_experiment``) rather than a small slice.

Reports the calibrated binary call-delta (the default ``binary_metric=True``,
the analogue of the CNN ``fac.deletion.experiment``). The first run calibrates
donor/acceptor thresholds over the validation genes (cached thereafter); pass
``binary_metric=False`` for the raw continuous readout delta.
"""

import os

import numpy as np
from alphagenome.models import dna_client
from alphagenome.models.dna_output import OutputType

from frame_alignment_checks.deletion import (
    affected_splice_sites,
    alphagenome_deletion_experiment,
    mutation_locations,
)

# ---------- config ----------
DISTANCE_OUT = 40
DELETE_UP_TO = 6
INTERVAL_LEN = 131072
# SPLICE_SITE_USAGE has no donor/acceptor-typed track, so neither the readout
# nor the calibration applies to it.
OUTPUT_TYPE = OutputType.SPLICE_SITES
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


def print_mean_deltas(result):
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


result = alphagenome_deletion_experiment(
    model,
    OUTPUT_TYPE,
    distance_out=DISTANCE_OUT,
    delete_up_to=DELETE_UP_TO,
    binary_metric=BINARY_METRIC,
    interval_len=INTERVAL_LEN,
)
print(f"Successfully processed {result.num_exons} exons\n")
print_mean_deltas(result)

print()
print("Done.")
