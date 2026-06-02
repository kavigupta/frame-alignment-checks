"""
Full AlphaGenome deletion experiment: run all 1-6nt deletions around the splice
sites of every canonical internal coding exon, for both splice-site output
types, and print the mean delta table for each.

Unlike the exp1/exp2 smoke tests, this loads the whole canonical exon set (via
``fac.deletion.alphagenome_deletion_experiment``) rather than a small slice.

Reports the calibrated binary call-delta (the default ``binary_metric=True``,
the analogue of the CNN ``fac.deletion.experiment``). The first run per output
type calibrates donor/acceptor thresholds over the validation genes (cached
thereafter); pass ``binary_metric=False`` for the raw continuous readout delta.
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
OUTPUT_TYPES = [OutputType.SPLICE_SITES, OutputType.SPLICE_SITE_USAGE]

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


for output_type in OUTPUT_TYPES:
    print(f"\n===== {output_type} =====")
    result = alphagenome_deletion_experiment(
        model,
        output_type,
        distance_out=DISTANCE_OUT,
        delete_up_to=DELETE_UP_TO,
        interval_len=INTERVAL_LEN,
    )
    print(f"Successfully processed {result.num_exons} exons\n")
    print_mean_deltas(result)

print()
print("Done.")
