"""
Run the full AlphaGenome deletion experiment: every 1-9 nt deletion around the
splice sites of all canonical internal coding exons. This populates the per-exon
cache and returns a ``DeletionAccuracyDeltaResult`` -- the same column type the
CNN path (``fac.deletion.experiment``) produces.

Presentation is left to a notebook: load this result and drop it into the
``deltas_by_model`` dict next to the CNN model columns, then render with
``fac.deletion.plot_by_deletion_loc_and_affected_site`` / ``plot_matrix_at_site``.
``distance_out`` and ``delete_up_to`` match the CNN deletion experiment so the
columns line up.
"""

import os

from alphagenome.models import dna_client
from alphagenome.models.dna_output import OutputType

from frame_alignment_checks.deletion import alphagenome_deletion_experiment

DISTANCE_OUT = 40
DELETE_UP_TO = 9
OUTPUT_TYPE = OutputType.SPLICE_SITES

with open(os.path.expanduser("~/.alphagenome")) as f:
    API_KEY = f.read().strip()

model = dna_client.create(API_KEY, model_version=dna_client.ModelVersion.ALL_FOLDS)

result = alphagenome_deletion_experiment(
    model, OUTPUT_TYPE, distance_out=DISTANCE_OUT, delete_up_to=DELETE_UP_TO
)
print(f"Done. Processed {result.num_exons} exons (cached per exon).")
