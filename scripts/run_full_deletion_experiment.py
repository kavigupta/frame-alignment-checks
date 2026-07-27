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
