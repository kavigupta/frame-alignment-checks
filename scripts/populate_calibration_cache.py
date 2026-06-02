"""
Populate the AlphaGenome splice-site calibration cache.

Runs ``alphagenome_calibration_thresholds`` for each splice-site output type so
the donor/acceptor decision thresholds are computed once and cached (package-
locally, via permacache) for later reuse. Safe to re-run: cached results are
returned instantly, so only missing (model fold / output type / ontology /
window) combinations are recomputed.

Usage::

    python scripts/populate_calibration_cache.py

Requires the ``alphagenome`` extra and an API key in ``~/.alphagenome``.
"""

import os

from alphagenome.models import dna_client
from alphagenome.models.dna_output import OutputType

from frame_alignment_checks.alphagenome_calibration import (
    alphagenome_calibration_thresholds,
)

# ---------- config ----------
INTERVAL_LEN = 131072
ONTOLOGY_TERMS = ("UBERON:0001157",)
OUTPUT_TYPES = [OutputType.SPLICE_SITES, OutputType.SPLICE_SITE_USAGE]
# ALL_FOLDS so the cache key is an explicit, non-None model version.
MODEL_VERSION = dna_client.ModelVersion.ALL_FOLDS

# ---------- setup ----------
with open(os.path.expanduser("~/.alphagenome")) as f:
    API_KEY = f.read().strip()

model = dna_client.create(API_KEY, model_version=MODEL_VERSION)

for output_type in OUTPUT_TYPES:
    print(f"\n===== {output_type} =====")
    thresholds = alphagenome_calibration_thresholds(
        model,
        output_type,
        interval_len=INTERVAL_LEN,
        ontology_terms=ONTOLOGY_TERMS,
    )
    for track_type in ("donor", "acceptor"):
        print(
            f"  {track_type:>8s}: threshold {thresholds[track_type]:.6f}  "
            f"(base rate {thresholds[f'frac_{track_type}']:.3e}, "
            f"recall {thresholds[f'recall_{track_type}']:.3f})"
        )

print("\nDone.")
