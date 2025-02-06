import unittest

from dconstruct import construct
import numpy as np
from run_batched import run_batched
import tqdm
from frame_alignment_checks.data.load import (
    load_long_canonical_internal_coding_exons,
    load_validation_gene,
)
from frame_alignment_checks.deletion import (
    accuracy_delta_given_deletion_experiment,
    basic_deletion_experiment,
    deletion_experiment,
)
from frame_alignment_checks.deletion_num_stops import num_open_reading_frames
from frame_alignment_checks.deletion_repair import repair_strategy_types
from frame_alignment_checks.utils import collect_windows, device_of, extract_center, stable_hash_cached
from tests.models.models import lssi_model, lssi_model_with_orf
from tests.utils import skip_on_mac


class TestDeletion(unittest.TestCase):
    @skip_on_mac
    def test_lssi_doesnt_have_any_impact(self):
        result = accuracy_delta_given_deletion_experiment(lssi_model(), distance_out=40)
        self.assertEqual((result.raw_data != 0).sum(), 0)


lm = lssi_model()
result = accuracy_delta_given_deletion_experiment(
    lssi_model_with_orf(), distance_out=40
)
num_frames_open = num_open_reading_frames(distance_out=40)
result_old = accuracy_delta_given_deletion_experiment(lm, distance_out=40)

import IPython

IPython.embed()
