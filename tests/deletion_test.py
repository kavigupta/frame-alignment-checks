import unittest

from frame_alignment_checks.deletion import accuracy_delta_given_deletion_experiment
from tests.models.models import lssi_model


class TestDeletion(unittest.TestCase):
    def test_lssi_doesnt_have_any_impact(self):
        result = accuracy_delta_given_deletion_experiment(lssi_model(), distance_out=40)
        self.assertEqual((result.raw_data != 0).sum(), 0)
