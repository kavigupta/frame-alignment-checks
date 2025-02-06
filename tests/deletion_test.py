import unittest

from frame_alignment_checks.deletion import (
    accuracy_delta_given_deletion_experiment,
    basic_deletion_experiment_locations as locs,
    basic_deletion_experiment_affected_splice_sites as aff,
)
from tests.models.models import lssi_model, lssi_model_with_orf
from tests.utils import skip_on_mac


class TestDeletion(unittest.TestCase):
    @skip_on_mac
    def test_lssi_doesnt_have_any_impact(self):
        result = accuracy_delta_given_deletion_experiment(lssi_model(), distance_out=40)
        self.assertEqual((result.raw_data != 0).sum(), 0)

    @skip_on_mac
    def test_lssi_with_orf_has_impact_in_certain_contexts_only(self):
        result = accuracy_delta_given_deletion_experiment(
            lssi_model_with_orf(), distance_out=40
        )
        for num_deletions in range(1, 1 + 9):
            matr = result.mean_effect_matrix(num_deletions)
            if num_deletions % 3 == 0:
                for i in range(len(locs)):
                    for j in range(len(aff)):
                        self.assertLess(abs(matr[i, j]), 5e-2)
            else:
                self.check_matrix_non_multiple(matr)

    def check_matrix_non_multiple(self, matr):
        for i, deletion_location in enumerate(locs):
            for j, affected_splice_site in enumerate(aff):
                if deletion_location in [
                    "left of A",
                    "right of D",
                ] or affected_splice_site in ["PD", "NA"]:
                    self.assertLess(abs(matr[i, j]), 1e-2)
                else:
                    self.assertLess(matr[i, j], -5e-2)
        nearer_splice_site = (
            matr[locs.index("right of A"), aff.index("A")]
            + matr[locs.index("left of D"), aff.index("D")]
        )
        farther_splice_site = (
            matr[locs.index("right of A"), aff.index("D")]
            + matr[locs.index("left of D"), aff.index("A")]
        )
        self.assertLess(abs(farther_splice_site), abs(nearer_splice_site))
