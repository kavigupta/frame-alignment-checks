import unittest

import frame_alignment_checks as fac
from frame_alignment_checks.deletion import (
    basic_deletion_experiment_affected_splice_sites as aff,
)
from frame_alignment_checks.deletion import basic_deletion_experiment_locations as locs
from frame_alignment_checks.deletion import num_open_reading_frames
from tests.models.models import lssi_model, lssi_model_with_orf
from tests.utils import skip_on_mac

num_exons_studied = 500


class TestDeletion(unittest.TestCase):
    @skip_on_mac
    def test_lssi_doesnt_have_any_impact(self):
        result = fac.deletion.experiment(
            lssi_model(), distance_out=40, limit=num_exons_studied
        )
        self.assertEqual((result.raw_data != 0).sum(), 0)

    @skip_on_mac
    def test_lssi_with_orf_has_impact_in_certain_contexts_only(self):
        result = fac.deletion.experiment(
            lssi_model_with_orf(), distance_out=40, limit=num_exons_studied
        )
        for num_deletions in range(1, 1 + 9):
            matr = result.mean_effect_matrix(num_deletions)
            print(num_deletions)
            print(matr)
            self.assertEqual(matr.shape, (4, 4))
            if num_deletions % 3 == 0:
                for i in range(len(locs)):
                    for j in range(len(aff)):
                        self.assertLess(abs(matr[i, j]), 7.5e-2)
            else:
                self.check_matrix_non_multiple(matr)

    @skip_on_mac
    def test_lssi_orf_num_stops_mask(self):
        result = fac.deletion.experiment(
            lssi_model_with_orf(), distance_out=40, limit=num_exons_studied
        )
        num_rf_each = num_open_reading_frames(distance_out=40, limit=num_exons_studied)
        for num_rf in range(1 + 3):
            mem = result.mean_effect_masked(num_rf_each == num_rf)
            self.assertEqual(mem.shape, (1, 9))
            [mem] = mem
            print(num_rf, mem)
            for x in mem:
                if num_rf == 0:
                    # does not matter whether its a multiple of 3
                    # always a large effect
                    self.assertLess(x, -9e-2)
                else:
                    # always a small effect
                    # not necessarily zero
                    self.assertLess(abs(x), 10e-2)

    @skip_on_mac
    def test_lssi_orf_num_stops_series(self):
        result = fac.deletion.experiment(
            lssi_model_with_orf(), distance_out=40, limit=num_exons_studied
        )
        mes = result.mean_effect_series("left of A", "A")
        self.assertEqual(mes.shape, (1, 9))
        for x in mes[0]:
            self.assertLess(abs(x), 5e-2)
        mes = result.mean_effect_series("right of A", "A")
        self.assertEqual(mes.shape, (1, 9))
        for num_deletions, x in enumerate(mes[0], 1):
            if num_deletions % 3 == 0:
                self.assertLess(abs(x), 5e-2)
            else:
                self.assertLess(x, -5e-2)

    def check_matrix_non_multiple(self, matr):
        for i, deletion_location in enumerate(locs):
            for j, affected_splice_site in enumerate(aff):
                if deletion_location in [
                    "left of A",
                    "right of D",
                ] or affected_splice_site in ["PD", "NA"]:
                    self.assertLess(abs(matr[i, j]), 2.5e-2)
                else:
                    self.assertLess(matr[i, j], -2e-2)
        nearer_splice_site = (
            matr[locs.index("right of A"), aff.index("A")]
            + matr[locs.index("left of D"), aff.index("D")]
        )
        farther_splice_site = (
            matr[locs.index("right of A"), aff.index("D")]
            + matr[locs.index("left of D"), aff.index("A")]
        )
        self.assertLess(abs(farther_splice_site), abs(nearer_splice_site))
