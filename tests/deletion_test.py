import unittest

import frame_alignment_checks as fac
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
                for i in range(len(fac.deletion.mutation_locations)):
                    for j in range(len(fac.deletion.affected_splice_sites)):
                        self.assertLess(abs(matr[i, j]), 7.5e-2)
            else:
                self.check_matrix_non_multiple(matr)

    @skip_on_mac
    def test_lssi_orf_num_stops_mask(self):
        result = fac.deletion.experiment(
            lssi_model_with_orf(), distance_out=40, limit=num_exons_studied
        )
        num_rf_each = fac.deletion.num_open_reading_frames(
            distance_out=40, limit=num_exons_studied
        )
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
        mes = result.mean_effect_series("u.s. of 3'SS", "3'SS")
        self.assertEqual(mes.shape, (1, 9))
        for x in mes[0]:
            self.assertLess(abs(x), 5e-2)
        mes = result.mean_effect_series("d.s. of 3'SS", "3'SS")
        self.assertEqual(mes.shape, (1, 9))
        for num_deletions, x in enumerate(mes[0], 1):
            if num_deletions % 3 == 0:
                self.assertLess(abs(x), 5e-2)
            else:
                self.assertLess(x, -5e-2)

    def check_matrix_non_multiple(self, matr):
        for i, deletion_location in enumerate(fac.deletion.mutation_locations):
            for j, fac.deletion.affected_splice_sitesected_splice_site in enumerate(
                fac.deletion.affected_splice_sites
            ):
                if deletion_location in [
                    "u.s. of 3'SS",
                    "d.s. of 5'SS",
                ] or fac.deletion.affected_splice_sitesected_splice_site in [
                    "P5'SS",
                    "N3'SS",
                ]:
                    self.assertLess(abs(matr[i, j]), 2.5e-2)
                else:
                    self.assertLess(matr[i, j], -2e-2)
        nearer_splice_site = (
            matr[
                fac.deletion.mutation_locations.index("d.s. of 3'SS"),
                fac.deletion.affected_splice_sites.index("3'SS"),
            ]
            + matr[
                fac.deletion.mutation_locations.index("u.s. of 5'SS"),
                fac.deletion.affected_splice_sites.index("5'SS"),
            ]
        )
        farther_splice_site = (
            matr[
                fac.deletion.mutation_locations.index("d.s. of 3'SS"),
                fac.deletion.affected_splice_sites.index("5'SS"),
            ]
            + matr[
                fac.deletion.mutation_locations.index("u.s. of 5'SS"),
                fac.deletion.affected_splice_sites.index("3'SS"),
            ]
        )
        self.assertLess(abs(farther_splice_site), abs(nearer_splice_site))
