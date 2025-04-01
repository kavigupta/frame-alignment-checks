import unittest
from collections import defaultdict

import numpy as np

import frame_alignment_checks as fac
from tests.models.models import lssi_model, lssi_model_with_orf
from tests.utils import skip_on_mac

num_pairs_studied = 500


class TestDeletion(unittest.TestCase):

    @skip_on_mac
    def test_lssi_doesnt_have_any_impact(self):
        result = fac.deletion.run_on_all_adjacent_deletions(
            lssi_model(), limit=num_pairs_studied
        )
        self.assertEqual(
            result.shape, (num_pairs_studied, len(fac.deletion.conditions), 4)
        )
        err = result != result[:, [fac.deletion.conditions.index((0, 0))]]
        self.assertEqual(err.sum(), 0, str(np.where(err)))

    @skip_on_mac
    def test_lssi_orf_has_independent_impact(self):
        result = fac.deletion.run_on_all_adjacent_deletions(
            lssi_model_with_orf(), limit=num_pairs_studied
        )
        self.assertEqual(
            result.shape, (num_pairs_studied, len(fac.deletion.conditions), 4)
        )
        result = result.sum(0).reshape((len(fac.deletion.conditions), 2, 2))
        # result_orig = result[fac.deletion.conditions.index((0, 0))]
        by_condition_first = defaultdict(list)
        by_condition_second = defaultdict(list)
        for i, (first, second) in enumerate(fac.deletion.conditions):
            by_condition_first[first].append(result[i][0])
            by_condition_second[second].append(result[i][1])
        self.assert_all_same_by_condition(by_condition_first)
        self.assert_all_same_by_condition(by_condition_second)

    def assert_all_same_by_condition(self, by_condition):
        for condition, values in by_condition.items():
            for val in values:
                self.assertEqual(
                    val.tolist(),
                    values[0].tolist(),
                    f"Condition {condition} has different values: {values}",
                )
