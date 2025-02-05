import unittest

import numpy as np
from parameterized import parameterized

from frame_alignment_checks.utils import bootstrap_series, parse_sequence_as_one_hot


class ParseSequenceAsOneHotTest(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(parse_sequence_as_one_hot("A").tolist(), [[1, 0, 0, 0]])
        self.assertEqual(
            parse_sequence_as_one_hot("AGAGGGATNAN").tolist(),
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        )

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            parse_sequence_as_one_hot("ABCDEF")


class BootstrapSeriesTest(unittest.TestCase):
    @parameterized.expand([(i,) for i in range(100)])
    def test_independence(self, seed):
        rng = np.random.RandomState(seed)
        count = rng.randint(10, 100)
        num_series = rng.randint(1, 10)
        xs_bank = rng.normal(size=(count, num_series))
        lo_all, hi_all = bootstrap_series(xs_bank)
        self.assertEqual(lo_all.shape, (num_series,))
        self.assertEqual(hi_all.shape, (num_series,))
        subset = rng.choice(
            num_series, size=rng.randint(1, num_series + 1), replace=True
        )
        bootstrap_subset = bootstrap_series(xs_bank[:, subset])
        self.assertTrue(np.allclose(bootstrap_subset, (lo_all[subset], hi_all[subset])))
