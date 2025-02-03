import unittest

from frame_alignment_checks.deletion import perform_deletion
from parameterized import parameterized
import numpy as np


class PerformDeletionTest(unittest.TestCase):
    @parameterized.expand([(seed,) for seed in range(1000)])
    def test_deletion_fuzz(self, seed):
        seed = 0
        rng = np.random.RandomState(seed)
        sequence = np.arange(1000)
        delete_start, delete_end = sorted(rng.choice(1000, 2, replace=False))
        valid_locations = list(range(delete_start)) + list(range(delete_end, 1000))
        indices = sorted(rng.choice(valid_locations, 4, replace=False))
        skip_amount = rng.choice([1, 2, 3, 4, 5])
        print(
            f"delete_start={delete_start}, delete_end={delete_end}, indices={indices}, skip_amount={skip_amount}"
        )
        sequence_updated, _, indices_updated = perform_deletion(
            sequence,
            deletion_range=(delete_start, delete_end),
            indices=tuple(indices),
            repair=lambda x: (x[::skip_amount], None),
        )
        if skip_amount == 1:
            self.assertEqual(
                len(sequence_updated), len(sequence) - (delete_end - delete_start)
            )
        self.assertEqual(
            sequence_updated[
                [
                    indices_updated[0],
                    indices_updated[1],
                    indices_updated[2] + 1,
                    indices_updated[3],
                ]
            ].tolist(),
            sequence[
                [
                    indices[0],
                    indices[1],
                    indices[2] + 1,
                    indices[3],
                ]
            ].tolist(),
        )
