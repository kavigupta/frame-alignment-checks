import unittest

from frame_alignment_checks.utils import parse_sequence_as_one_hot


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
