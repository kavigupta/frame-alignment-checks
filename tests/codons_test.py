import unittest

import numpy as np
from parameterized import parameterized

import frame_alignment_checks as fac


class TestSequenceToCodons(unittest.TestCase):
    def test_basic_with_argmax(self):
        sequence = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
        codons = fac.compute_stop_codons.sequence_to_codons(sequence)
        expected = np.array([[2, 1, 0]])
        np.testing.assert_array_equal(codons, expected)

    def test_basic_without_argmax(self):
        sequence = np.array([2, 3, 0])
        codons = fac.compute_stop_codons.sequence_to_codons(sequence)
        expected = np.array([[2, 3, 0]])
        np.testing.assert_array_equal(codons, expected)

    def test_multiple_of_3(self):
        sequence = np.array([2, 3, 0, 1, 2, 3, 0, 0, 0, 2, 2, 3])
        codons = fac.compute_stop_codons.sequence_to_codons(sequence)
        expected = np.array([[2, 3, 0], [1, 2, 3], [0, 0, 0], [2, 2, 3]])
        np.testing.assert_array_equal(codons, expected)

    def test_extra(self):
        sequence = np.array([2, 3, 0, 1, 2, 3, 0, 0, 0, 2, 2, 3, 1])
        codons = fac.compute_stop_codons.sequence_to_codons(sequence)
        expected = np.array([[2, 3, 0], [1, 2, 3], [0, 0, 0], [2, 2, 3]])
        np.testing.assert_array_equal(codons, expected)

    def test_offset_multiple_3(self):
        sequence = np.array([2, 3, 0, 1, 2, 3, 0, 0, 0, 2, 2, 3, 1])
        codons = fac.compute_stop_codons.sequence_to_codons(sequence, off=2)
        expected = np.array([[0, 1, 2], [3, 0, 0], [0, 2, 2]])
        np.testing.assert_array_equal(codons, expected)

    def test_offset_extra(self):
        sequence = np.array([2, 3, 0, 1, 2, 3, 0, 0, 0, 2, 2, 3, 1, 2])
        codons = fac.compute_stop_codons.sequence_to_codons(sequence, off=2)
        expected = np.array([[0, 1, 2], [3, 0, 0], [0, 2, 2], [3, 1, 2]])
        np.testing.assert_array_equal(codons, expected)

    def test_invalid_offsets(self):
        sequence = np.array([2, 3, 0, 1, 2, 3, 0, 0, 0, 2, 2, 3, 1, 2])
        with self.assertRaises(ValueError):
            fac.compute_stop_codons.sequence_to_codons(sequence, off=3)
        with self.assertRaises(ValueError):
            fac.compute_stop_codons.sequence_to_codons(sequence, off=-1)


class TestIsStop(unittest.TestCase):
    def test_basic(self):
        codons = np.array([[3, 0, 2], [3, 0, 0], [3, 2, 0], [0, 0, 0]])
        stops = fac.compute_stop_codons.is_stop(codons)
        expected = np.array([True, True, True, False])
        np.testing.assert_array_equal(stops, expected)

    def test_ensure_all_known(self):
        codons = np.random.RandomState(0).choice(4, size=(10_000, 3))
        stop_mask = fac.compute_stop_codons.is_stop(codons)
        drawn = fac.utils.draw_bases(codons)
        stops = {draw for draw, stop in zip(drawn, stop_mask) if stop}
        self.assertEqual(stops, {"TAG", "TAA", "TGA"})


class TestAllFramesClosed(unittest.TestCase):
    def to_sequence(self, xs):
        return np.array(["ACGT".index(x) for x in xs])

    def test_all_closed(self):
        exons = [
            self.to_sequence(xs)
            for xs in ["TAGATGAATAA", "TAGTAATGAATAAATAA", "TAGTAATGA"]
        ]
        closed = fac.compute_stop_codons.all_frames_closed(exons)
        self.assertTrue(closed.tolist(), [True, True, False])

    @parameterized.expand([(i,) for i in range(100)])
    def test_independent(self, i):
        rng = np.random.RandomState(i)
        sequences = [rng.choice(4, size=rng.randint(100, 200)) for _ in range(1000)]
        closed = fac.compute_stop_codons.all_frames_closed(sequences)
        subset_idxs = rng.choice(len(sequences), size=100, replace=False)
        subset_sequences = [sequences[i] for i in subset_idxs]
        subset_closed = fac.compute_stop_codons.all_frames_closed(subset_sequences)
        self.assertTrue((closed[subset_idxs] == subset_closed).all())
