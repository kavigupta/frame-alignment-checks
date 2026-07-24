"""
Unit tests for the frameshift guard's peak voter,
``fac.deletion.alphagenome_deletion._frameshift_votes``. Pure numpy -- no
AlphaGenome and no network: the ref/alt track columns are built by hand.

The guard exists to detect if AlphaGenome ever stops left-shifting deletion alt
tracks by del_len (google-deepmind/alphagenome issue #23). A vote *passes* when
the shifted lookup ``alt[i - del_len]`` matches the reference peak better than
the un-shifted ``alt[i]``; it *flips* otherwise.
"""

import unittest

import numpy as np

from frame_alignment_checks.deletion.alphagenome_deletion import (
    FRAMESHIFT_PEAK_REL_FLOOR,
    _frameshift_votes,
)

W = 2000
DEL_LEN = 5
TRACK_START = 10_000
# deletion ends left of every peak below, so the peaks sit in the shifted region
DEL_END = TRACK_START + 100
PEAKS = [300, 700, 1100, 1500]


def _ref_with_peaks(peak_positions, height=1.0, background=0.001):
    ref = np.full(W, background)
    for p in peak_positions:
        ref[p] = height
    return ref


class TestFrameshiftVotes(unittest.TestCase):
    def test_left_shifted_alt_passes(self):
        # alt is ref shifted left by del_len (the un-fixed bug): alt[i - del_len]
        # == ref[i], so the shifted lookup is exact and no peak flips.
        ref = _ref_with_peaks(PEAKS)
        alt = np.roll(ref, -DEL_LEN)
        n_total, n_fail, flipped = _frameshift_votes(
            ref, alt, DEL_LEN, TRACK_START, DEL_END, ti=0
        )
        self.assertEqual(n_total, len(PEAKS))
        self.assertEqual(n_fail, 0)
        self.assertEqual(flipped, [])

    def test_unshifted_alt_flips_every_peak(self):
        # alt == ref (a release fixed the frameshift): the un-shifted lookup is
        # exact, so every peak flips and the guard would fire.
        ref = _ref_with_peaks(PEAKS)
        alt = ref.copy()
        n_total, n_fail, flipped = _frameshift_votes(
            ref, alt, DEL_LEN, TRACK_START, DEL_END, ti=0
        )
        self.assertEqual(n_total, len(PEAKS))
        self.assertEqual(n_fail, len(PEAKS))
        self.assertEqual(len(flipped), len(PEAKS))

    def test_peaks_before_the_deletion_do_not_vote(self):
        # With the deletion end past every peak, none is in the shifted region,
        # so the correction isn't under test and nothing votes.
        ref = _ref_with_peaks(PEAKS)
        alt = np.roll(ref, -DEL_LEN)
        del_end_past_all = TRACK_START + max(PEAKS) + 1
        n_total, _, _ = _frameshift_votes(
            ref, alt, DEL_LEN, TRACK_START, del_end_past_all, ti=0
        )
        self.assertEqual(n_total, 0)

    def test_peak_below_relative_floor_excluded(self):
        # A peak at 0.4 * track-max is below FRAMESHIFT_PEAK_REL_FLOOR (0.5) and
        # must not vote; only the full-height peak does.
        self.assertLess(0.4, FRAMESHIFT_PEAK_REL_FLOOR)
        ref = _ref_with_peaks([300])
        ref[700] = 0.4
        alt = np.roll(ref, -DEL_LEN)
        n_total, _, _ = _frameshift_votes(ref, alt, DEL_LEN, TRACK_START, DEL_END, ti=0)
        self.assertEqual(n_total, 1)

    def test_peak_destroyed_by_deletion_excluded(self):
        # If the deletion wipes the peak out of the alt track (both lookups ~0),
        # the survival gate drops it -- the shift is not testable there.
        ref = _ref_with_peaks(PEAKS)
        alt = np.zeros(W)
        n_total, _, _ = _frameshift_votes(ref, alt, DEL_LEN, TRACK_START, DEL_END, ti=0)
        self.assertEqual(n_total, 0)


if __name__ == "__main__":
    unittest.main()
