import unittest

import numpy as np

from frame_alignment_checks.replace_3mer.stop_codon_replacement_no_undesired_changes import (
    stop_codon_no_undesired_changes_mask,
)
from frame_alignment_checks.utils import all_3mers, draw_bases

num_exons_studied = 100


class TestNoUndesiredChangesMask(unittest.TestCase):
    def test_no_removal_of_out_of_frame_stop(self):
        """
        This is a weird single test to include, but it exists because it came up as a bug.

        Basically, there's an out of frame TGA stop codon [at phase 2], so if you insert CAA into location
        1, it should reduce the number of stop codons by 1, which is an undesired change.
        """
        o_seq = np.array(
            [
                # CCT GAC CCC
                [1, 1, 3, 2, 0, 1, 1, 1, 1],
                # CCC CCC CCC
                [1] * 9,
            ]
        )
        [[no_undesired_changes_a, no_undesired_changes_d]] = (
            stop_codon_no_undesired_changes_mask(o_seq[None])
        )
        # no undesired changes in D because there's no way to induce a stop
        # codon out of the frame being modified
        self.assertTrue(no_undesired_changes_d.all())
        # phase -1 of A has no undesired changes because the thing being replaced is
        # exactly the stop codon, so all changes are desired
        self.assertTrue(no_undesired_changes_a[0].all())
        # phase 0 of A has undesired changes iff the first two nucleotides are not GA, AG, or AA
        for i in range(64):
            self.assertEqual(
                no_undesired_changes_a[1, i],
                draw_bases(all_3mers()[i])[:2] in {"GA", "AG", "AA"},
                i,
            )
        # phase 1 of A has undesired changes iff the first nucleotide is not A
        for i in range(64):
            self.assertEqual(
                no_undesired_changes_a[2, i], draw_bases(all_3mers()[i])[0] == "A", i
            )
