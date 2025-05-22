import unittest

import frame_alignment_checks as fac


class TestAminoCatCoverage(unittest.TestCase):
    def test_amino_cat_coverage(self):
        self.assertEqual(
            sorted(fac.amino_acid_to_codons.keys()),
            sorted(fac.amino_replacement.amino_classification.keys()),
        )

    def test_amino_cat_unique(self):
        for cat in fac.amino_replacement.amino_classification.values():
            for amino in cat:
                self.assertEqual(cat, fac.amino_replacement.amino_classification[amino])
