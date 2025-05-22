import unittest

import frame_alignment_checks as fac


class TestCodonTableConsistency(unittest.TestCase):
    def test_isomorphism(self):
        forward = fac.amino_acid_to_codons
        backward = fac.codon_to_amino_acid
        for x, codons_x in forward.items():
            for codon_x in codons_x:
                self.assertEqual(x, backward[codon_x])

        for y, amino_y in backward.items():
            self.assertIn(y, forward[amino_y])

    def test_backwards(self):
        # transcribed from
        # https://en.wikipedia.org/wiki/DNA_and_RNA_codon_tables#Standard_RNA_codon_table
        expected = {
            "TTT": "F",
            "TTC": "F",
            "TTA": "L",
            "TTG": "L",
            "CTT": "L",
            "CTC": "L",
            "CTA": "L",
            "CTG": "L",
            "ATT": "I",
            "ATC": "I",
            "ATA": "I",
            "ATG": "M",
            "GTT": "V",
            "GTC": "V",
            "GTA": "V",
            "GTG": "V",
            "TCT": "S",
            "TCC": "S",
            "TCA": "S",
            "TCG": "S",
            "CCT": "P",
            "CCC": "P",
            "CCA": "P",
            "CCG": "P",
            "ACT": "T",
            "ACC": "T",
            "ACA": "T",
            "ACG": "T",
            "GCT": "A",
            "GCC": "A",
            "GCA": "A",
            "GCG": "A",
            "TAT": "Y",
            "TAC": "Y",
            "TAA": "*",
            "TAG": "*",
            "CAT": "H",
            "CAC": "H",
            "CAA": "Q",
            "CAG": "Q",
            "AAT": "N",
            "AAC": "N",
            "AAA": "K",
            "AAG": "K",
            "GAT": "D",
            "GAC": "D",
            "GAA": "E",
            "GAG": "E",
            "TGT": "C",
            "TGC": "C",
            "TGA": "*",
            "TGG": "W",
            "CGT": "R",
            "CGC": "R",
            "CGA": "R",
            "CGG": "R",
            "AGT": "S",
            "AGC": "S",
            "AGA": "R",
            "AGG": "R",
            "GGT": "G",
            "GGC": "G",
            "GGA": "G",
            "GGG": "G",
        }
        self.assertEqual(fac.codon_to_amino_acid, expected)
