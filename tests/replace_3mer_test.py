import tempfile
import unittest

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import frame_alignment_checks as fac
from frame_alignment_checks.utils import all_3mers, draw_bases
from tests.models.models import lssi_model, lssi_model_with_orf
from tests.utils import skip_on_mac

num_exons_studied = 100
is_testing = True

rendered_codons = draw_bases(all_3mers())


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
            fac.replace_3mer.no_undesired_changes_mask(o_seq[None])
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


class TestStopCodons(unittest.TestCase):
    @skip_on_mac
    def test_shape(self):
        result = fac.replace_3mer.experiment(
            model_for_analysis=lssi_model(), distance_out=40, limit=num_exons_studied
        )
        self.assertEqual(
            result.no_undesired_changes_mask.shape, (num_exons_studied, 2, 3, 64)
        )
        self.assertEqual(result.acc_delta.shape, (1, num_exons_studied, 2, 3, 64))

    @skip_on_mac
    def test_lssi_doesnt_have_any_impact(self):
        result = fac.replace_3mer.experiment(
            model_for_analysis=lssi_model(), distance_out=40, limit=num_exons_studied
        )
        self.assertTrue((result.acc_delta == 0).all())

    @skip_on_mac
    def test_no_undesired_changes_effect(self):
        result = fac.replace_3mer.experiment(
            model_for_analysis=lssi_model_with_orf(),
            distance_out=40,
            limit=num_exons_studied,
        )
        [phase_2, phase_0, phase_1] = (
            result.acc_delta[0] * result.no_undesired_changes_mask != 0
        ).any((0, 1))
        self.assertFalse(phase_2.any())
        self.assertFalse(phase_1.any())
        self.assertEqual(
            {rendered_codons[i] for i in np.where(phase_0)[0]}, {"TAA", "TAG", "TGA"}
        )

    @skip_on_mac
    def test_effect_direction(self):
        result = fac.replace_3mer.experiment(
            model_for_analysis=lssi_model_with_orf(),
            distance_out=40,
            limit=num_exons_studied,
        )
        idxs = [rendered_codons.index(c) for c in ("TAA", "TAG", "TAA")]
        nuc, acc_delta = (
            result.no_undesired_changes_mask[..., idxs],
            result.acc_delta[0][..., idxs],
        )
        result_w_nuc = acc_delta[nuc]
        result_w_nuc = result_w_nuc[result_w_nuc != 0]
        self.assertGreater(
            (result_w_nuc < 0).mean(), 0.9, "Results must be overwhelmingly positive"
        )


class TestPlotting(unittest.TestCase):
    def create_image(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            plt.savefig(f.name)
            return Image.open(f.name)

    def check_image(self, path):
        img_as_array = np.array(self.create_image())
        if is_testing:
            saved_img_as_array = np.array(Image.open(path))
            np.testing.assert_array_equal(img_as_array, saved_img_as_array)
        else:
            Image.fromarray(img_as_array).save(path)

    @skip_on_mac
    def test_plot_effect_grouped(self):
        result = fac.replace_3mer.experiment(
            model_for_analysis=lssi_model_with_orf(),
            distance_out=40,
            limit=num_exons_studied,
        )
        fac.replace_3mer.plot_by_codon(result, result.no_undesired_changes_mask)
        self.check_image("tests/images/replace_3mer_plot_effect_grouped.png")
        plt.close()

    @skip_on_mac
    def test_plot_effect_grouped_nuc(self):
        result = fac.replace_3mer.experiment(
            model_for_analysis=lssi_model_with_orf(),
            distance_out=40,
            limit=num_exons_studied,
        )
        fac.replace_3mer.plot_by_codon(
            result, np.ones_like(result.no_undesired_changes_mask)
        )
        self.check_image("tests/images/replace_3mer_plot_effect_grouped_nuc.png")
        plt.close()

    def test_is_testing(self):
        self.assertTrue(is_testing)
