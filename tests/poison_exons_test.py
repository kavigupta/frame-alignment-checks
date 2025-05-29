import unittest

import numpy as np

import frame_alignment_checks as fac
from tests.models.models import lssi_model, lssi_model_with_orf
from tests.utils import ImageTestBase, skip_on_mac

num_exons_studied = 500


class TestPoisonExons(unittest.TestCase):
    @skip_on_mac
    def test_lssi_doesnt_have_any_impact(self):
        result = fac.poison_exons.poison_exon_scores(lssi_model())
        all_closed = fac.poison_exons.load_all_closed()
        print()
        self.assertGreater(
            result[all_closed].mean() - result[~all_closed].mean(), -0.05
        )

    @skip_on_mac
    def test_lssi_orf_does_have_impact(self):
        result = fac.poison_exons.poison_exon_scores(lssi_model_with_orf())
        result = np.clip(result, -10, 0)
        all_closed = fac.poison_exons.load_all_closed()
        self.assertLess(result[all_closed].mean() - result[~all_closed].mean(), -1)


class TestPlotting(ImageTestBase):

    def models(self):
        return {"LSSI-orf": [lssi_model_with_orf()], "LSSI": [lssi_model()]}

    def model_results(self):
        return {
            k: np.clip(v, -10, 0)
            for k, v in fac.poison_exons.poison_exon_scores_for_model_series(
                self.models()
            ).items()
        }

    def test_scatterplots(self):
        fac.poison_exons.poison_exon_scatterplots(
            self.model_results(),
        )
        self.check_image()

    def test_summary_plot(self):
        fac.poison_exons.poison_exons_summary_plot(
            self.model_results(),
            k=100,
        )
        self.check_image()
