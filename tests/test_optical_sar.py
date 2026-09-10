"""
Unit tests for Optical + SAR Cross-Modal Fusion Engine.
"""

import unittest
import numpy as np
from satquery.models.optical_sar import OpticalSARFusionModel
from satquery.preprocessing.sar import SARPreprocessor


class TestOpticalSARFusion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fusion_model = OpticalSARFusionModel()
        cls.opt_img = np.random.randint(60, 200, (256, 256, 3), dtype=np.uint8)
        cls.sar_img = np.random.randint(20, 240, (256, 256), dtype=np.uint8)

    def test_sar_despeckle_filter(self):
        filtered, meta = SARPreprocessor.process_sar(self.sar_img, apply_speckle_filter=True)
        self.assertEqual(filtered.shape, (256, 256, 3))
        self.assertTrue(meta["speckle_filtered"])
        self.assertIn("db_range", meta)

    def test_cross_modal_fusion_execution(self):
        res = self.fusion_model.analyze(self.opt_img, self.sar_img, "Synthesize optical and SAR observations")
        self.assertEqual(res["status"], "EXECUTED")
        self.assertEqual(res["task"], "optical_sar_fusion")
        self.assertIn("answer", res)
        self.assertIn("confidence", res)
        self.assertIn("optical_evidence", res)
        self.assertIn("sar_evidence", res)
        self.assertIn("fusion_visualization", res)
        self.assertIsInstance(res["fusion_visualization"], np.ndarray)


if __name__ == "__main__":
    unittest.main()
