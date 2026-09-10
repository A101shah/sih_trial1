"""
Unit tests for Remote Sensing Text Grounding Engine.
"""

import unittest
import numpy as np
from satquery.models.grounding import TextGroundingEngine


class TestGroundingEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = TextGroundingEngine()
        cls.dummy_img = np.random.randint(40, 220, (512, 512, 3), dtype=np.uint8)

    def test_ground_text_localization(self):
        res = self.engine.ground_text(self.dummy_img, "industrial warehouse storage tanks")
        self.assertEqual(res["status"], "EXECUTED")
        self.assertEqual(res["task"], "grounding")
        self.assertIn("bboxes", res)
        self.assertGreater(len(res["bboxes"]), 0)
        
        # Check box coordinates [ymin, xmin, ymax, xmax]
        for bbox in res["bboxes"]:
            ymin, xmin, ymax, xmax = bbox
            self.assertGreaterEqual(ymin, 0)
            self.assertGreaterEqual(xmin, 0)
            self.assertGreaterEqual(ymax, ymin)
            self.assertGreaterEqual(xmax, xmin)

        self.assertIn("confidence", res)
        self.assertIn("visualization", res)
        self.assertIsInstance(res["visualization"], np.ndarray)


if __name__ == "__main__":
    unittest.main()
