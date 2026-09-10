"""
Unit tests for Remote Sensing VQA Engine.
"""

import unittest
import numpy as np
from satquery.models.vqa import RemoteSensingVQA


class TestRemoteSensingVQA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.vqa = RemoteSensingVQA()
        cls.dummy_img = np.random.randint(40, 220, (256, 256, 3), dtype=np.uint8)

    def test_scene_vqa(self):
        res = self.vqa.answer(self.dummy_img, "What type of land cover is shown?")
        self.assertIn("answer", res)
        self.assertIn("confidence", res)
        self.assertGreater(res["confidence"], 0.0)
        self.assertIn("top_predictions", res)
        self.assertGreaterEqual(len(res["top_predictions"]), 1)

    def test_numerical_vqa(self):
        res = self.vqa.answer(self.dummy_img, "How many buildings are in this satellite image?")
        self.assertEqual(res["task"], "numerical_vqa")
        self.assertIn("count", res)
        self.assertIsInstance(res["count"], int)
        self.assertGreaterEqual(res["count"], 1)

    def test_presence_vqa(self):
        res = self.vqa.answer(self.dummy_img, "Is there an airport in this scene?")
        self.assertEqual(res["task"], "presence_vqa")
        self.assertIn("is_present", res)
        self.assertIsInstance(res["is_present"], bool)


if __name__ == "__main__":
    unittest.main()
