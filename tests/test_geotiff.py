"""
Unit tests for GeoTIFF and Geospatial Engine.
"""

import unittest
from satquery.preprocessing.geotiff import GeoTIFFProcessor


class TestGeoTIFFEngine(unittest.TestCase):

    def test_pixel_to_geo_transform(self):
        mock_meta = {
            "crs": "EPSG:4326",
            "bounds": [-122.5, 37.7, -122.3, 37.9],
            "shape": (512, 512),
            "transform": [0.00039, 0.0, -122.5, 0.0, -0.00039, 37.9]
        }
        
        geo_x, geo_y = GeoTIFFProcessor.pixel_to_geo(mock_meta, 256, 256)
        self.assertIsInstance(geo_x, float)
        self.assertIsInstance(geo_y, float)
        self.assertAlmostEqual(geo_x, -122.40, places=1)
        self.assertAlmostEqual(geo_y, 37.80, places=1)

    def test_bbox_to_geojson_polygon(self):
        mock_meta = {
            "crs": "EPSG:4326",
            "bounds": [-122.5, 37.7, -122.3, 37.9],
            "shape": (512, 512),
            "transform": [0.00039, 0.0, -122.5, 0.0, -0.00039, 37.9]
        }
        
        poly = GeoTIFFProcessor.bbox_to_geojson_polygon(mock_meta, [100, 100, 200, 200])
        self.assertEqual(poly["type"], "Polygon")
        self.assertEqual(len(poly["coordinates"][0]), 5)  # 4 vertices + closed point

    def test_spatial_overlap_check(self):
        meta1 = {"bounds": [10.0, 50.0, 10.5, 50.5]}
        meta2 = {"bounds": [10.2, 50.2, 10.7, 50.7]}
        
        overlap = GeoTIFFProcessor.check_spatial_overlap(meta1, meta2)
        self.assertTrue(overlap["has_overlap"])
        self.assertEqual(overlap["overlap_status"], "VERIFIED_OVERLAP")


if __name__ == "__main__":
    unittest.main()
