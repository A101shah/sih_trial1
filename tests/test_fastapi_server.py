"""
Integration tests for FastAPI REST API endpoints using TestClient.
"""

import unittest
from fastapi.testclient import TestClient
from satquery.server.fastapi_app import app


class TestFastAPIServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("models", data)

    def test_models_registry_endpoint(self):
        response = self.client.get("/api/v1/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreater(data["count"], 0)
        self.assertIn("models", data)

    def test_samples_endpoint(self):
        response = self.client.get("/api/v1/samples")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("levir_samples", data)

    def test_benchmark_endpoint(self):
        response = self.client.get("/api/v1/benchmark")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("benchmark_markdown", data)


if __name__ == "__main__":
    unittest.main()
