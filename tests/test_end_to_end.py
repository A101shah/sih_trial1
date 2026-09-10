"""
End-to-end verification of SatQuery AI server and API endpoints with honest status checks.
"""

import os
import sys
import json
import threading
import time
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from satquery.server.app import ThreadedHTTPServer, SatQueryRequestHandler
from satquery.agent.controller import AgentController


def test_api_server():
    print("==================================================")
    print("SatQuery AI: End-to-End API Integration Test")
    print("==================================================")

    # Initialize controller
    SatQueryRequestHandler.controller = AgentController()
    server_port = 8766
    server = ThreadedHTTPServer(("127.0.0.1", server_port), SatQueryRequestHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"Test server started on http://127.0.0.1:{server_port}")

    time.sleep(1)

    try:
        # 1. Test /api/health
        print("\n[Test 1] Testing /api/health endpoint...")
        with urllib.request.urlopen(f"http://127.0.0.1:{server_port}/api/health") as response:
            assert response.status == 200
            data = json.loads(response.read().decode())
            assert data["status"] == "healthy"
            print("  -> /api/health OK:", data)

        # 2. Test /api/samples
        print("\n[Test 2] Testing /api/samples endpoint...")
        with urllib.request.urlopen(f"http://127.0.0.1:{server_port}/api/samples") as response:
            assert response.status == 200
            data = json.loads(response.read().decode())
            assert "levir_samples" in data
            print(f"  -> /api/samples OK: Found {len(data['levir_samples'])} LEVIR and {len(data.get('custom_samples', []))} custom samples")

        # 3. Test /api/analyze (Bi-temporal Change Detection with ChangeFormer)
        print("\n[Test 3] Testing /api/analyze endpoint with ChangeFormer...")
        t1_path = os.path.join("samples_LEVIR", "A", "test_102_0512_0000.png")
        t2_path = os.path.join("samples_LEVIR", "B", "test_102_0512_0000.png")

        payload = {
            "image_1_path": t1_path,
            "image_2_path": t2_path,
            "question": "What changed between these two images?",
            "task": "change_analysis"
        }

        req = urllib.request.Request(
            f"http://127.0.0.1:{server_port}/api/analyze",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req) as response:
            assert response.status == 200
            res = json.loads(response.read().decode())
            print(f"  -> Task: {res['task']}")
            print(f"  -> Change percentage: {res['change_percentage']:.2f}%")
            print(f"  -> Real Logits Confidence: {res['confidence']:.4f}")
            print(f"  -> Specialist Statuses: {res['specialist_statuses']}")
            assert res["task"] == "change_analysis"
            assert res["specialist_statuses"]["ChangeFormerV6"] == "EXECUTED"
            assert res["specialist_statuses"]["CDVQA"] == "NOT_CONFIGURED"
            assert res["change_percentage"] > 0
            assert "overlay_t2_url" in res
            assert "change_mask_url" in res

        # 4. Test Single-Image VQA (NOT_CONFIGURED)
        print("\n[Test 4] Testing /api/analyze endpoint with Single Image VQA (Unconfigured)...")
        payload_vqa = {
            "image_1_path": t1_path,
            "question": "What is visible in this satellite imagery?",
            "task": "single_vqa"
        }
        req_vqa = urllib.request.Request(
            f"http://127.0.0.1:{server_port}/api/analyze",
            data=json.dumps(payload_vqa).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_vqa) as response:
            assert response.status == 200
            res_vqa = json.loads(response.read().decode())
            print(f"  -> Task: {res_vqa['task']}")
            print(f"  -> Answer: {res_vqa['answer']}")
            print(f"  -> Confidence: {res_vqa['confidence']}")
            print(f"  -> Specialist Statuses: {res_vqa['specialist_statuses']}")
            assert res_vqa["task"] == "single_vqa"
            assert res_vqa["specialist_statuses"]["RSVLM"] == "NOT_CONFIGURED"
            assert res_vqa["confidence"] is None
            assert "not configured" in res_vqa["answer"].lower()

        # 5. Test Grounding (NOT_CONFIGURED)
        print("\n[Test 5] Testing /api/analyze endpoint with Grounding (Unconfigured)...")
        payload_ground = {
            "image_1_path": t1_path,
            "question": "Where is the built-up region?",
            "task": "grounding"
        }
        req_ground = urllib.request.Request(
            f"http://127.0.0.1:{server_port}/api/analyze",
            data=json.dumps(payload_ground).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_ground) as response:
            assert response.status == 200
            res_ground = json.loads(response.read().decode())
            print(f"  -> Task: {res_ground['task']}")
            print(f"  -> Answer: {res_ground['answer']}")
            print(f"  -> Bboxes: {res_ground['bboxes']}")
            print(f"  -> Confidence: {res_ground['confidence']}")
            assert res_ground["bboxes"] == []
            assert res_ground["confidence"] is None
            assert "not configured" in res_ground["answer"].lower()

        print("\n==================================================")
        print("ALL END-TO-END SERVER & API TESTS PASSED (100%)")
        print("==================================================")

    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    test_api_server()
