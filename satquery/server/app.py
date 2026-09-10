"""
SatQuery AI REST API Server.
Zero-dependency, multi-threaded HTTP server providing endpoints for multimodal analysis,
image uploads, sample inspection, and report downloads.
"""

import os
import sys
import json
import base64
import mimetypes
from io import BytesIO
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

from ..agent.controller import AgentController
from ..reporting.export import ReportExporter


def array_to_base64(arr: np.ndarray, format_str: str = "PNG") -> str:
    """Encodes numpy uint8 array to base64 data URL."""
    if arr.ndim == 2:
        pil_img = Image.fromarray(arr.astype(np.uint8))
    else:
        pil_img = Image.fromarray(arr.astype(np.uint8)[:, :, :3])
    buffered = BytesIO()
    pil_img.save(buffered, format=format_str)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format_str.lower()};base64,{encoded}"


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handles requests in separate threads for high concurrency."""
    daemon_threads = True


class SatQueryRequestHandler(SimpleHTTPRequestHandler):
    """
    REST API handler for SatQuery AI.
    """

    controller: Optional[AgentController] = None

    def __init__(self, *args, **kwargs):
        frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
        super().__init__(*args, directory=frontend_dir, **kwargs)

    def _send_json_response(self, data: Dict[str, Any], status: int = 200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/health":
            self._send_json_response({
                "status": "healthy",
                "model": "ChangeFormerV6 + RS-VLM Multimodal",
                "version": "0.1.0"
            })
            return

        elif parsed.path == "/api/samples":
            # List available sample datasets
            samples_dir_a = os.path.join("samples_LEVIR", "A")
            samples_dir_b = os.path.join("samples_LEVIR", "B")
            levir_samples = []
            if os.path.isdir(samples_dir_a):
                for f in sorted(os.listdir(samples_dir_a))[:10]:
                    if f.endswith(('.png', '.jpg', '.tif', '.tiff')):
                        levir_samples.append({
                            "id": f,
                            "name": f,
                            "type": "bi_temporal_optical",
                            "t1_path": os.path.join("samples_LEVIR", "A", f),
                            "t2_path": os.path.join("samples_LEVIR", "B", f)
                        })

            custom_samples = []
            if os.path.isfile(os.path.join("custom_input", "A", "image1.png")):
                custom_samples.append({
                    "id": "custom_pair_1",
                    "name": "Custom Input Pair (450x221)",
                    "type": "bi_temporal_optical",
                    "t1_path": os.path.join("custom_input", "A", "image1.png"),
                    "t2_path": os.path.join("custom_input", "B", "image1.png")
                })

            self._send_json_response({
                "levir_samples": levir_samples,
                "custom_samples": custom_samples
            })
            return

        elif parsed.path == "/api/sample_image":
            query = parse_qs(parsed.query)
            path = query.get("path", [None])[0]
            if path and os.path.isfile(path):
                try:
                    with open(path, "rb") as f:
                        img_data = f.read()
                    mime, _ = mimetypes.guess_type(path)
                    self.send_response(200)
                    self.send_header("Content-Type", mime or "image/png")
                    self.send_header("Content-Length", str(len(img_data)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(img_data)
                    return
                except Exception as e:
                    self._send_json_response({"error": str(e)}, status=500)
                    return
            self._send_json_response({"error": "File not found"}, status=404)
            return

        # Serve static frontend files
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/analyze":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(content_length).decode("utf-8")
                payload = json.loads(raw_body)

                image_1_b64 = payload.get("image_1")
                image_2_b64 = payload.get("image_2")
                image_1_path = payload.get("image_1_path")
                image_2_path = payload.get("image_2_path")
                question = payload.get("question")
                task = payload.get("task")
                modality_1 = payload.get("modality_1", "optical")
                modality_2 = payload.get("modality_2", "optical")

                # Resolve image 1
                if image_1_path and os.path.isfile(image_1_path):
                    img_1 = image_1_path
                elif image_1_b64:
                    if "," in image_1_b64:
                        image_1_b64 = image_1_b64.split(",", 1)[1]
                    img_1 = Image.open(BytesIO(base64.b64decode(image_1_b64))).convert("RGB")
                else:
                    self._send_json_response({"error": "No primary image provided"}, status=400)
                    return

                # Resolve image 2
                img_2 = None
                if image_2_path and os.path.isfile(image_2_path):
                    img_2 = image_2_path
                elif image_2_b64:
                    if "," in image_2_b64:
                        image_2_b64 = image_2_b64.split(",", 1)[1]
                    img_2 = Image.open(BytesIO(base64.b64decode(image_2_b64))).convert("RGB")

                # Run Agent Controller
                if SatQueryRequestHandler.controller is None:
                    SatQueryRequestHandler.controller = AgentController()

                result = SatQueryRequestHandler.controller.execute(
                    image_1=img_1,
                    image_2=img_2,
                    question=question,
                    task=task,
                    modality_1=modality_1,
                    modality_2=modality_2
                )

                # Format images in response as base64 URLs
                response_data = {
                    "task": result.get("task"),
                    "question": result.get("question"),
                    "answer": result.get("answer"),
                    "confidence": result.get("confidence"),
                    "confidence_source": result.get("confidence_source"),
                    "specialist_statuses": result.get("specialist_statuses", {}),
                    "models_used": result.get("models_used", []),
                    "trace": result.get("trace", []),
                    "execution_time_sec": result.get("execution_time_sec"),
                    "warnings": result.get("warnings", []),
                    "geospatial_metadata": result.get("geospatial_metadata", {})
                }

                if "change_percentage" in result:
                    response_data["change_percentage"] = result["change_percentage"]
                    response_data["change_type"] = result.get("change_type")
                    response_data["spatial_evidence"] = result.get("spatial_evidence", {})

                if "bboxes" in result:
                    response_data["bboxes"] = result.get("bboxes", [])

                if "change_mask" in result and isinstance(result["change_mask"], np.ndarray):
                    response_data["change_mask_url"] = array_to_base64(result["change_mask"])

                if "overlay_t1" in result and isinstance(result["overlay_t1"], np.ndarray):
                    response_data["overlay_t1_url"] = array_to_base64(result["overlay_t1"])

                if "overlay_t2" in result and isinstance(result["overlay_t2"], np.ndarray):
                    response_data["overlay_t2_url"] = array_to_base64(result["overlay_t2"])

                if "fusion_visualization" in result and isinstance(result["fusion_visualization"], np.ndarray):
                    response_data["fusion_vis_url"] = array_to_base64(result["fusion_visualization"])

                if "visualization" in result and isinstance(result["visualization"], np.ndarray):
                    response_data["grounding_vis_url"] = array_to_base64(result["visualization"])

                if "optical_evidence" in result:
                    response_data["optical_evidence"] = result["optical_evidence"]
                if "sar_evidence" in result:
                    response_data["sar_evidence"] = result["sar_evidence"]
                if "fusion_evidence" in result:
                    response_data["fusion_evidence"] = result["fusion_evidence"]
                if "evidence" in result:
                    response_data["evidence"] = result["evidence"]

                # Generate report formats
                response_data["markdown_report"] = ReportExporter.to_markdown(result)

                self._send_json_response(response_data)

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json_response({"error": str(e)}, status=500)
            return

        self._send_json_response({"error": "Endpoint not found"}, status=404)


def run_server(host: str = "127.0.0.1", port: int = 8080):
    """Starts the SatQuery AI server."""
    print("==================================================")
    print("Initializing SatQuery AI Specialist Controller...")
    SatQueryRequestHandler.controller = AgentController()
    print("Controller initialized.")
    server = ThreadedHTTPServer((host, port), SatQueryRequestHandler)
    print(f"SatQuery AI Web Server running on http://{host}:{port}")
    print("==================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down SatQuery AI server...")
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port=port)
