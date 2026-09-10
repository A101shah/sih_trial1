"""
FastAPI High-Performance REST API Microservice for SatQuery AI.
Provides async endpoints for remote sensing multi-modal analysis, image validation,
model registry inspection, quantitative benchmarks, and geospatial report exports.
"""

import os
import sys
import uuid
import time
import base64
import mimetypes
from io import BytesIO
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from ..agent.controller import AgentController
from ..agent.registry import ModelRegistry
from ..preprocessing.geotiff import GeoTIFFProcessor
from ..preprocessing.validation import InputValidator
from ..reporting.export import ReportExporter
from ..evaluation.benchmark import EvaluationBenchmark


# Initialize FastAPI app
app = FastAPI(
    title="SatQuery AI API",
    description="Interactive Multimodal Remote-Sensing AI Assistant API (SIH26167)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
controller = AgentController()
registry = ModelRegistry()
benchmark_runner = EvaluationBenchmark(controller)

# In-memory results cache for task retrieval
results_cache: Dict[str, Dict[str, Any]] = {}


# Pydantic Schemas
class AnalyzeRequest(BaseModel):
    image_1: Optional[str] = Field(None, description="Base64 data URL for Image 1")
    image_2: Optional[str] = Field(None, description="Base64 data URL for Image 2")
    image_1_path: Optional[str] = Field(None, description="Server file path for Image 1")
    image_2_path: Optional[str] = Field(None, description="Server file path for Image 2")
    question: Optional[str] = Field(None, description="User question or prompt")
    task: Optional[str] = Field(None, description="Explicit task override (change_analysis, vqa, grounding, optical_sar)")
    modality_1: Optional[str] = Field("optical", description="Modality of Image 1")
    modality_2: Optional[str] = Field("optical", description="Modality of Image 2")


def array_to_base64(arr: np.ndarray, format_str: str = "PNG") -> str:
    """Encodes numpy array into base64 data URL."""
    if arr.ndim == 2:
        pil_img = Image.fromarray(arr.astype(np.uint8))
    else:
        pil_img = Image.fromarray(arr.astype(np.uint8)[:, :, :3])
    buffered = BytesIO()
    pil_img.save(buffered, format=format_str)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format_str.lower()};base64,{encoded}"


# Endpoints
@app.get("/api/v1/health")
@app.get("/api/health")
async def health_check():
    """Health check and model readiness status."""
    return {
        "status": "healthy",
        "system": "SatQuery AI Remote Sensing Assistant",
        "version": "2.0.0",
        "models": {
            "change_detection": "ChangeFormerV6 (Loaded)",
            "vqa": "RS-VQA-ResNet50-DualHead (Loaded)",
            "grounding": "RS-Grounding-FPN (Loaded)",
            "optical_sar": "Optical-SAR CrossModal (Loaded)",
            "geospatial": "Rasterio CRS Engine (Ready)"
        }
    }


@app.get("/api/v1/models")
async def list_models():
    """Returns the catalog of registered remote-sensing AI specialists."""
    return {
        "count": len(registry.list_all()),
        "models": registry.list_all()
    }


@app.get("/api/v1/samples")
@app.get("/api/samples")
async def list_samples():
    """Lists available remote-sensing demo samples."""
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

    return {
        "levir_samples": levir_samples,
        "custom_samples": custom_samples
    }


@app.get("/api/v1/sample_image")
@app.get("/api/sample_image")
async def get_sample_image(path: str):
    """Serves a sample image file from disk."""
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Sample image not found")
    mime, _ = mimetypes.guess_type(path)
    return FileResponse(path, media_type=mime or "image/png")


@app.post("/api/v1/upload")
async def upload_image(file: UploadFile = File(...)):
    """Uploads a satellite / aerial image or GeoTIFF for processing."""
    upload_dir = os.path.join("custom_input", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    file_id = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(upload_dir, file_id)
    
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
        
    geo_meta = GeoTIFFProcessor.inspect(file_path) if GeoTIFFProcessor.is_geotiff(file_path) else {}
    
    return {
        "status": "UPLOADED",
        "filename": file.filename,
        "file_path": file_path,
        "is_geotiff": GeoTIFFProcessor.is_geotiff(file_path),
        "geospatial_metadata": geo_meta
    }


@app.post("/api/v1/validate")
async def validate_pair(
    image_1_path: str = Form(...),
    image_2_path: Optional[str] = Form(None),
    task_type: str = Form("bi_temporal")
):
    """Validates image pair dimensions, bands, modalities, and geospatial overlap."""
    val_result = InputValidator.validate_pair(image_1_path, image_2_path, task_type=task_type)
    
    overlap_info = {}
    if image_2_path and GeoTIFFProcessor.is_geotiff(image_1_path) and GeoTIFFProcessor.is_geotiff(image_2_path):
        m1 = GeoTIFFProcessor.inspect(image_1_path)
        m2 = GeoTIFFProcessor.inspect(image_2_path)
        overlap_info = GeoTIFFProcessor.check_spatial_overlap(m1, m2)
        
    return {
        "validation": val_result,
        "geospatial_overlap": overlap_info
    }


@app.post("/api/v1/analyze")
@app.post("/api/analyze")
async def analyze_scene(req: AnalyzeRequest):
    """Main multimodal analysis pipeline."""
    try:
        # Resolve Image 1
        if req.image_1_path and os.path.isfile(req.image_1_path):
            img_1 = req.image_1_path
        elif req.image_1:
            raw_b64 = req.image_1.split(",", 1)[1] if "," in req.image_1 else req.image_1
            img_1 = Image.open(BytesIO(base64.b64decode(raw_b64))).convert("RGB")
        else:
            raise HTTPException(status_code=400, detail="Primary image (image_1) required")

        # Resolve Image 2
        img_2 = None
        if req.image_2_path and os.path.isfile(req.image_2_path):
            img_2 = req.image_2_path
        elif req.image_2:
            raw_b64 = req.image_2.split(",", 1)[1] if "," in req.image_2 else req.image_2
            img_2 = Image.open(BytesIO(base64.b64decode(raw_b64))).convert("RGB")

        # Execute Agent Controller
        result = controller.execute(
            image_1=img_1,
            image_2=img_2,
            question=req.question,
            task=req.task,
            modality_1=req.modality_1 or "optical",
            modality_2=req.modality_2 or "optical"
        )

        # Generate task ID and cache
        task_id = uuid.uuid4().hex[:12]
        
        # Format response
        resp = {
            "task_id": task_id,
            "task": result.get("task"),
            "question": result.get("question"),
            "answer": result.get("answer"),
            "confidence": result.get("confidence"),
            "confidence_source": result.get("confidence_source"),
            "uncertainty": result.get("uncertainty"),
            "specialist_statuses": result.get("specialist_statuses", {}),
            "models_used": result.get("models_used", []),
            "trace": result.get("trace", []),
            "execution_plan": result.get("execution_plan", []),
            "execution_time_sec": result.get("execution_time_sec"),
            "warnings": result.get("warnings", []),
            "geospatial_metadata": result.get("geospatial_metadata", {})
        }

        if "change_percentage" in result:
            resp["change_percentage"] = result["change_percentage"]
            resp["change_type"] = result.get("change_type")
            resp["spatial_evidence"] = result.get("spatial_evidence", {})

        if "bboxes" in result:
            resp["bboxes"] = result.get("bboxes", [])

        if "geojson" in result and result["geojson"]:
            resp["geojson"] = result["geojson"]

        # Base64 image visualizations
        if "change_mask" in result and isinstance(result["change_mask"], np.ndarray):
            resp["change_mask_url"] = array_to_base64(result["change_mask"])

        if "difference_map" in result and isinstance(result["difference_map"], np.ndarray):
            resp["difference_map_url"] = array_to_base64(result["difference_map"])

        if "overlay_t1" in result and isinstance(result["overlay_t1"], np.ndarray):
            resp["overlay_t1_url"] = array_to_base64(result["overlay_t1"])

        if "overlay_t2" in result and isinstance(result["overlay_t2"], np.ndarray):
            resp["overlay_t2_url"] = array_to_base64(result["overlay_t2"])

        if "fusion_visualization" in result and isinstance(result["fusion_visualization"], np.ndarray):
            resp["fusion_vis_url"] = array_to_base64(result["fusion_visualization"])

        if "visualization" in result and isinstance(result["visualization"], np.ndarray):
            resp["grounding_vis_url"] = array_to_base64(result["visualization"])

        if "optical_evidence" in result:
            resp["optical_evidence"] = result["optical_evidence"]
        if "sar_evidence" in result:
            resp["sar_evidence"] = result["sar_evidence"]
        if "fusion_evidence" in result:
            resp["fusion_evidence"] = result["fusion_evidence"]

        if "top_predictions" in result:
            resp["top_predictions"] = result["top_predictions"]

        resp["markdown_report"] = ReportExporter.to_markdown(result)
        results_cache[task_id] = resp

        return resp

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/result/{task_id}")
async def get_cached_result(task_id: str):
    """Retrieves cached analysis result by task ID."""
    if task_id not in results_cache:
        raise HTTPException(status_code=404, detail="Task result not found or expired")
    return results_cache[task_id]


@app.get("/api/v1/benchmark")
async def get_benchmark_report():
    """Generates quantitative benchmark summary across models."""
    table_md = benchmark_runner.generate_comparison_table()
    return {
        "status": "SUCCESS",
        "benchmark_markdown": table_md
    }


# Mount Frontend static assets
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def serve_index():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.isfile(index_file):
            return FileResponse(index_file)
        return HTMLResponse("<h1>SatQuery AI Backend Running</h1>")
