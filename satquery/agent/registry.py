"""
Model & Tool Registry for SatQuery AI.
Maintains structured catalog of available remote sensing AI specialists, their hardware requirements,
input/output schemas, and runtime capabilities.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class SpecialistMetadata:
    id: str
    name: str
    category: str
    description: str
    required_inputs: List[str]
    supported_modalities: List[str]
    device: str
    vram_mb: int
    confidence_source: str
    status: str = "READY"


class ModelRegistry:
    """
    Centralized registry for SatQuery AI specialist models.
    """

    def __init__(self):
        self._registry: Dict[str, SpecialistMetadata] = {}
        self._register_default_models()

    def _register_default_models(self):
        # 1. ChangeFormerV6
        self.register(SpecialistMetadata(
            id="change_detection",
            name="ChangeFormerV6 (Transformer-based Bi-temporal CD)",
            category="change_detection",
            description="Siamese Transformer network for high-resolution optical bi-temporal building & land change detection.",
            required_inputs=["image_t1", "image_t2"],
            supported_modalities=["optical", "multispectral"],
            device="cuda/cpu",
            vram_mb=1200,
            confidence_source="Softmax Logit Class Posterior"
        ))

        # 2. RS-VQA Engine
        self.register(SpecialistMetadata(
            id="vqa",
            name="RS-VQA Multi-Task Engine (ResNet50-DualHead)",
            category="vision_language",
            description="Deep feature extraction and domain-adapted classifier for remote-sensing scene VQA, object presence, and numerical counting.",
            required_inputs=["image_1", "question"],
            supported_modalities=["optical", "multispectral", "sar"],
            device="cuda/cpu",
            vram_mb=800,
            confidence_source="Calibrated Softmax Logits"
        ))

        # 3. RS Text Grounding
        self.register(SpecialistMetadata(
            id="grounding",
            name="RS-Grounding Multi-Scale Feature Pyramid Engine",
            category="spatial_grounding",
            description="Open-vocabulary text-guided spatial localization generating bounding boxes [ymin, xmin, ymax, xmax] and masks.",
            required_inputs=["image_1", "target_text"],
            supported_modalities=["optical", "multispectral"],
            device="cuda/cpu",
            vram_mb=950,
            confidence_source="Feature Activation Energy Peak"
        ))

        # 4. Optical + SAR Cross-Modal Fusion
        self.register(SpecialistMetadata(
            id="optical_sar_fusion",
            name="Optical-SAR Cross-Modal Fusion Engine",
            category="multimodal_fusion",
            description="Fuses visible/NIR spectral reflectance with Synthetic Aperture Radar (SAR) dielectric backscatter and double-bounce returns.",
            required_inputs=["optical_image", "sar_image"],
            supported_modalities=["optical", "sar"],
            device="cpu",
            vram_mb=250,
            confidence_source="Cross-Modal Agreement Index"
        ))

        # 5. GeoTIFF / Geospatial Pipeline
        self.register(SpecialistMetadata(
            id="geotiff_engine",
            name="Rasterio Geospatial & CRS Engine",
            category="geospatial_preprocessing",
            description="Reads CRS, transforms, resolution, converts pixel bounding boxes to GeoJSON coordinates (WGS84/UTM).",
            required_inputs=["geotiff_path"],
            supported_modalities=["optical", "sar", "multispectral"],
            device="cpu",
            vram_mb=100,
            confidence_source="Deterministic Geospatial Header"
        ))

    def register(self, specialist: SpecialistMetadata):
        self._registry[specialist.id] = specialist

    def get(self, specialist_id: str) -> Optional[SpecialistMetadata]:
        return self._registry.get(specialist_id)

    def list_all(self) -> List[Dict[str, Any]]:
        return [asdict(spec) for spec in self._registry.values()]

    def find_for_task(self, task_name: str) -> Optional[SpecialistMetadata]:
        for spec in self._registry.values():
            if spec.category == task_name or spec.id == task_name:
                return spec
        return None
