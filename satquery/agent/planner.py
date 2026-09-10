"""
Intelligent Task Planner and Dependency Graph for SatQuery AI.
Decomposes user queries into structured execution steps, verifies modal and geospatial prerequisites,
and sequences specialist AI models.
"""

from typing import Dict, Any, List, Optional
from .registry import ModelRegistry


class TaskPlanner:
    """
    Constructs an optimized execution plan based on input data and user query.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()

    def plan(
        self,
        has_image_1: bool,
        has_image_2: bool,
        modality_1: str,
        modality_2: Optional[str],
        question: Optional[str],
        explicit_task: Optional[str],
        is_geotiff_1: bool,
        is_geotiff_2: bool
    ) -> Dict[str, Any]:
        """
        Creates an ordered list of execution stages.
        """
        q_lower = (question or "").lower().strip()
        steps: List[Dict[str, Any]] = []

        # 1. Geospatial & Input Inspection
        steps.append({
            "stage": 1,
            "name": "Input & Modality Validation",
            "action": "validate_inputs",
            "description": f"Validate Image 1 ({modality_1})" + (f" and Image 2 ({modality_2})" if has_image_2 else "")
        })

        if is_geotiff_1 or is_geotiff_2:
            steps.append({
                "stage": 2,
                "name": "Geospatial CRS & Overlap Inspection",
                "action": "geotiff_inspection",
                "description": "Extract CRS, transform matrices, and verify spatial overlap"
            })

        # 2. Task Classification & Routing
        if explicit_task:
            task_type = explicit_task.lower()
        elif has_image_2:
            if (modality_1 == "optical" and modality_2 == "sar") or (modality_1 == "sar" and modality_2 == "optical") or "sar" in q_lower:
                task_type = "optical_sar_fusion"
            else:
                task_type = "change_analysis"
        elif any(k in q_lower for k in ["where", "locate", "ground", "find the", "highlight", "bounding box", "show me"]):
            task_type = "grounding"
        elif any(k in q_lower for k in ["how many", "count", "number of", "is there", "are there"]):
            task_type = "vqa"
        elif not question or any(k in q_lower for k in ["describe", "caption", "overview", "what is"]):
            task_type = "vqa"
        else:
            task_type = "vqa"

        # 3. Specialist Execution Stage
        if task_type in ["change_analysis", "bi_temporal"]:
            steps.append({
                "stage": 3,
                "name": "Bi-temporal Change Detection",
                "specialist_id": "change_detection",
                "action": "execute_changeformer",
                "description": "Run ChangeFormerV6 on aligned bi-temporal image pair"
            })
            steps.append({
                "stage": 4,
                "name": "Connected Component & Contour Extraction",
                "action": "extract_change_clusters",
                "description": "Segment changed regions and extract bounding box coordinates"
            })
            if is_geotiff_1:
                steps.append({
                    "stage": 5,
                    "name": "GeoJSON Export",
                    "action": "export_geojson",
                    "description": "Convert pixel change clusters to geographic polygons (WGS84/UTM)"
                })

        elif task_type in ["optical_sar_fusion", "optical_sar"]:
            steps.append({
                "stage": 3,
                "name": "Optical Spectral Preprocessing",
                "action": "process_optical",
                "description": "Compute vegetation (ExG) and water spectral indices"
            })
            steps.append({
                "stage": 4,
                "name": "SAR Speckle Filtering & Backscatter Scaling",
                "action": "process_sar",
                "description": "Apply Lee despeckling and extract double-bounce radar returns"
            })
            steps.append({
                "stage": 5,
                "name": "Cross-Modal Evidence Fusion",
                "specialist_id": "optical_sar_fusion",
                "action": "execute_fusion",
                "description": "Synthesize spectral and radar evidence with cloud penetration analysis"
            })

        elif task_type in ["grounding", "spatial_grounding"]:
            steps.append({
                "stage": 3,
                "name": "Open-Vocabulary Spatial Grounding",
                "specialist_id": "grounding",
                "action": "execute_grounding",
                "description": f"Extract multi-scale feature maps to locate target: '{question or 'structures'}'"
            })

        else:  # vqa / captioning
            steps.append({
                "stage": 3,
                "name": "Remote Sensing Vision-Language QA",
                "specialist_id": "vqa",
                "action": "execute_vqa",
                "description": f"Analyze satellite image features for question: '{question or 'scene description'}'"
            })

        # 4. Confidence & Evidence Synthesis
        steps.append({
            "stage": len(steps) + 1,
            "name": "Multi-Modal Confidence & Report Synthesis",
            "action": "synthesize_report",
            "description": "Aggregate model probabilities and generate structured Markdown & JSON evidence"
        })

        return {
            "determined_task": task_type,
            "total_stages": len(steps),
            "execution_plan": steps
        }
