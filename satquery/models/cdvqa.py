"""
Change Visual Question Answering (CDVQA) Interface for SatQuery AI.
Designed for bi-temporal remote-sensing change reasoning.
Explicitly distinguishes ChangeFormer (spatial change specialist) from CDVQA (language reasoning model).
"""

from typing import Dict, Any, Union, Optional
import numpy as np
from PIL import Image


class ChangeReasoningEngine:
    """
    Multimodal change reasoning and CDVQA interface.
    Explicitly reports NOT_CONFIGURED when no fine-tuned CDVQA language model is loaded.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.name = "CDVQA-Reasoner-Interface"
        self.is_configured = bool(model_path and len(model_path) > 0)

    def analyze_change(
        self,
        image_t1: Union[str, np.ndarray, Image.Image],
        image_t2: Union[str, np.ndarray, Image.Image],
        change_mask: np.ndarray,
        change_percentage: float,
        bboxes: list,
        user_question: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Interprets bi-temporal changes if a real reasoning model is loaded.
        """
        if not self.is_configured:
            cluster_info = f"{len(bboxes)} distinct change region{'s' if len(bboxes) != 1 else ''}"
            summary = (
                f"ChangeFormer detected spatial change across {change_percentage:.2f}% of the scene ({cluster_info}). "
                f"CDVQA language reasoning model is not configured to provide semantic interpretations."
            )
            return {
                "status": "NOT_CONFIGURED",
                "explanation": "CDVQA model not configured.",
                "answer": summary,
                "change_type": "Spatial Change Detected",
                "change_percentage": change_percentage,
                "num_clusters": len(bboxes),
                "confidence": None,
                "model": self.name,
                "spatial_evidence": {
                    "num_clusters": len(bboxes),
                    "top_bboxes": bboxes[:5]
                }
            }

        raise NotImplementedError("Real CDVQA model weights not loaded.")
