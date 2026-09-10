"""
Remote Sensing Vision-Language Model (RS-VLM) Interface for SatQuery AI.
Designed for integration with fine-tuned remote-sensing VLMs (BigEarthNet, RSVQA, VRSBench).
Explicitly marks unconfigured status when no real model checkpoint is loaded.
"""

from typing import Dict, Any, Union, Optional
import numpy as np
from PIL import Image


class RemoteSensingVLM:
    """
    Modular Remote Sensing Vision-Language Model interface.
    Explicitly reports NOT_CONFIGURED when no trained weights are loaded.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.name = "RS-VLM-Interface"
        self.is_configured = bool(model_path and len(model_path) > 0)

    def generate_caption(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Generates descriptive remote sensing captioning if a model is loaded.
        """
        if not self.is_configured:
            return {
                "status": "NOT_CONFIGURED",
                "caption": "Remote-sensing VLM is not configured.",
                "primary_land_cover": None,
                "detected_land_covers": [],
                "confidence": None,
                "model": self.name
            }

        raise NotImplementedError("Real VLM weights not loaded.")

    def answer_question(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        question: str
    ) -> Dict[str, Any]:
        """
        Answers visual questions if a model is loaded.
        """
        if not self.is_configured:
            return {
                "status": "NOT_CONFIGURED",
                "question": question,
                "answer": "Remote-sensing VLM is not configured.",
                "confidence": None,
                "model": self.name,
                "evidence": {}
            }

        raise NotImplementedError("Real VLM weights not loaded.")
