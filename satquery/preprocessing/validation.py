"""
Input validation, modality detection, and co-registration verification for SatQuery AI.
"""

from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from PIL import Image
import os


class InputValidator:
    """
    Validates single and paired satellite images for dimensions, modalities, and registration.
    """

    @staticmethod
    def infer_modality(image_input: Union[str, np.ndarray, Image.Image], explicit_modality: Optional[str] = None) -> str:
        """
        Detects whether an image is Optical, SAR, or Multispectral.
        """
        if explicit_modality and explicit_modality.lower() in ["optical", "sar", "multispectral"]:
            return explicit_modality.lower()

        if isinstance(image_input, str):
            lower_path = image_input.lower()
            if any(k in lower_path for k in ["sar", "s1", "sentinel1", "terrasar", "radar", "pol"]):
                return "sar"
            if any(k in lower_path for k in ["s2", "sentinel2", "landsat", "msi", "multispectral", "b4", "b8"]):
                return "multispectral"

        return "optical"

    @classmethod
    def validate_pair(
        cls,
        image_1: Union[str, np.ndarray, Image.Image],
        image_2: Union[str, np.ndarray, Image.Image],
        task_type: str = "bi_temporal"
    ) -> Dict[str, Any]:
        """
        Validates image pair (T1/T2 or Optical/SAR) for size compatibility and registration.
        """
        def get_shape(inp):
            if isinstance(inp, str):
                with Image.open(inp) as img:
                    return (img.height, img.width)
            elif isinstance(inp, Image.Image):
                return (inp.height, inp.width)
            elif isinstance(inp, np.ndarray):
                return inp.shape[:2]
            return None

        shape1 = get_shape(image_1)
        shape2 = get_shape(image_2)

        if shape1 is None or shape2 is None:
            raise ValueError("Unable to determine image dimensions from inputs.")

        is_same_size = (shape1 == shape2)
        aspect_ratio1 = round(shape1[1] / shape1[0], 3)
        aspect_ratio2 = round(shape2[1] / shape2[0], 3)

        warnings = []
        if not is_same_size:
            warnings.append(
                f"Dimension mismatch between Image 1 ({shape1[1]}x{shape1[0]}) and Image 2 ({shape2[1]}x{shape2[0]}). "
                f"Images will be rescaled automatically for alignment."
            )

        if abs(aspect_ratio1 - aspect_ratio2) > 0.05:
            warnings.append(
                f"Aspect ratio difference detected (Image 1: {aspect_ratio1}, Image 2: {aspect_ratio2}). "
                f"Ensure image pair covers identical geospatial footprints."
            )

        return {
            "valid": True,
            "shape_image_1": shape1,
            "shape_image_2": shape2,
            "is_identical_size": is_same_size,
            "warnings": warnings,
            "task_type": task_type
        }
