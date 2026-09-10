"""
Optical remote sensing imagery preprocessing.
"""

from typing import Union, Tuple
import numpy as np
from PIL import Image
import cv2

from .geotiff import GeoTIFFProcessor


class OpticalPreprocessor:
    """
    Standardizes and enhances optical satellite/aerial imagery.
    """

    @staticmethod
    def load_image(
        image_input: Union[str, np.ndarray, Image.Image]
    ) -> Tuple[np.ndarray, dict]:
        """
        Loads optical image, handling GeoTIFF, PNG, JPG, and NumPy inputs.
        Returns uint8 RGB array (H, W, 3) and metadata.
        """
        metadata = {"modality": "optical"}

        if isinstance(image_input, str):
            if GeoTIFFProcessor.is_geotiff(image_input):
                rgb, geo_meta = GeoTIFFProcessor.read_as_rgb(image_input)
                metadata.update(geo_meta)
                return rgb, metadata
            else:
                pil_img = Image.open(image_input).convert("RGB")
                arr = np.array(pil_img)
                metadata["shape"] = (pil_img.height, pil_img.width)
                metadata["format"] = pil_img.format or "RGB"
                return arr, metadata

        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            arr = np.array(pil_img)
            metadata["shape"] = (pil_img.height, pil_img.width)
            metadata["format"] = "PIL.Image"
            return arr, metadata

        elif isinstance(image_input, np.ndarray):
            arr = image_input.copy()
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[2] == 4:
                arr = arr[:, :, :3]
            metadata["shape"] = arr.shape[:2]
            metadata["format"] = "numpy.ndarray"
            return arr.astype(np.uint8), metadata

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    @classmethod
    def process_optical(
        cls,
        image_input: Union[str, np.ndarray, Image.Image],
        enhance: bool = False
    ) -> Tuple[np.ndarray, dict]:
        """Loads and standardizes optical image."""
        rgb, meta = cls.load_image(image_input)
        if enhance:
            rgb = cls.enhance_contrast(rgb)
        return rgb, meta
