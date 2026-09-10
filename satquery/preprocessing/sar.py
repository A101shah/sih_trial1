"""
Synthetic Aperture Radar (SAR) remote sensing preprocessing.
Handles backscatter conversion (dB scale), speckle noise reduction, and SAR-specific enhancement.
"""

from typing import Union, Tuple, Optional
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import uniform_filter

from .geotiff import GeoTIFFProcessor


class SARPreprocessor:
    """
    Specialized preprocessor for Synthetic Aperture Radar (SAR / Sentinel-1 / TerraSAR-X) imagery.
    """

    @staticmethod
    def lee_filter(img_2d: np.ndarray, size: int = 5, damping_factor: float = 1.0) -> np.ndarray:
        """
        Lee filter for speckle reduction in SAR imagery.
        Preserves edges while smoothing homogeneous radar backscatter regions.
        """
        img = img_2d.astype(np.float32)
        mean = uniform_filter(img, size)
        sqr_mean = uniform_filter(img ** 2, size)
        var = sqr_mean - mean ** 2
        var = np.maximum(var, 0)

        # Overall noise variance estimation
        overall_var = np.var(img) + 1e-6
        weights = var / (var + overall_var / damping_factor)
        weights = np.clip(weights, 0.0, 1.0)

        filtered = mean + weights * (img - mean)
        return filtered

    @classmethod
    def process_sar(
        cls,
        sar_input: Union[str, np.ndarray, Image.Image],
        to_db: bool = True,
        apply_speckle_filter: bool = True,
        filter_size: int = 5
    ) -> Tuple[np.ndarray, dict]:
        """
        Processes raw SAR data into normalized 3-channel visualization suitable for VLM and vision models.
        Returns:
            rgb_vis: (H, W, 3) uint8 image with SAR backscatter dynamics preserved
            meta: dictionary with SAR characteristics (dB range, speckle stats)
        """
        meta = {"modality": "sar"}

        if isinstance(sar_input, str):
            if GeoTIFFProcessor.is_geotiff(sar_input):
                raw_data, geo_meta = GeoTIFFProcessor.read_as_rgb(sar_input)
                meta.update(geo_meta)
                single_band = raw_data[:, :, 0].astype(np.float32)
            else:
                pil_img = Image.open(sar_input).convert("L")
                single_band = np.array(pil_img, dtype=np.float32)
                meta["shape"] = (pil_img.height, pil_img.width)
        elif isinstance(sar_input, Image.Image):
            pil_img = sar_input.convert("L")
            single_band = np.array(pil_img, dtype=np.float32)
            meta["shape"] = (pil_img.height, pil_img.width)
        elif isinstance(sar_input, np.ndarray):
            if sar_input.ndim == 3:
                single_band = sar_input[:, :, 0].astype(np.float32)
            else:
                single_band = sar_input.astype(np.float32)
            meta["shape"] = single_band.shape[:2]
        else:
            raise TypeError(f"Unsupported SAR input type: {type(sar_input)}")

        # Convert linear intensity to dB scale if needed
        if to_db:
            # Shift positive if negative values exist
            min_val = np.nanmin(single_band)
            if min_val <= 0:
                shifted = single_band - min_val + 1.0
            else:
                shifted = single_band + 1e-6
            sar_db = 10.0 * np.log10(shifted)
        else:
            sar_db = single_band

        # Apply speckle suppression filter
        if apply_speckle_filter:
            sar_filtered = cls.lee_filter(sar_db, size=filter_size)
        else:
            sar_filtered = sar_db

        # Percentile contrast stretch for SAR visualization
        p_low, p_high = np.percentile(sar_filtered, (1.0, 99.0))
        if p_high > p_low:
            sar_norm = np.clip((sar_filtered - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            sar_norm = np.zeros_like(sar_filtered)

        sar_uint8 = (sar_norm * 255.0).astype(np.uint8)

        # Create 3-channel pseudo-RGB for neural network backbones
        # Channel 0: Despeckled SAR, Channel 1: High-frequency texture (gradient), Channel 2: Raw intensity
        grad_x = cv2.Sobel(sar_uint8, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(sar_uint8, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_uint8 = np.clip(grad_mag, 0, 255).astype(np.uint8)

        sar_rgb = np.dstack([sar_uint8, grad_uint8, sar_uint8])

        meta["db_range"] = [float(p_low), float(p_high)]
        meta["speckle_filtered"] = apply_speckle_filter

        return sar_rgb, meta
