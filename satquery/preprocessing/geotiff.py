"""
GeoTIFF and Geospatial Pipeline Engine for SatQuery AI.
Handles multi-band inspection, CRS / affine transform reading, pixel-to-geographic coordinate
mapping (WGS84 / UTM), spatial intersection verification, and GeoJSON export.
"""

import os
import json
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.transform import xy as rasterio_xy
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from shapely.geometry import box, mapping, Polygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


class GeoTIFFProcessor:
    """
    Geospatial engine for GeoTIFF / multispectral remote sensing data.
    Preserves and propagates CRS, transform matrices, real ground resolution, and coordinates.
    """

    @staticmethod
    def is_geotiff(file_path: str) -> bool:
        """Checks if a file path is a TIFF/GeoTIFF."""
        if not isinstance(file_path, str):
            return False
        ext = os.path.splitext(file_path)[1].lower()
        return ext in [".tif", ".tiff", ".geotiff"]

    @classmethod
    def inspect(cls, file_path: str) -> Dict[str, Any]:
        """Inspects geospatial header, CRS, bounds, transform, and resolution."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"GeoTIFF file not found: {file_path}")

        meta: Dict[str, Any] = {
            "filename": os.path.basename(file_path),
            "file_path": file_path,
            "format": "TIFF/GeoTIFF",
            "has_geospatial_meta": False,
            "crs": None,
            "crs_epsg": None,
            "bounds": None,
            "transform": None,
            "resolution": None,
            "bands": 1,
            "shape": (),
            "dtype": "unknown"
        }

        if HAS_RASTERIO:
            try:
                with rasterio.open(file_path) as src:
                    meta["shape"] = (src.height, src.width)
                    meta["bands"] = src.count
                    meta["dtype"] = str(src.dtypes[0])
                    meta["crs"] = str(src.crs) if src.crs else "EPSG:4326 (Default Geographic)"
                    meta["crs_epsg"] = src.crs.to_epsg() if src.crs else 4326
                    meta["bounds"] = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
                    meta["transform"] = list(src.transform) if src.transform else None
                    meta["resolution"] = [abs(src.res[0]), abs(src.res[1])] if src.res else [0.5, 0.5]
                    meta["has_geospatial_meta"] = bool(src.crs or src.transform)
                    return meta
            except Exception:
                pass

        if HAS_TIFFFILE:
            try:
                with tifffile.TiffFile(file_path) as tif:
                    page = tif.pages[0]
                    meta["shape"] = page.shape[:2] if len(page.shape) >= 2 else page.shape
                    meta["bands"] = page.shape[2] if len(page.shape) == 3 else 1
                    meta["dtype"] = str(page.dtype)
                    meta["has_geospatial_meta"] = True
                    return meta
            except Exception:
                pass

        with Image.open(file_path) as img:
            meta["shape"] = (img.height, img.width)
            meta["bands"] = len(img.getbands()) if hasattr(img, "getbands") else 1
            meta["dtype"] = img.mode

        return meta

    @classmethod
    def pixel_to_geo(
        cls,
        meta_or_path: Union[str, Dict[str, Any]],
        pixel_x: float,
        pixel_y: float
    ) -> Tuple[float, float]:
        """
        Converts pixel coordinates (column, row) to geographic coordinates (X / Lon, Y / Lat).
        """
        meta = cls.inspect(meta_or_path) if isinstance(meta_or_path, str) else meta_or_path
        
        transform = meta.get("transform")
        if transform and len(transform) >= 6:
            # Affine: X = a * col + b * row + c ; Y = d * col + e * row + f
            a, b, c, d, e, f = transform[:6]
            geo_x = a * pixel_x + b * pixel_y + c
            geo_y = d * pixel_x + e * pixel_y + f
            return round(geo_x, 6), round(geo_y, 6)
            
        # Fallback approximation from bounds if transform unavailable
        bounds = meta.get("bounds")
        shape = meta.get("shape")
        if bounds and shape and len(shape) >= 2 and shape[0] > 0 and shape[1] > 0:
            left, bottom, right, top = bounds
            geo_x = left + (pixel_x / shape[1]) * (right - left)
            geo_y = top - (pixel_y / shape[0]) * (top - bottom)
            return round(geo_x, 6), round(geo_y, 6)

        # Default synthetic grid coordinate fallback
        return round(float(pixel_x * 0.00001), 6), round(float(pixel_y * 0.00001), 6)

    @classmethod
    def bbox_to_geojson_polygon(
        cls,
        meta_or_path: Union[str, Dict[str, Any]],
        bbox: Union[List[int], Tuple[int, int, int, int]]
    ) -> Dict[str, Any]:
        """
        Converts pixel bounding box [ymin, xmin, ymax, xmax] into a GeoJSON Polygon Feature.
        """
        ymin, xmin, ymax, xmax = bbox
        
        # 4 corners
        p1 = cls.pixel_to_geo(meta_or_path, xmin, ymin)  # Top-left
        p2 = cls.pixel_to_geo(meta_or_path, xmax, ymin)  # Top-right
        p3 = cls.pixel_to_geo(meta_or_path, xmax, ymax)  # Bottom-right
        p4 = cls.pixel_to_geo(meta_or_path, xmin, ymax)  # Bottom-left
        
        coordinates = [[
            [p1[0], p1[1]],
            [p2[0], p2[1]],
            [p3[0], p3[1]],
            [p4[0], p4[1]],
            [p1[0], p1[1]]  # Closed ring
        ]]
        
        return {
            "type": "Polygon",
            "coordinates": coordinates
        }

    @classmethod
    def check_spatial_overlap(
        cls,
        meta_1: Dict[str, Any],
        meta_2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluates whether two geospatial images cover overlapping geographic areas.
        """
        b1, b2 = meta_1.get("bounds"), meta_2.get("bounds")
        
        if not b1 or not b2 or not HAS_SHAPELY:
            return {
                "overlap_status": "UNVERIFIED (Missing Bounds / Non-Georeferenced)",
                "overlap_percentage": 100.0,
                "has_overlap": True
            }
            
        try:
            poly1 = box(b1[0], b1[1], b1[2], b1[3])
            poly2 = box(b2[0], b2[1], b2[2], b2[3])
            
            if not poly1.intersects(poly2):
                return {
                    "overlap_status": "NO_OVERLAP",
                    "overlap_percentage": 0.0,
                    "has_overlap": False,
                    "warning": "Images do not intersect geographically. Change detection may produce invalid results."
                }
                
            intersection = poly1.intersection(poly2)
            pct = (intersection.area / min(poly1.area, poly2.area)) * 100.0
            
            return {
                "overlap_status": "VERIFIED_OVERLAP",
                "overlap_percentage": round(pct, 2),
                "has_overlap": pct > 10.0,
                "intersection_bounds": list(intersection.bounds)
            }
        except Exception as e:
            return {
                "overlap_status": f"ERROR: {str(e)}",
                "overlap_percentage": 100.0,
                "has_overlap": True
            }

    @classmethod
    def export_geojson_report(
        cls,
        meta: Dict[str, Any],
        bboxes: List[Dict[str, Any]],
        change_percentage: float,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Exports detected change clusters as a standardized GeoJSON FeatureCollection.
        """
        features = []
        for i, box_info in enumerate(bboxes):
            raw_box = box_info.get("bbox", [0, 0, 0, 0])
            poly_geom = cls.bbox_to_geojson_polygon(meta, raw_box)
            
            feature = {
                "type": "Feature",
                "id": i + 1,
                "properties": {
                    "cluster_id": i + 1,
                    "pixel_area": box_info.get("area"),
                    "pixel_bbox": raw_box,
                    "centroid_px": box_info.get("centroid"),
                    "detection_model": "ChangeFormerV6",
                    "scene_change_percentage": change_percentage
                },
                "geometry": poly_geom
            }
            features.append(feature)
            
        geojson_doc = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": meta.get("crs", "urn:ogc:def:crs:OGC:1.3:CRS84")}
            },
            "features": features
        }
        
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(geojson_doc, f, indent=2)
                
        return geojson_doc

    @classmethod
    def read_as_rgb(
        cls,
        file_path: str,
        bands: Optional[List[int]] = None,
        stretch_percentile: Tuple[float, float] = (2.0, 98.0)
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reads GeoTIFF data and normalizes into a 3-channel RGB uint8 matrix."""
        meta = cls.inspect(file_path)
        data: Optional[np.ndarray] = None

        if HAS_RASTERIO:
            try:
                with rasterio.open(file_path) as src:
                    arr = src.read()  # (C, H, W)
                    if arr.ndim == 3:
                        data = np.transpose(arr, (1, 2, 0))
                    else:
                        data = arr
            except Exception:
                data = None

        if data is None and HAS_TIFFFILE:
            try:
                data = tifffile.imread(file_path)
            except Exception:
                data = None

        if data is None:
            pil_img = Image.open(file_path)
            data = np.array(pil_img)

        # Standardize shape
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        elif data.ndim == 3 and data.shape[0] in [1, 2, 3, 4, 8, 12, 13] and data.shape[0] < data.shape[1]:
            data = np.transpose(data, (1, 2, 0))

        num_channels = data.shape[2]
        meta["bands"] = num_channels
        meta["min_val"] = float(np.nanmin(data))
        meta["max_val"] = float(np.nanmax(data))
        meta["mean_val"] = float(np.nanmean(data))

        if bands is not None and len(bands) >= 3:
            selected_indices = [min(b, num_channels - 1) for b in bands[:3]]
            rgb_data = data[:, :, selected_indices]
        elif num_channels >= 3:
            rgb_data = data[:, :, :3]
        elif num_channels == 1:
            rgb_data = np.repeat(data, 3, axis=2)
        else:
            rgb_data = np.dstack([data[:, :, 0], data[:, :, 1], data[:, :, 0]])

        # Percentile contrast stretch
        stretched_channels = []
        for c in range(3):
            ch = rgb_data[:, :, c].astype(np.float32)
            valid = ch[~np.isnan(ch)]
            if valid.size > 0:
                p_low, p_high = np.percentile(valid, stretch_percentile)
                if p_high > p_low:
                    ch_norm = np.clip((ch - p_low) / (p_high - p_low), 0.0, 1.0)
                else:
                    ch_norm = np.clip(ch / (meta["max_val"] + 1e-6), 0.0, 1.0)
            else:
                ch_norm = np.zeros_like(ch)
            stretched_channels.append((ch_norm * 255.0).astype(np.uint8))

        rgb_uint8 = np.dstack(stretched_channels)
        return rgb_uint8, meta
