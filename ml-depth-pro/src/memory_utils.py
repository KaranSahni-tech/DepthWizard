import os
import sys
import gc
import numpy as np
import cv2
from PIL import Image

def estimate_array_mb(arr):
    if arr is None or not isinstance(arr, np.ndarray):
        return 0.0
    return float(arr.nbytes) / (1024.0 * 1024.0)

def log_memory_usage(stage_name, arrays_dict):
    total_mb = 0.0
    details = []
    for name, arr in arrays_dict.items():
        if isinstance(arr, np.ndarray):
            mb = estimate_array_mb(arr)
            total_mb += mb
            details.append(f"{name}: {arr.shape} {arr.dtype} ~{mb:.2f}MB")
    print(f"[MEMORY][{stage_name}] Total tracked memory: {total_mb:.2f}MB | Details: {', '.join(details)}")
    return total_mb

def safe_float(val, default=0.0, name="value"):
    """
    Safely converts input value to float, handling None, strings, 0D numpy arrays,
    torch tensors, and replacing NaN/Inf with default. Never crashes on NoneType.
    """
    if val is None:
        return float(default)
    try:
        if hasattr(val, "item"):
            val = val.item()
        res = float(val)
        if not np.isfinite(res):
            return float(default)
        return res
    except (TypeError, ValueError):
        return float(default)

def safe_int(val, default=0, name="value"):
    """
    Safely converts input value to int, handling None, strings, numpy scalars.
    """
    if val is None:
        return int(default)
    try:
        if hasattr(val, "item"):
            val = val.item()
        res = float(val)
        if not np.isfinite(res):
            return int(default)
        return int(res)
    except (TypeError, ValueError):
        return int(default)

def validate_and_normalize_metadata(meta_dict, image_w=None, image_h=None):
    """
    Validates and normalizes metadata variations (focal_length, focal_length_px, focalLength,
    width, image_width, height, image_height) into a unified internal schema.
    """
    if not isinstance(meta_dict, dict):
        meta_dict = {}

    w = safe_int(meta_dict.get("width") or meta_dict.get("image_width") or meta_dict.get("imageWidth") or image_w, default=image_w or 1400)
    h = safe_int(meta_dict.get("height") or meta_dict.get("image_height") or meta_dict.get("imageHeight") or image_h, default=image_h or 1400)

    fl_raw = meta_dict.get("focal_length_px") or meta_dict.get("focal_length") or meta_dict.get("focalLength") or meta_dict.get("focalLengthPx") or meta_dict.get("fx")
    
    if fl_raw is not None and safe_float(fl_raw, default=0.0) > 0.0:
        f_px = safe_float(fl_raw)
    else:
        # Documented fallback camera focal length estimation based on field of view (~60 deg)
        f_px = 1342.898681640625 * (w / 1400.0)

    cx = safe_float(meta_dict.get("cx") or meta_dict.get("principal_point_x"), default=w / 2.0)
    cy = safe_float(meta_dict.get("cy") or meta_dict.get("principal_point_y"), default=h / 2.0)

    return {
        "image": {
            "width": w,
            "height": h
        },
        "depth": {
            "width": w,
            "height": h,
            "shape": [h, w],
            "dtype": "float32",
            "units": "meters"
        },
        "camera": {
            "focal_length_px": f_px,
            "cx": cx,
            "cy": cy
        }
    }

def to_uint8_image(img_arr):
    """
    Safely converts an input image to uint8 RGB (3-channel) or Grayscale (1-channel),
    discarding unnecessary alpha channels and avoiding float64 allocations.
    """
    if img_arr is None:
        return None
    
    if img_arr.dtype == np.uint8:
        if img_arr.ndim == 3 and img_arr.shape[2] == 4:
            return img_arr[:, :, :3]  # Discard alpha channel for structure processing
        return img_arr
    
    # Handle float inputs [0.0, 1.0] or [0, 255]
    if np.issubdtype(img_arr.dtype, np.floating):
        max_val = np.nanmax(img_arr)
        if max_val <= 1.0:
            uint8_arr = np.clip(img_arr * 255.0, 0, 255).astype(np.uint8)
        else:
            uint8_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        
        if uint8_arr.ndim == 3 and uint8_arr.shape[2] == 4:
            uint8_arr = uint8_arr[:, :, :3]
        return uint8_arr
        
    return img_arr.astype(np.uint8)

def to_float32_depth(depth_arr):
    """
    Ensures depth maps remain float32, preventing float64 silent type promotion.
    """
    if depth_arr is None:
        return None
    if depth_arr.dtype == np.float32:
        return depth_arr
    return depth_arr.astype(np.float32, copy=False)

def clean_memory():
    """Triggers Python garbage collection to free unreferenced arrays."""
    gc.collect()

def save_colormap_safe(data, path, cmap="inferno", vmin=0.0, vmax=1.0):
    """
    Memory-safe colormap image saver. Uses OpenCV C++ colormaps on uint8 arrays
    instead of Matplotlib figure canvases, avoiding 39.3 MiB float64 RGBA array allocations.
    """
    if data is None:
        return
    
    data_f32 = data.astype(np.float32, copy=False) if isinstance(data, np.ndarray) else np.array(data, dtype=np.float32)
    
    if vmin is None:
        vmin = safe_float(np.nanmin(data_f32), default=0.0)
    if vmax is None:
        vmax = safe_float(np.nanmax(data_f32), default=1.0)
        
    if vmax > vmin:
        norm = np.clip((data_f32 - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        norm = np.zeros_like(data_f32, dtype=np.float32)
        
    norm = np.nan_to_num(norm, nan=0.0)
    gray_u8 = (norm * 255.0).astype(np.uint8)
    
    cmap_map = {
        "turbo": cv2.COLORMAP_TURBO,
        "magma": cv2.COLORMAP_MAGMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "inferno": cv2.COLORMAP_INFERNO,
        "plasma": cv2.COLORMAP_PLASMA,
        "jet": cv2.COLORMAP_JET,
        "rainbow": cv2.COLORMAP_RAINBOW
    }
    
    if cmap == "gray" or cmap == "grayscale":
        Image.fromarray(gray_u8).save(path)
        return
        
    cv_cmap = cmap_map.get(str(cmap).lower(), cv2.COLORMAP_INFERNO)
    bgr = cv2.applyColorMap(gray_u8, cv_cmap)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(rgb).save(path)

def safe_scharr_gradients(depth_clean):
    """
    Computes Scharr gradients using float32 (CV_32F) instead of float64 (CV_64F).
    """
    depth_f32 = to_float32_depth(depth_clean)
    grad_x = cv2.Scharr(depth_f32, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(depth_f32, cv2.CV_32F, 0, 1)
    return grad_x, grad_y
