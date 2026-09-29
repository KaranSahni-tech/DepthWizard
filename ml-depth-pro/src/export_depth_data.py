import os
import sys
import shutil
import json
import time
import math
import numpy as np
import cv2
from PIL import Image
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import depth_pro
from memory_utils import (
    to_float32_depth, to_uint8_image, save_colormap_safe,
    clean_memory, safe_float
)
from detect_boundaries import detect_boundaries
from segment_scene import segment_scene
from estimate_ground_surface import estimate_ground_surface
from estimate_building_heights import estimate_building_heights

# Check rasterio for GeoTIFF metadata
try:
    import rasterio
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

def compute_hillshade(dem, azimuth_deg=315.0, altitude_deg=45.0, z_factor=1.0):
    """
    Computes professional cartographic shaded relief (hillshade) from an elevation/depth surface.
    """
    dem_f = dem.astype(np.float32)
    # Sobel gradients
    dz_dx = cv2.Sobel(dem_f, cv2.CV_32F, 1, 0, ksize=3) * z_factor
    dz_dy = cv2.Sobel(dem_f, cv2.CV_32F, 0, 1, ksize=3) * z_factor

    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect = np.arctan2(dz_dy, -dz_dx)

    zenith_rad = np.radians(90.0 - altitude_deg)
    azimuth_math = 360.0 - azimuth_deg + 90.0
    azimuth_rad = np.radians(azimuth_math % 360.0)

    shaded = 255.0 * (
        (np.cos(zenith_rad) * np.cos(slope)) +
        (np.sin(zenith_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))
    )
    return np.clip(shaded, 0, 255).astype(np.uint8)

def render_colorbar_map(data, cmap_name, label_text, min_val, max_val, unit="relative"):
    """
    Renders an elevation/depth map with a clean horizontal colorbar and annotation banner at bottom.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    h, w = data.shape
    dpi = 120
    fig_w = max(6.0, w / float(dpi))
    fig_h = max(5.0, h / float(dpi) + 0.9)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor('#0d0e12')
    ax.set_facecolor('#0d0e12')

    norm_data = np.clip(data, min_val, max_val)
    im = ax.imshow(norm_data, cmap=cmap_name, vmin=min_val, vmax=max_val)
    ax.axis('off')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.03)
    cbar.ax.tick_params(colors='#e2e8f0', labelsize=8)
    cbar.set_label(f"{label_text} [{unit}]", color='#e2e8f0', fontsize=9, fontweight='bold', labelpad=4)
    cbar.outline.set_edgecolor('#334155')

    plt.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(rgba[:, :, :3])

def main():
    start_time = time.time()

    if len(sys.argv) >= 3:
        input_path = sys.argv[1]
        output_dir = sys.argv[2]
        progress_file = sys.argv[3] if len(sys.argv) > 3 else None
    else:
        print("Usage: python export_depth_data.py <input_image> <output_dir> [progress_file]")
        sys.exit(1)

    # Clean standardized output structure
    maps_dir = os.path.join(output_dir, "maps")
    data_dir = os.path.join(output_dir, "data")
    source_dir = os.path.join(output_dir, "source")
    for d in [output_dir, maps_dir, data_dir, source_dir]:
        os.makedirs(d, exist_ok=True)

    generation_id = os.path.basename(output_dir)

    def write_step(step_idx, step_name, progress, msg=""):
        if progress_file:
            try:
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "status": "running",
                        "step_index": step_idx,
                        "step_name": step_name,
                        "stage": f"Step {step_idx}/8: {step_name}",
                        "progress": progress,
                        "message": msg or f"Step {step_idx}/8: {step_name}...",
                        "session_id": generation_id,
                        "generation_id": generation_id,
                        "timestamp": time.time()
                    }, f, indent=2)
            except Exception as pe:
                print(f"[PROGRESS_WARN] Failed writing progress: {pe}")
        print(f"[PIPELINE] STEP {step_idx}/8 [{step_name}] ({progress}%): {msg}")

    def write_error(step_idx, step_name, err_msg, cause=""):
        print(f"\n[ERROR] Step {step_idx}/8 [{step_name}] failed: {err_msg} - Cause: {cause}")
        if progress_file:
            try:
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "status": "error",
                        "step_index": step_idx,
                        "step_name": step_name,
                        "stage": f"Step {step_idx}/8: {step_name}",
                        "error": str(err_msg),
                        "cause": str(cause),
                        "message": f"Error in {step_name}: {err_msg}",
                        "session_id": generation_id,
                        "generation_id": generation_id,
                        "timestamp": time.time()
                    }, f, indent=2)
            except Exception as pe:
                print(f"[PROGRESS_WARN] Failed writing error to progress: {pe}")

    try:
        # ==============================================================================
        # STEP 1: LOAD IMAGE & DETECT GEOSPATIAL METADATA
        # ==============================================================================
        write_step(1, "Input & Geospatial Analysis", 10, "Detecting image format and reading geospatial metadata...")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found at: {input_path}")

        file_size_bytes = os.path.getsize(input_path)
        input_ext = os.path.splitext(input_path)[1].lower()

        # Save pristine source copy
        source_copy_path = os.path.join(source_dir, os.path.basename(input_path))
        if os.path.abspath(input_path) != os.path.abspath(source_copy_path):
            shutil.copy2(input_path, source_copy_path)

        # Geospatial detection
        is_geotiff = False
        crs_str = None
        epsg_code = None
        geo_transform = None
        geo_bounds = None
        pixel_size = None
        geo_coords = None
        raw_img = None
        input_format = "JPEG"

        if input_ext in [".tif", ".tiff"] and HAS_RASTERIO:
            try:
                with rasterio.open(input_path) as src:
                    if src.crs is not None:
                        is_geotiff = True
                        input_format = "GeoTIFF"
                        crs_str = str(src.crs)
                        try:
                            epsg_code = src.crs.to_epsg()
                        except Exception:
                            epsg_code = None
                        geo_transform = list(src.transform)
                        geo_bounds = [float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top)]
                        pixel_size = [abs(float(src.res[0])), abs(float(src.res[1]))]
                        geo_coords = {
                            "center": [float((src.bounds.left + src.bounds.right) / 2.0), float((src.bounds.top + src.bounds.bottom) / 2.0)],
                            "bounds": geo_bounds
                        }
                    else:
                        input_format = "TIFF"

                    if src.count >= 3:
                        r = src.read(1)
                        g = src.read(2)
                        b = src.read(3)
                        raw_img = np.dstack((r, g, b))
                    elif src.count == 1:
                        mono = src.read(1)
                        raw_img = np.dstack((mono, mono, mono))
            except Exception as tiff_err:
                print(f"[INFO] rasterio read note: {tiff_err}. Using PIL fallback.")

        if raw_img is None:
            pil_img = Image.open(input_path)
            input_format = pil_img.format or ("PNG" if input_ext == ".png" else "JPEG")
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            raw_img = np.array(pil_img)

        # Normalize pixel values to uint8 RGB
        if raw_img.dtype != np.uint8:
            valid_px = raw_img[np.isfinite(raw_img) & (raw_img > 0)]
            if len(valid_px) > 0:
                p2 = np.percentile(valid_px, 2)
                p98 = np.percentile(valid_px, 98)
                denom = p98 - p2 if p98 > p2 else 1.0
                norm_f = np.clip((raw_img - p2) / denom, 0.0, 1.0)
                img_np = (norm_f * 255.0).astype(np.uint8)
            else:
                img_np = np.zeros(raw_img.shape, dtype=np.uint8)
        else:
            img_np = raw_img

        if len(img_np.shape) == 2:
            img_np = np.dstack((img_np, img_np, img_np))
        elif img_np.shape[2] > 3:
            img_np = img_np[:, :, :3]

        h, w = img_np.shape[:2]
        if w < 16 or h < 16:
            raise ValueError(f"Input image dimensions too small: {w}x{h}")

        # MAP 1: Original RGB Image
        rgb_map_path = os.path.join(maps_dir, "rgb.png")
        Image.fromarray(img_np).save(rgb_map_path, quality=95)
        # Also copy to root for compatibility
        Image.fromarray(img_np).save(os.path.join(output_dir, "original_image.jpg"), quality=95)
        Image.fromarray(img_np).save(os.path.join(output_dir, "image.jpg"), quality=95)

        # ==============================================================================
        # STEP 2: MONOCULAR DEPTH ESTIMATION
        # ==============================================================================
        write_step(2, "AI Depth Estimation", 25, "Running monocular depth neural estimation...")

        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        precision = torch.half if device.type == "cuda" else torch.float32

        model, transform = depth_pro.create_model_and_transforms(precision=precision)
        model.to(device).eval()

        image_tensor = transform(img_np).to(device)

        try:
            _, _, exif_f_px = depth_pro.load_rgb(rgb_map_path)
            if exif_f_px is not None and float(exif_f_px) > 0:
                f_px = float(exif_f_px)
            else:
                f_px = 1342.898681640625 * (w / 1400.0)
        except Exception:
            f_px = 1342.898681640625 * (w / 1400.0)

        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    prediction = model.infer(image_tensor, f_px=f_px)
            else:
                prediction = model.infer(image_tensor, f_px=f_px)

        raw_depth = prediction["depth"].cpu().numpy().astype(np.float32)

        if "focallength_px" in prediction and prediction["focallength_px"] is not None:
            f_val = prediction["focallength_px"]
            f_px = float(f_val.item() if isinstance(f_val, torch.Tensor) else f_val)

        del model, transform, image_tensor, prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        clean_memory()

        if raw_depth.shape != (h, w):
            raw_depth = cv2.resize(raw_depth, (w, h), interpolation=cv2.INTER_CUBIC)

        # Save depth arrays
        np.save(os.path.join(data_dir, "depth.npy"), raw_depth)
        np.save(os.path.join(output_dir, "depth.npy"), raw_depth)

        # ==============================================================================
        # STEP 3: DEPTH CLEANING & STATISTICS
        # ==============================================================================
        write_step(3, "Depth Cleaning & Statistics", 40, "Filtering outliers and computing robust depth statistics...")

        valid_mask = np.isfinite(raw_depth) & (raw_depth > 0)
        valid_pixels_count = int(np.sum(valid_mask))
        nan_count = int(np.isnan(raw_depth).sum())
        inf_count = int(np.isinf(raw_depth).sum())

        valid_vals = raw_depth[valid_mask] if valid_pixels_count > 0 else np.array([1.0], dtype=np.float32)
        min_depth = float(np.min(valid_vals))
        max_depth = float(np.max(valid_vals))
        mean_depth = float(np.mean(valid_vals))
        median_depth = float(np.median(valid_vals))
        std_depth = float(np.std(valid_vals))
        p01 = float(np.percentile(valid_vals, 1.0))
        p02 = float(np.percentile(valid_vals, 2.0))
        p05 = float(np.percentile(valid_vals, 5.0))
        p95 = float(np.percentile(valid_vals, 95.0))
        p98 = float(np.percentile(valid_vals, 98.0))
        p99 = float(np.percentile(valid_vals, 99.0))

        clean_depth = np.copy(raw_depth)
        clean_depth[(raw_depth < p01) | (raw_depth > p99) | (~valid_mask)] = np.nan
        np.save(os.path.join(data_dir, "clean_depth.npy"), clean_depth)

        # MAP 2: Depth Map (Continuous colormap with min, max, mean, median)
        depth_norm = np.clip((raw_depth - p02) / (p98 - p02 + 1e-6), 0.0, 1.0)
        depth_u8 = ((1.0 - depth_norm) * 255.0).astype(np.uint8)
        depth_turbo_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        depth_turbo_rgb = cv2.cvtColor(depth_turbo_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(depth_turbo_rgb).save(os.path.join(maps_dir, "depth_map.png"))
        Image.fromarray(depth_turbo_rgb).save(os.path.join(output_dir, "depth_heatmap.png"))

        # ==============================================================================
        # STEP 4: BOUNDARIES, SEGMENTATION & GROUND ESTIMATION
        # ==============================================================================
        write_step(4, "Surface Feature Extraction", 55, "Detecting structural edges, segmenting landcover, and ground surface...")

        npy_path = os.path.join(output_dir, "depth.npy")
        try:
            detect_boundaries(rgb_map_path, npy_path, output_dir)
        except Exception as bnd_e:
            print(f"[BOUNDARY_NOTE] {bnd_e}. Computing fallback boundary map.")
            gx = cv2.Sobel(raw_depth, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(raw_depth, cv2.CV_32F, 0, 1, ksize=3)
            g_mag = np.sqrt(gx**2 + gy**2)
            g_norm = np.clip(g_mag / (np.percentile(g_mag, 98) + 1e-6), 0.0, 1.0)
            np.save(os.path.join(output_dir, "boundary_confidence.npy"), g_norm.astype(np.float32))

        boundary_npy_path = os.path.join(output_dir, "boundary_confidence.npy")
        boundary = np.load(boundary_npy_path)

        seg_config = {
            "ground_percentile": 95.0,
            "building_min_h": 0.2,
            "ground_max_h": 0.1,
            "road_max_h": 0.08,
            "road_sat_max": 80,
            "road_roughness_max": 0.08,
            "exg_thresh": 0.04,
            "veg_hue_min": 25,
            "veg_hue_max": 85,
            "veg_sat_min": 30,
            "roughness_thresh": 0.12,
            "boundary_thresh": 0.35,
            "min_building_area": 50,
            "overlay_alpha": 0.45
        }
        segment_scene(rgb_map_path, npy_path, boundary_npy_path, output_dir, seg_config)
        seg_npy_path = os.path.join(output_dir, "segmentation_mask.npy")
        seg_mask = np.load(seg_npy_path)

        ground_config = {
            "smooth_kernel_size": 51,
            "smooth_sigma": 20.0,
            "max_boundary_seed_thresh": 0.4
        }
        estimate_ground_surface(npy_path, seg_npy_path, boundary_npy_path, output_dir, ground_config)
        ground_npy_path = os.path.join(output_dir, "ground_surface.npy")
        ground_surface = np.load(ground_npy_path)

        # ==============================================================================
        # STEP 5: RELATIVE DSM (rDSM) & ELEVATION SURFACE
        # ==============================================================================
        write_step(5, "Elevation & rDSM Computation", 68, "Computing relative DSM and elevation surfaces...")

        # Relative DSM: Height above ground reference
        rdsm = np.maximum(0.0, ground_surface - raw_depth).astype(np.float32)
        rdsm[~valid_mask] = 0.0
        np.save(os.path.join(data_dir, "rdsm.npy"), rdsm)
        np.save(os.path.join(output_dir, "height_above_ground.npy"), rdsm)

        rdsm_valid = rdsm[valid_mask]
        min_elev = float(np.min(rdsm_valid)) if len(rdsm_valid) > 0 else 0.0
        max_elev = float(np.max(rdsm_valid)) if len(rdsm_valid) > 0 else 1.0
        mean_elev = float(np.mean(rdsm_valid)) if len(rdsm_valid) > 0 else 0.0
        median_elev = float(np.median(rdsm_valid)) if len(rdsm_valid) > 0 else 0.0
        p98_elev = float(np.percentile(rdsm_valid, 98.0)) if len(rdsm_valid) > 0 else 1.0

        # MAP 3: Relative DSM / rDSM (low-to-high elevation surface)
        rdsm_norm = np.clip(rdsm / (max(p98_elev, 0.05)), 0.0, 1.0)
        rdsm_u8 = (rdsm_norm * 255.0).astype(np.uint8)
        rdsm_bgr = cv2.applyColorMap(rdsm_u8, cv2.COLORMAP_VIRIDIS)
        rdsm_rgb = cv2.cvtColor(rdsm_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(rdsm_rgb).save(os.path.join(maps_dir, "rdsm.png"))

        # MAP 4: Elevation Heatmap (dedicated colorbar legend with min, max, mean, median, relative unit)
        elev_cbar_img = render_colorbar_map(
            rdsm,
            cmap_name='plasma',
            label_text='Relative Elevation (rDSM)',
            min_val=min_elev,
            max_val=max(p98_elev, 0.1),
            unit='relative'
        )
        elev_cbar_img.save(os.path.join(maps_dir, "elevation_heatmap.png"))

        # ==============================================================================
        # STEP 6: BUILDING DETECTION & FOOTPRINTS EXTRACTION
        # ==============================================================================
        write_step(6, "Building & Footprint Analysis", 80, "Detecting building polygons, estimating heights, and generating footprints...")

        bldg_mask = (seg_mask == 1) & valid_mask & (rdsm > 0.02)
        # Morphological close to unify roof planes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bldg_clean = cv2.morphologyEx(bldg_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(bldg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        buildings_list = []
        bldg_detection_img = img_np.copy()
        bldg_height_canvas = np.zeros((h, w), dtype=np.float32)
        bldg_boundary_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        # Subtle dark backdrop for boundary map
        bldg_boundary_canvas[:, :] = [18, 20, 26]

        min_area_thresh = max(30, int((w * h) * 0.00015))
        b_idx = 1

        for cnt in contours:
            area_px = int(cv2.contourArea(cnt))
            if area_px < min_area_thresh:
                continue

            c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(c_mask, [cnt], -1, 1, -1)
            inst_px = (c_mask == 1) & valid_mask

            if np.sum(inst_px) == 0:
                continue

            inst_h = rdsm[inst_px]
            b_height = float(np.median(inst_h))
            if b_height < 0.01:
                continue

            b_id = f"B{b_idx:03d}"
            bx, by, bw, bh = cv2.boundingRect(cnt)

            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx_b = float(M["m10"] / M["m00"])
                cy_b = float(M["m01"] / M["m00"])
            else:
                cx_b = float(bx + bw / 2.0)
                cy_b = float(by + bh / 2.0)

            conf = round(float(min(0.98, max(0.60, 0.65 + min(area_px / 1200.0, 0.3)))), 2)

            buildings_list.append({
                "id": b_id,
                "bounding_box": [int(bx), int(by), int(bx + bw), int(by + bh)],
                "center": [round(cx_b, 1), round(cy_b, 1)],
                "area_pixels": area_px,
                "estimated_height": round(b_height, 2),
                "height_unit": "relative",
                "confidence": conf
            })

            # Overlay on MAP 5 (Building Detection)
            # Tinted polygon overlay
            tint = np.array([0, 230, 255], dtype=np.uint8)
            overlay = bldg_detection_img.copy()
            cv2.drawContours(overlay, [cnt], -1, (0, 230, 255), -1)
            cv2.addWeighted(overlay, 0.35, bldg_detection_img, 0.65, 0, bldg_detection_img)
            cv2.rectangle(bldg_detection_img, (bx, by), (bx + bw, by + bh), (0, 255, 120), 2)
            cv2.putText(bldg_detection_img, f"{b_id}:{b_height:.1f}", (bx, max(15, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

            # Map 6 (Height Map values)
            bldg_height_canvas[inst_px] = b_height

            # Map 8 (Polygonal Footprints)
            epsilon = max(1.5, 0.015 * cv2.arcLength(cnt, True))
            approx_poly = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(bldg_boundary_canvas, [approx_poly], -1, (30, 100, 200), -1)
            cv2.drawContours(bldg_boundary_canvas, [approx_poly], -1, (0, 220, 255), 2)
            cv2.putText(bldg_boundary_canvas, b_id, (int(cx_b - 12), int(cy_b + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            b_idx += 1

        # MAP 5: Building Detection Map
        Image.fromarray(bldg_detection_img).save(os.path.join(maps_dir, "building_detection.png"))

        # MAP 6: Building Height Map
        max_bh = max(1.0, float(np.max(bldg_height_canvas)))
        bh_norm = np.clip(bldg_height_canvas / max_bh, 0.0, 1.0)
        bh_u8 = (bh_norm * 255.0).astype(np.uint8)
        bh_bgr = cv2.applyColorMap(bh_u8, cv2.COLORMAP_HOT)
        bh_rgb = cv2.cvtColor(bh_bgr, cv2.COLOR_BGR2RGB)
        # Darken non-building background
        bh_rgb[bldg_height_canvas <= 0] = [16, 18, 24]
        # Overlay height values
        for b in buildings_list:
            cx_i, cy_i = int(b["center"][0]), int(b["center"][1])
            cv2.putText(bh_rgb, f"{b['estimated_height']:.1f}", (cx_i - 10, cy_i), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        Image.fromarray(bh_rgb).save(os.path.join(maps_dir, "building_height_map.png"))

        # MAP 7: Edge / Boundary Map (Canny + Sobel + Morphology)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        canny_edges = cv2.Canny(gray, 50, 150)
        gx = cv2.Sobel(raw_depth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(raw_depth, cv2.CV_32F, 0, 1, ksize=3)
        g_depth = np.sqrt(gx**2 + gy**2)
        depth_edges = (g_depth > np.percentile(g_depth[np.isfinite(g_depth)], 85)).astype(np.uint8) * 255

        edges_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        edges_canvas[canny_edges > 0] = [255, 210, 0]      # Yellow: texture/RGB boundaries
        edges_canvas[depth_edges > 0] = [0, 240, 255]      # Cyan: depth discontinuity edges
        edges_canvas[(boundary > 0.4)] = [255, 255, 255]   # White: confirmed structural boundary
        Image.fromarray(edges_canvas).save(os.path.join(maps_dir, "edges.png"))

        # MAP 8: Building Boundary / Footprint Map
        Image.fromarray(bldg_boundary_canvas).save(os.path.join(maps_dir, "building_boundaries.png"))

        # ==============================================================================
        # STEP 7: SLOPE, TERRAIN RELIEF, SEMANTIC & ERROR/CONFIDENCE MAPS
        # ==============================================================================
        write_step(7, "Advanced Terrain & Thematic Maps", 90, "Computing slope gradient, shaded relief, semantic classes, and confidence...")

        # MAP 9: Slope Map (Relative slope estimation)
        dz_dx = cv2.Sobel(rdsm, cv2.CV_32F, 1, 0, ksize=3)
        dz_dy = cv2.Sobel(rdsm, cv2.CV_32F, 0, 1, ksize=3)
        slope_mag = np.sqrt(dz_dx**2 + dz_dy**2)
        slope_deg = np.degrees(np.arctan(slope_mag))

        slope_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        # Low slope (<10 deg): Green
        low_mask = slope_deg < 10.0
        # Moderate slope (10-25 deg): Yellow/Orange
        mod_mask = (slope_deg >= 10.0) & (slope_deg < 25.0)
        # High slope (>=25 deg): Red
        high_mask = slope_deg >= 25.0

        slope_canvas[low_mask] = [34, 197, 94]     # #22c55e Low
        slope_canvas[mod_mask] = [234, 179, 8]     # #eab308 Moderate
        slope_canvas[high_mask] = [239, 68, 68]    # #ef4444 High

        # Add legend watermark banner
        cv2.putText(slope_canvas, "Relative Slope: Low (<10 deg) | Mod (10-25 deg) | High (>25 deg)", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        Image.fromarray(slope_canvas).save(os.path.join(maps_dir, "slope_map.png"))

        # MAP 10: Terrain / Relief Map (Hillshade / shaded relief)
        hillshade_img = compute_hillshade(rdsm, azimuth_deg=315.0, altitude_deg=45.0, z_factor=2.0)
        # Blend hillshade with subtle copper elevation tint
        hillshade_bgr = cv2.cvtColor(hillshade_img, cv2.COLOR_GRAY2BGR)
        elev_tint = cv2.applyColorMap(rdsm_u8, cv2.COLORMAP_BONE)
        relief_blended = cv2.addWeighted(hillshade_bgr, 0.70, elev_tint, 0.30, 0)
        relief_rgb = cv2.cvtColor(relief_blended, cv2.COLOR_BGR2RGB)
        Image.fromarray(relief_rgb).save(os.path.join(maps_dir, "terrain_relief.png"))

        # MAP 11: Classification / Semantic Map
        # 1: Building, 2: Road, 3: Ground, 4: Vegetation
        semantic_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        semantic_canvas[seg_mask == 3] = [180, 140, 70]   # Ground / Terrain: Tan
        semantic_canvas[seg_mask == 2] = [249, 115, 22]   # Road / Pathway: Orange
        semantic_canvas[seg_mask == 4] = [16, 185, 129]   # Vegetation: Green
        semantic_canvas[seg_mask == 1] = [59, 130, 246]   # Building: Blue
        semantic_canvas[seg_mask == 0] = [100, 116, 139]  # Other / Unclassified: Slate

        # Embed clean legend swatches
        cv2.putText(semantic_canvas, "Blue: Building | Orange: Road | Green: Veg | Tan: Ground", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        Image.fromarray(semantic_canvas).save(os.path.join(maps_dir, "semantic_map.png"))

        # MAP 12: Error / Confidence Map (Model Confidence / Uncertainty)
        # Confidence derived from depth regularity and boundary certainty
        conf_raw = 1.0 - np.clip(boundary * 0.7 + (g_depth / (np.percentile(g_depth, 98) + 1e-6)) * 0.3, 0.0, 1.0)
        conf_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        # High confidence (conf >= 0.7): Green
        conf_canvas[conf_raw >= 0.7] = [16, 185, 129]
        # Medium confidence (0.4 <= conf < 0.7): Amber
        conf_canvas[(conf_raw >= 0.4) & (conf_raw < 0.7)] = [245, 158, 11]
        # Low confidence (conf < 0.4): Rose
        conf_canvas[conf_raw < 0.4] = [244, 63, 94]

        cv2.putText(conf_canvas, "Model Confidence: High (>70%) | Medium (40-70%) | Low (<40%)", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        Image.fromarray(conf_canvas).save(os.path.join(maps_dir, "error_map.png"))

        # Save standard TIFFs
        dsm_path = os.path.join(data_dir, "dsm.tif")
        depth_tiff_path = os.path.join(data_dir, "depth.tiff")
        if is_geotiff and crs_str is not None and HAS_RASTERIO:
            try:
                t = Affine(*geo_transform[:6]) if geo_transform and len(geo_transform) >= 6 else None
                with rasterio.open(
                    depth_tiff_path, 'w', driver='GTiff',
                    height=h, width=w, count=1, dtype='float32',
                    crs=crs_str, transform=t
                ) as dst:
                    dst.write(raw_depth.astype(np.float32), 1)

                with rasterio.open(
                    dsm_path, 'w', driver='GTiff',
                    height=h, width=w, count=1, dtype='float32',
                    crs=crs_str, transform=t
                ) as dst:
                    dst.write(rdsm.astype(np.float32), 1)
            except Exception as geo_e:
                print(f"[GEOSPATIAL_NOTE] {geo_e}. Saving standard TIFF.")
                tifffile.imwrite(depth_tiff_path, raw_depth.astype(np.float32))
                tifffile.imwrite(dsm_path, rdsm.astype(np.float32))
        else:
            tifffile.imwrite(depth_tiff_path, raw_depth.astype(np.float32))
            tifffile.imwrite(dsm_path, rdsm.astype(np.float32))

        # ==============================================================================
        # STEP 8: COMPILE JSON REPORTS (METADATA, BUILDINGS, STATISTICS)
        # ==============================================================================
        write_step(8, "Exporting Analysis Data & Reports", 98, "Writing metadata.json, buildings.json, and statistics.json...")

        elapsed_sec = round(time.time() - start_time, 2)
        total_bldg_count = len(buildings_list)
        b_heights = [b["estimated_height"] for b in buildings_list] if total_bldg_count > 0 else [0.0]
        b_areas = [b["area_pixels"] for b in buildings_list] if total_bldg_count > 0 else [0]
        b_confs = [b["confidence"] for b in buildings_list] if total_bldg_count > 0 else [0.0]

        avg_b_h = round(float(np.mean(b_heights)), 2)
        max_b_h = round(float(np.max(b_heights)), 2)
        min_b_h = round(float(np.min(b_heights)), 2)
        total_b_area = int(np.sum(b_areas))
        avg_b_conf = round(float(np.mean(b_confs)), 2)
        avg_model_conf = round(float(np.mean(conf_raw)), 3)

        # 1. METADATA.JSON (Strictly per Section 7 specification)
        metadata_json_content = {
            "image": {
                "filename": os.path.basename(input_path),
                "width": int(w),
                "height": int(h),
                "channels": int(img_np.shape[2]),
                "format": input_format,
                "file_size_bytes": file_size_bytes,
                "processing_time_seconds": elapsed_sec
            },
            "geospatial": {
                "georeferenced": is_geotiff,
                "crs": crs_str if is_geotiff else "Georeferencing: Not Available",
                "epsg": epsg_code,
                "pixel_size": pixel_size,
                "bounds": geo_bounds,
                "transform": geo_transform,
                "coordinates": geo_coords
            },
            "depth": {
                "min": round(min_depth, 3),
                "max": round(max_depth, 3),
                "mean": round(mean_depth, 3),
                "median": round(median_depth, 3),
                "std": round(std_depth, 3),
                "unit": "relative"
            },
            "elevation": {
                "type": "relative",
                "min": round(min_elev, 3),
                "max": round(max_elev, 3),
                "mean": round(mean_elev, 3),
                "median": round(median_elev, 3),
                "unit": "relative"
            },
            "buildings": {
                "count": total_bldg_count,
                "average_height": avg_b_h,
                "maximum_height": max_b_h,
                "minimum_height": min_b_h,
                "total_area": total_b_area,
                "average_confidence": avg_b_conf
            },
            "quality": {
                "confidence": avg_model_conf,
                "mae": None,
                "rmse": None,
                "correlation": None,
                "quality_status": "PASS" if valid_pixels_count > 0 else "WARNING"
            },
            "outputs": {
                "rgb": "maps/rgb.png",
                "depth": "maps/depth_map.png",
                "rdsm": "maps/rdsm.png",
                "elevation_heatmap": "maps/elevation_heatmap.png",
                "building_detection": "maps/building_detection.png",
                "building_height": "maps/building_height_map.png",
                "edges": "maps/edges.png",
                "building_boundaries": "maps/building_boundaries.png",
                "slope": "maps/slope_map.png",
                "terrain_relief": "maps/terrain_relief.png",
                "semantic": "maps/semantic_map.png",
                "error": "maps/error_map.png"
            }
        }

        # Write to data/metadata.json and root metadata.json
        with open(os.path.join(data_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata_json_content, f, indent=4)
        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata_json_content, f, indent=4)

        # 2. BUILDINGS.JSON (Strictly per Section 9 specification)
        buildings_json_content = {
            "session_id": generation_id,
            "summary": {
                "count": total_bldg_count,
                "average_height": avg_b_h,
                "maximum_height": max_b_h,
                "minimum_height": min_b_h,
                "total_area": total_b_area,
                "average_confidence": avg_b_conf
            },
            "buildings": buildings_list
        }
        with open(os.path.join(data_dir, "buildings.json"), "w", encoding="utf-8") as f:
            json.dump(buildings_json_content, f, indent=4)
        with open(os.path.join(output_dir, "buildings.json"), "w", encoding="utf-8") as f:
            json.dump(buildings_json_content, f, indent=4)

        # 3. STATISTICS.JSON (Detailed numerical breakdown)
        statistics_json_content = {
            "session_id": generation_id,
            "elapsed_seconds": elapsed_sec,
            "depth_statistics": {
                "shape": [int(h), int(w)],
                "dtype": "float32",
                "min": round(min_depth, 3),
                "max": round(max_depth, 3),
                "mean": round(mean_depth, 3),
                "median": round(median_depth, 3),
                "std": round(std_depth, 3),
                "p01": round(p01, 3),
                "p02": round(p02, 3),
                "p05": round(p05, 3),
                "p95": round(p95, 3),
                "p98": round(p98, 3),
                "p99": round(p99, 3),
                "valid_pixels": valid_pixels_count,
                "nan_pixels": nan_count,
                "infinite_pixels": inf_count
            },
            "elevation_statistics": {
                "type": "relative_surface_above_ground",
                "min": round(min_elev, 3),
                "max": round(max_elev, 3),
                "mean": round(mean_elev, 3),
                "median": round(median_elev, 3),
                "p98": round(p98_elev, 3),
                "unit": "relative"
            },
            "building_statistics": {
                "total_detected": total_bldg_count,
                "total_area_px": total_b_area,
                "average_area_px": round(float(np.mean(b_areas)), 1) if total_bldg_count > 0 else 0,
                "average_height": avg_b_h,
                "max_height": max_b_h,
                "min_height": min_b_h,
                "average_confidence": avg_b_conf
            },
            "geospatial": metadata_json_content["geospatial"]
        }
        with open(os.path.join(data_dir, "statistics.json"), "w", encoding="utf-8") as f:
            json.dump(statistics_json_content, f, indent=4)
        with open(os.path.join(output_dir, "statistics.json"), "w", encoding="utf-8") as f:
            json.dump(statistics_json_content, f, indent=4)

        # Backwards-compatible analysis_report.json
        analysis_report_content = {
            "project": {"name": "Terracraft Remote Sensing", "version": "2.0.0"},
            "session": {
                "session_id": generation_id,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "status": "completed",
                "processing_time_seconds": elapsed_sec
            },
            "metadata": metadata_json_content,
            "buildings": buildings_json_content,
            "statistics": statistics_json_content,
            "outputs": metadata_json_content["outputs"]
        }
        with open(os.path.join(output_dir, "analysis_report.json"), "w", encoding="utf-8") as f:
            json.dump(analysis_report_content, f, indent=4)

        # Final progress signal
        if progress_file:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "status": "completed",
                    "step_index": 8,
                    "stage": "Complete",
                    "step_name": "Ready",
                    "message": "12-map elevation & remote-sensing analysis completed successfully.",
                    "progress": 100,
                    "session_id": generation_id,
                    "generation_id": generation_id,
                    "timestamp": time.time()
                }, f, indent=2)

        print(f"\n[SUCCESS] 2D 12-Map Remote Sensing pipeline completed in {elapsed_sec}s. Outputs in: {output_dir}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        write_error(0, "Pipeline Execution", str(e), cause=traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
