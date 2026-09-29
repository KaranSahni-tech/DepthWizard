import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import depth_pro

# ==============================================================================
# EXCEPTIONS
# ==============================================================================
class PipelineValidationError(Exception):
    def __init__(self, check_name, expected, actual, array_source, location=""):
        self.check_name = check_name
        self.expected = expected
        self.actual = actual
        self.array_source = array_source
        self.location = location
        super().__init__(f"Validation Failed [{check_name}]: expected {expected}, got {actual}")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TILE_SIZE = 768
OVERLAP = 128
LOWER_PERCENTILE = 1.0
UPPER_PERCENTILE = 99.0

# ==============================================================================
# VALIDATION ENGINE
# ==============================================================================
def validate_pipeline(
    img_h, img_w, total_pixels,
    tiling_enabled, alignment_pairs,
    raw_tile_depth, final_raw_depth, clean_depth, relative_surface,
    outlier_mask, masked_pixels, masked_percentage,
    raw_stats, clean_stats, relative_stats,
    tile_stats=None
):
    """
    The ultimate semantic and mathematical validation gate.
    MUST pass silently to return True, otherwise raises PipelineValidationError.
    """
    
    # [x] image shape & depth shape
    if final_raw_depth.shape != (img_h, img_w):
        raise PipelineValidationError("depth shape", (img_h, img_w), final_raw_depth.shape, "final_raw_depth")
    if clean_depth.shape != (img_h, img_w):
        raise PipelineValidationError("clean depth shape", (img_h, img_w), clean_depth.shape, "clean_depth")
    if relative_surface.shape != (img_h, img_w):
        raise PipelineValidationError("relative surface shape", (img_h, img_w), relative_surface.shape, "relative_surface")
        
    # [x] total pixels
    if final_raw_depth.size != total_pixels:
        raise PipelineValidationError("total pixels", total_pixels, final_raw_depth.size, "final_raw_depth")

    # [x] finite checks
    if np.sum(np.isinf(final_raw_depth)) > 0:
        raise PipelineValidationError("finite raw depth", 0, np.sum(np.isinf(final_raw_depth)), "final_raw_depth")
    if np.sum(np.isinf(clean_depth[~np.isnan(clean_depth)])) > 0:
        raise PipelineValidationError("finite clean depth", 0, "Infs found", "clean_depth")
    if np.sum(np.isinf(relative_surface[~np.isnan(relative_surface)])) > 0:
        raise PipelineValidationError("finite relative surface", 0, "Infs found", "relative_surface")

    # [x] outlier mask consistency
    if outlier_mask.shape != (img_h, img_w):
        raise PipelineValidationError("outlier mask shape", (img_h, img_w), outlier_mask.shape, "outlier_mask")
    actual_mask_count = np.count_nonzero(outlier_mask)
    if masked_pixels != actual_mask_count:
        raise PipelineValidationError("masked pixel count", masked_pixels, actual_mask_count, "outlier_mask")
    if abs(masked_percentage - (masked_pixels/total_pixels)*100) > 1e-6:
        raise PipelineValidationError("masked percentage", (masked_pixels/total_pixels)*100, masked_percentage, "outlier_mask")

    # [x] relative surface range
    rel_min = np.nanmin(relative_surface)
    if rel_min < -1e-5:
        raise PipelineValidationError("relative surface min >= 0", ">=0", rel_min, "relative_surface")
    
    max_clean_depth = np.nanmax(clean_depth)
    min_clean_depth = np.nanmin(clean_depth)
    expected_max_rel = max_clean_depth - min_clean_depth
    actual_max_rel = np.nanmax(relative_surface)
    if not np.isclose(actual_max_rel, expected_max_rel, rtol=1e-5, atol=1e-5):
        raise PipelineValidationError("relative surface max", expected_max_rel, actual_max_rel, "relative_surface")

    # [x] alignment logic
    if not tiling_enabled:
        if len(alignment_pairs) > 0:
            raise PipelineValidationError("single-tile alignment", 0, len(alignment_pairs), "alignment_pairs")
    else:
        for pair in alignment_pairs:
            if pair["tile_a"] == pair["tile_b"]:
                raise PipelineValidationError("no self-alignment", "tile_a != tile_b", "tile_a == tile_b", "alignment_pairs")

    # [x] tile/final raw depth consistency
    if not tiling_enabled:
        if not np.allclose(raw_tile_depth, final_raw_depth, rtol=1e-5, atol=1e-6):
            raise PipelineValidationError("tile raw depth == final raw depth", "Match", "Mismatch", "final_raw_depth")
        for field in tile_stats:
            if not np.isclose(tile_stats[field], raw_stats[field], rtol=1e-5, atol=1e-6):
                raise PipelineValidationError(f"tile_stats[{field}] == final_stats[{field}]", tile_stats[field], raw_stats[field], "raw_stats")

    return True

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def get_tiles(img_w, img_h, tile_size, overlap):
    if img_w <= tile_size and img_h <= tile_size:
        return [(0, 0, img_w, img_h)]
        
    stride = tile_size - overlap
    tiles = []
    
    y = 0
    while y < img_h:
        y_start = y
        y_end = min(y + tile_size, img_h)
        if y_end - y_start < tile_size and img_h > tile_size:
            y_start = max(0, img_h - tile_size)
            y_end = img_h
            
        x = 0
        while x < img_w:
            x_start = x
            x_end = min(x + tile_size, img_w)
            if x_end - x_start < tile_size and img_w > tile_size:
                x_start = max(0, img_w - tile_size)
                x_end = img_w
                
            tiles.append((x_start, y_start, x_end, y_end))
            if x_end == img_w:
                break
            x += stride
            
        if y_end == img_h:
            break
        y += stride
        
    return list(set(tiles))

def mad_filter(data, thresh=3.5):
    median = np.median(data)
    diff = np.abs(data - median)
    med_abs_deviation = np.median(diff)
    if med_abs_deviation == 0:
        med_abs_deviation = 1e-6
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score < thresh

def robust_align_least_squares(source, target):
    valid = np.isfinite(source) & np.isfinite(target)
    s_valid = source[valid]
    t_valid = target[valid]
    
    if len(s_valid) < 10:
        return 1.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0
        
    diff = t_valid - s_valid
    inlier_mask = mad_filter(diff, thresh=3.0)
    
    s_inliers = s_valid[inlier_mask]
    t_inliers = t_valid[inlier_mask]
    
    if len(s_inliers) < 10:
        s_inliers, t_inliers = s_valid, t_valid 
        inlier_mask = np.ones(len(s_valid), dtype=bool)

    A = np.vstack([s_inliers, np.ones(len(s_inliers))]).T
    res = np.linalg.lstsq(A, t_inliers, rcond=None)[0]
    
    scale = np.clip(res[0], 0.2, 5.0)
    shift = res[1]
    
    aligned_s = (s_valid * scale) + shift
    residuals = np.abs(t_valid - aligned_s)
    
    inlier_percentage = float(np.sum(inlier_mask) / len(s_valid)) * 100.0
    residual_median = float(np.median(residuals))
    residual_mean = float(np.mean(residuals))
    residual_p95 = float(np.percentile(residuals, 95))
    
    return float(scale), float(shift), len(s_valid), inlier_percentage, residual_median, residual_mean, residual_p95

def create_2d_window(h, w):
    wy = np.hanning(h + 2)[1:-1]
    wx = np.hanning(w + 2)[1:-1]
    return np.outer(wy, wx)

def calculate_statistics(array):
    valid = array[np.isfinite(array)]
    if len(valid) == 0:
        return {}
    mean_val = float(np.mean(valid))
    std_val = float(np.std(valid))
    return {
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": mean_val,
        "median": float(np.median(valid)),
        "std": std_val,
        "coefficient_of_variation": float(std_val / mean_val) if mean_val != 0 else 0,
        "p01": float(np.percentile(valid, 1)),
        "p05": float(np.percentile(valid, 5)),
        "p25": float(np.percentile(valid, 25)),
        "p50": float(np.median(valid)),
        "p75": float(np.percentile(valid, 75)),
        "p95": float(np.percentile(valid, 95)),
        "p99": float(np.percentile(valid, 99))
    }

def normalize_for_heatmap(array, p_low, p_high):
    arr_valid = array[np.isfinite(array)]
    if len(arr_valid) == 0:
        return array
    min_val = np.percentile(arr_valid, p_low)
    max_val = np.percentile(arr_valid, p_high)
    norm = np.clip(array, min_val, max_val)
    return (norm - min_val) / (max_val - min_val + 1e-8)

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/simage.jpg"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output/simage_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print(f" PIPELINE START: {input_path}")
    print("="*50)
    
    # ---------------------------------------------------------
    # 1. INPUT IMAGE
    # ---------------------------------------------------------
    original_img_pil = Image.open(input_path)
    img_w, img_h = original_img_pil.size
    img_format = str(original_img_pil.format)
    img_channels = len(original_img_pil.getbands())
    original_img_pil = original_img_pil.convert("RGB")
    
    total_pixels = img_w * img_h
    original_img_pil.save(os.path.join(output_dir, "01_original.png"))

    # ---------------------------------------------------------
    # 2. MODEL INFERENCE
    # ---------------------------------------------------------
    model, transform = depth_pro.create_model_and_transforms(precision=torch.half)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    image_dp, _, f_px_global_estimated = depth_pro.load_rgb(input_path)
    
    tiles = get_tiles(img_w, img_h, TILE_SIZE, OVERLAP)
    tiling_enabled = len(tiles) > 1
    
    alignment_pairs = []
    
    tile_stats = None

    if not tiling_enabled:
        image_tensor = transform(image_dp).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = model.infer(image_tensor, f_px=f_px_global_estimated)
            raw_model_depth = pred["depth"].cpu().numpy()
            used_f_px = pred["focallength_px"]
            
        raw_model_depth = raw_model_depth[:img_h, :img_w]
        
        raw_tile_depth = raw_model_depth.copy()
        final_raw_depth = raw_tile_depth.copy()
        tile_stats = calculate_statistics(raw_tile_depth)

    else:
        raw_tile_depth = None
        image_tensor = transform(image_dp).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            base_pred = model.infer(image_tensor, f_px=f_px_global_estimated)
            raw_model_depth = base_pred["depth"].cpu().numpy()
            raw_model_depth = raw_model_depth[:img_h, :img_w]
            used_f_px = base_pred["focallength_px"]
            
        fused_raw_depth = np.zeros((img_h, img_w), dtype=np.float32)
        fused_weights = np.zeros((img_h, img_w), dtype=np.float32)

        for idx, (x1, y1, x2, y2) in enumerate(tiles):
            tile_img = image_dp[y1:y2, x1:x2, :]
            tile_tensor = transform(tile_img).to(device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                tile_pred = model.infer(tile_tensor, f_px=None)
                current_raw_tile_depth = tile_pred["depth"].cpu().numpy()
                
            base_crop = raw_model_depth[y1:y2, x1:x2]
            
            tile_a_id = idx
            tile_b_id = "base_structure"
            
            scale, shift, n_ov, inlier_pct, r_med, r_mean, r_p95 = robust_align_least_squares(current_raw_tile_depth, base_crop)
            tile_depth_aligned = (current_raw_tile_depth * scale) + shift
            
            alignment_pairs.append({
                "tile_a": tile_a_id,
                "tile_b": tile_b_id,
                "overlap_pixels": n_ov,
                "scale": scale,
                "shift": shift,
                "residual_median": r_med,
                "residual_mean": r_mean,
                "residual_p95": r_p95,
                "inlier_percentage": inlier_pct
            })
            
            window = create_2d_window(tile_depth_aligned.shape[0], tile_depth_aligned.shape[1])
            fused_raw_depth[y1:y2, x1:x2] += (tile_depth_aligned * window)
            fused_weights[y1:y2, x1:x2] += window
            
        final_raw_depth = fused_raw_depth / (fused_weights + 1e-8)

    # ---------------------------------------------------------
    # 3. STATISTICAL OUTLIER FILTERING
    # ---------------------------------------------------------
    np.save(os.path.join(output_dir, "02_raw_depth.npy"), final_raw_depth)
    final_raw_stats = calculate_statistics(final_raw_depth)
    
    n_inf = np.sum(np.isinf(final_raw_depth))
    n_nan = np.sum(np.isnan(final_raw_depth))
    if n_inf > 0 or n_nan > 0:
        print(f"Warning: Raw depth contains {n_nan} NaNs and {n_inf} Infs.")
        
    lower_bound = final_raw_stats[f"p{int(LOWER_PERCENTILE):02d}"]
    upper_bound = final_raw_stats[f"p{int(UPPER_PERCENTILE):02d}"]
    
    outlier_mask = (final_raw_depth < lower_bound) | (final_raw_depth > upper_bound)
    masked_pixels = int(np.sum(outlier_mask))
    masked_percentage = (masked_pixels / total_pixels) * 100.0
    
    p05_threshold = final_raw_stats["p05"]
    p95_threshold = final_raw_stats["p95"]
    pixels_below_p05 = int(np.sum(final_raw_depth < p05_threshold))
    pixels_above_p95 = int(np.sum(final_raw_depth > p95_threshold))
    
    Image.fromarray((outlier_mask * 255).astype(np.uint8), mode='L').save(os.path.join(output_dir, "04_outlier_mask.png"))
    
    # ---------------------------------------------------------
    # 4. CLEAN DEPTH
    # ---------------------------------------------------------
    clean_depth = final_raw_depth.copy()
    clean_depth[outlier_mask] = np.nan
    clean_stats = calculate_statistics(clean_depth)
    
    np.save(os.path.join(output_dir, "06_clean_depth.npy"), clean_depth)

    # ---------------------------------------------------------
    # 5. RELATIVE SURFACE
    # ---------------------------------------------------------
    max_clean = float(np.nanmax(clean_depth))
    relative_surface = max_clean - clean_depth
    relative_stats = calculate_statistics(relative_surface)
    
    np.save(os.path.join(output_dir, "08_relative_surface.npy"), relative_surface)

    # ---------------------------------------------------------
    # 6. MANDATORY VALIDATION GATE
    # ---------------------------------------------------------
    try:
        validate_pipeline(
            img_h=img_h, img_w=img_w, total_pixels=total_pixels,
            tiling_enabled=tiling_enabled, alignment_pairs=alignment_pairs,
            raw_tile_depth=raw_tile_depth, final_raw_depth=final_raw_depth, 
            clean_depth=clean_depth, relative_surface=relative_surface,
            outlier_mask=outlier_mask, masked_pixels=masked_pixels, 
            masked_percentage=masked_percentage,
            raw_stats=final_raw_stats, clean_stats=clean_stats, relative_stats=relative_stats,
            tile_stats=tile_stats
        )
    except PipelineValidationError as e:
        print("\n" + "!"*50)
        print(" VALIDATION FAILED - PIPELINE HALTED")
        print("!"*50)
        print(f"Check Failed: {e.check_name}")
        print(f"Expected: {e.expected}")
        print(f"Actual: {e.actual}")
        print(f"Array Source: {e.array_source}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 7. FINAL JSON GENERATION
    # ---------------------------------------------------------
    report = {
        "input": {
            "file": os.path.basename(input_path),
            "width": img_w,
            "height": img_h,
            "total_pixels": total_pixels,
            "channels": img_channels,
            "format": str(img_format)
        },
        "model": {
            "name": "Apple Depth Pro",
            "output_type": "unvalidated_monocular_depth",
            "estimated_focal_length_pixels": float(used_f_px) if used_f_px is not None else None,
            "focal_length_source": "model_estimate",
            "focal_length_physically_validated": False,
            "depth_units": "model_reported_depth",
            "depth_scale_validated": False,
            "metric_interpretation": "unvalidated"
        },
        "tiling": {
            "enabled": tiling_enabled,
            "number_of_tiles": len(tiles)
        },
        "tile_alignment": {
            "status": "performed" if tiling_enabled else "not_applicable",
            "reason": None if tiling_enabled else "Only one tile exists; inter-tile alignment is not required.",
            "pairs": alignment_pairs
        },
        "raw_depth": {
            "source_array": "final_raw_depth",
            "shape": list(final_raw_depth.shape),
            "statistics": final_raw_stats
        },
        "outliers": {
            "method": "percentile",
            "lower_percentile": LOWER_PERCENTILE,
            "upper_percentile": UPPER_PERCENTILE,
            "lower_threshold": float(lower_bound),
            "upper_threshold": float(upper_bound),
            "masked_pixels": masked_pixels,
            "masked_percentage": masked_percentage,
            "pixels_below_p05": pixels_below_p05,
            "pixels_above_p95": pixels_above_p95
        },
        "clean_depth": {
            "source_array": "clean_depth",
            "statistics": clean_stats
        },
        "relative_surface": {
            "available": True,
            "source_array": "clean_depth",
            "type": "depth_derived_relative_surface_proxy",
            "transformation": "relative_height = max_clean_depth - clean_depth",
            "direction": {
                "model_depth": "larger_value_means_farther_from_camera",
                "surface_proxy": "larger_value_means_higher_relative_surface_proxy"
            },
            "geometric_interpretation": "relative_surface_proxy",
            "normalization": "none",
            "representation": "depth_difference_from_maximum",
            "units": "relative_model_depth_units",
            "metric_scale": False,
            "absolute_elevation": False,
            "ground_truth_validated": False,
            "physical_elevation_equivalence_validated": False,
            "statistics": relative_stats
        },
        "metric_calibration": {
            "available": False,
            "method": None,
            "scale_factor": None,
            "reference_source": None,
            "validated": False
        },
        "absolute_elevation": {
            "available": False,
            "validated": False,
            "unit": None
        },
        "geospatial": {
            "crs": None,
            "georeferenced": False
        },
        "pipeline_validation": {
            "status": "passed",
            "validation_scope": "internal_pipeline_consistency"
        },
        "scientific_validation": {
            "metric_depth_validated": False,
            "absolute_elevation_validated": False,
            "ground_truth_available": False,
            "validation_method": None
        }
    }
    
    with open(os.path.join(output_dir, "11_quality_report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    print("\n==================================================")
    print(" PIPELINE FINISHED SUCCESSFULLY ")
    print("==================================================")

if __name__ == "__main__":
    main()
