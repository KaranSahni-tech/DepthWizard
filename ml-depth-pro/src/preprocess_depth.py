import os
import sys
import json
import numpy as np
import cv2
from PIL import Image
from memory_utils import to_float32_depth, save_colormap_safe, clean_memory, log_memory_usage

def preprocess_depth(image_path, depth_path, output_dir):
    print(f"[PREPROCESS] Memory-safe preprocessing for {depth_path}...")
    os.makedirs(output_dir, exist_ok=True)
    
    depth = to_float32_depth(np.load(depth_path))
    h, w = depth.shape
    
    valid_mask = np.isfinite(depth) & (depth > 0)
    
    nan_count = int(np.isnan(depth).sum())
    inf_count = int(np.isinf(depth).sum())
    zero_or_neg_count = int(((depth <= 0) & np.isfinite(depth)).sum())
    
    valid_pixels = depth[valid_mask]
    valid_count = len(valid_pixels)
    total_count = w * h
    valid_percentage = float((valid_count / total_count) * 100)
    
    min_depth = float(np.min(valid_pixels)) if valid_count > 0 else 0.0
    max_depth = float(np.max(valid_pixels)) if valid_count > 0 else 1.0
    mean_depth = float(np.mean(valid_pixels)) if valid_count > 0 else 0.5
    median_depth = float(np.median(valid_pixels)) if valid_count > 0 else 0.5
    std_depth = float(np.std(valid_pixels)) if valid_count > 0 else 0.1
    
    p02 = float(np.percentile(valid_pixels, 2)) if valid_count > 0 else min_depth
    p05 = float(np.percentile(valid_pixels, 5)) if valid_count > 0 else min_depth
    p95 = float(np.percentile(valid_pixels, 95)) if valid_count > 0 else max_depth
    p98 = float(np.percentile(valid_pixels, 98)) if valid_count > 0 else max_depth
    
    outlier_min = max(0.0, median_depth - 3.0 * std_depth)
    outlier_max = median_depth + 3.0 * std_depth
    
    outlier_mask = (depth < outlier_min) | (depth > outlier_max)
    final_valid_mask = valid_mask & ~outlier_mask
    del outlier_mask
    clean_memory()
    
    cleaned_depth = np.copy(depth)
    cleaned_depth[~final_valid_mask] = median_depth
    
    print("[PREPROCESS] Applying Bilateral Filter...")
    smoothed_depth = cv2.bilateralFilter(cleaned_depth, d=9, sigmaColor=0.1, sigmaSpace=7).astype(np.float32)
    smoothed_depth[~final_valid_mask] = np.nan
    del cleaned_depth
    clean_memory()
    
    ground_reference = p95
    relative_depth = np.maximum(0.0, ground_reference - smoothed_depth).astype(np.float32)
    
    log_memory_usage("Preprocess Depth", {
        "depth": depth,
        "smoothed_depth": smoothed_depth,
        "relative_depth": relative_depth
    })

    print("[PREPROCESS] Generating visualizations...")
    save_colormap_safe(depth, os.path.join(output_dir, "depth_raw.png"), cmap="turbo", vmin=p05, vmax=p95)
    save_colormap_safe(smoothed_depth, os.path.join(output_dir, "depth_cleaned.png"), cmap="turbo", vmin=p05, vmax=p95)
    save_colormap_safe(relative_depth, os.path.join(output_dir, "depth_relative.png"), cmap="magma", vmin=0, vmax=max(0.1, ground_reference-p05))
    
    Image.fromarray((final_valid_mask * 255).astype(np.uint8)).save(os.path.join(output_dir, "depth_validity_mask.png"))
    np.save(os.path.join(output_dir, "cleaned_depth.npy"), smoothed_depth)
    
    stats = {
        "width": int(w),
        "height": int(h),
        "dtype": str(depth.dtype),
        "valid_percentage": float(valid_percentage),
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "zero_or_neg_count": int(zero_or_neg_count),
        "min": float(min_depth),
        "max": float(max_depth),
        "mean": float(mean_depth),
        "median": float(median_depth),
        "std_dev": float(std_depth),
        "p02": float(p02),
        "p05": float(p05),
        "p95": float(p95),
        "p98": float(p98),
        "outlier_min_threshold": float(outlier_min),
        "outlier_max_threshold": float(outlier_max)
    }
    
    with open(os.path.join(output_dir, "depth_statistics.json"), "w") as f:
        json.dump(stats, f, indent=4)
        
    del depth, smoothed_depth, relative_depth
    clean_memory()
    print(f"[PREPROCESS] Finished. Results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input RGB image")
    parser.add_argument("--depth", required=True, help="Path to input depth.npy")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    
    preprocess_depth(args.image, args.depth, args.output)
