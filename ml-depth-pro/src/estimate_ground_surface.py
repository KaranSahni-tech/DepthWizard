import os
import sys
import json
import argparse
import numpy as np
import cv2
from memory_utils import to_float32_depth, save_colormap_safe, log_memory_usage, clean_memory

CLASS_UNKNOWN = 0
CLASS_BUILDING = 1
CLASS_ROAD = 2
CLASS_GROUND = 3
CLASS_VEGETATION = 4

def estimate_ground_surface(depth_path, seg_path, boundary_path, output_dir, config):
    print(f"[GROUND_ESTIMATION] Loading depth, segmentation, and boundary inputs...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Inputs (Enforce float32 / uint8)
    depth = to_float32_depth(np.load(depth_path))
    seg_mask = np.load(seg_path).astype(np.uint8, copy=False)
    boundary = np.load(boundary_path).astype(np.float32, copy=False)

    h, w = depth.shape
    valid_mask = np.isfinite(depth) & (depth > 0)

    # 2. Identify Ground Seed Points
    is_road_or_ground = (seg_mask == CLASS_ROAD) | (seg_mask == CLASS_GROUND)
    is_building = (seg_mask == CLASS_BUILDING)
    is_veg = (seg_mask == CLASS_VEGETATION)

    kernel_veg = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    depth_dilated_veg = cv2.dilate(depth, kernel_veg)
    veg_ground_seed = is_veg & (np.abs(depth - depth_dilated_veg) < 0.15)
    del depth_dilated_veg
    clean_memory()

    ground_seed_mask = (is_road_or_ground | veg_ground_seed) & valid_mask & (boundary < config['max_boundary_seed_thresh'])
    del is_road_or_ground, veg_ground_seed
    clean_memory()

    print(f"[GROUND_ESTIMATION] Ground seeds: {np.sum(ground_seed_mask)} pixels ({np.sum(ground_seed_mask)/(h*w)*100:.2f}% of total image)")

    # 3. Normalized Convolution / Smooth Inpainting (float32)
    z_seed = np.zeros((h, w), dtype=np.float32)
    z_seed[ground_seed_mask] = depth[ground_seed_mask]
    w_seed = ground_seed_mask.astype(np.float32)

    kernel_size = max(21, int(min(w, h) * 0.08) | 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = kernel_size / 2.5

    z_blur = cv2.GaussianBlur(z_seed, (kernel_size, kernel_size), sigma)
    w_blur = cv2.GaussianBlur(w_seed, (kernel_size, kernel_size), sigma)
    del z_seed, w_seed
    clean_memory()
    
    med_val = float(np.nanmedian(depth[valid_mask])) if np.any(valid_mask) else 2.5
    ground_surface = np.where(w_blur > 1e-5, z_blur / (w_blur + 1e-6), med_val).astype(np.float32)
    del z_blur, w_blur
    clean_memory()

    large_kernel = kernel_size * 2 + 1
    large_sigma = sigma * 1.8
    ground_surface = cv2.GaussianBlur(ground_surface, (large_kernel, large_kernel), large_sigma).astype(np.float32)
    ground_surface[~valid_mask] = med_val

    log_memory_usage("Estimate Ground", {
        "depth": depth,
        "ground_surface": ground_surface
    })

    # Save output array as float32
    np.save(os.path.join(output_dir, "ground_surface.npy"), ground_surface)

    # 5. Save Visualizations (Memory Safe)
    p05_g = float(np.percentile(ground_surface, 5))
    p95_g = float(np.percentile(ground_surface, 95))
    save_colormap_safe(ground_surface, os.path.join(output_dir, "ground_surface.png"), cmap="viridis", vmin=p05_g, vmax=p95_g)

    # 6. Metadata JSON
    stats = {
        "config": config,
        "image_dimensions": {"width": w, "height": h},
        "ground_seed_pixel_count": int(np.sum(ground_seed_mask)),
        "ground_seed_percentage": float(round((np.sum(ground_seed_mask)/(h*w))*100, 2)),
        "min_ground_depth_m": float(np.min(ground_surface)),
        "max_ground_depth_m": float(np.max(ground_surface)),
        "mean_ground_depth_m": float(np.mean(ground_surface)),
        "median_ground_depth_m": float(np.median(ground_surface)),
        "std_ground_depth_m": float(np.std(ground_surface))
    }

    with open(os.path.join(output_dir, "ground_metadata.json"), "w") as f:
        json.dump(stats, f, indent=4)

    del depth, seg_mask, boundary
    if 'ground_mask_seed' in locals():
        del ground_mask_seed
    clean_memory()
    print(f"[GROUND_ESTIMATION] Finished successfully. All outputs written to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ground / Base Surface Reconstruction Stage")
    parser.add_argument("--depth", required=True, help="Path to cleaned depth.npy")
    parser.add_argument("--segmentation", required=True, help="Path to segmentation_mask.npy")
    parser.add_argument("--boundary", required=True, help="Path to boundary_confidence.npy")
    parser.add_argument("--output", required=True, help="Output directory path")

    parser.add_argument("--smooth_kernel_size", type=int, default=51, help="Gaussian smoothing kernel size for ground fitting")
    parser.add_argument("--smooth_sigma", type=float, default=20.0, help="Gaussian sigma for ground spatial continuity")
    parser.add_argument("--max_boundary_seed_thresh", type=float, default=0.4, help="Max boundary confidence allowed for ground seeds")

    args = parser.parse_args()

    config = {
        "smooth_kernel_size": args.smooth_kernel_size,
        "smooth_sigma": args.smooth_sigma,
        "max_boundary_seed_thresh": args.max_boundary_seed_thresh
    }

    estimate_ground_surface(args.depth, args.segmentation, args.boundary, args.output, config)
