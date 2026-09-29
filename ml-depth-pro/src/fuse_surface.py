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

def fuse_surface(depth_path, ground_path, seg_path, boundary_path, bldg_h_path, bldg_conf_path, output_dir, config):
    print(f"[SURFACE_FUSION] Memory-safe surface fusion...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Inputs (Enforce float32 / uint8)
    depth = to_float32_depth(np.load(depth_path))
    ground = to_float32_depth(np.load(ground_path))
    seg_mask = np.load(seg_path).astype(np.uint8, copy=False)
    boundary = np.load(boundary_path).astype(np.float32, copy=False)
    bldg_h = to_float32_depth(np.load(bldg_h_path))
    bldg_conf = to_float32_depth(np.load(bldg_conf_path))

    h, w = depth.shape
    valid_mask = np.isfinite(depth) & (depth > 0)

    # 2. Build Structural Layer Estimate (Z_struct)
    z_struct = np.copy(ground)
    w_struct = np.full((h, w), 0.5, dtype=np.float32)

    is_ground = (seg_mask == CLASS_GROUND)
    w_struct[is_ground] = 0.85

    is_road = (seg_mask == CLASS_ROAD)
    z_struct[is_road] = ground[is_road]
    w_struct[is_road] = 0.90

    is_veg = (seg_mask == CLASS_VEGETATION)
    raw_veg_height = np.maximum(0.0, ground - depth).astype(np.float32)
    veg_height_smoothed = cv2.medianBlur(raw_veg_height, 5)
    veg_height_smoothed = cv2.GaussianBlur(veg_height_smoothed, (5, 5), 1.5)
    z_struct[is_veg] = ground[is_veg] - veg_height_smoothed[is_veg]
    w_struct[is_veg] = 0.75
    del raw_veg_height, veg_height_smoothed
    clean_memory()

    is_bldg = (seg_mask == CLASS_BUILDING) & (bldg_h > 0)
    z_struct[is_bldg] = ground[is_bldg] - bldg_h[is_bldg]
    w_struct[is_bldg] = 0.95

    kernel_halo = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated_bldg = cv2.dilate(is_bldg.astype(np.uint8), kernel_halo, iterations=2).astype(bool)
    bldg_halo = dilated_bldg & ~is_bldg
    del dilated_bldg
    clean_memory()

    is_unknown = (seg_mask == CLASS_UNKNOWN)
    w_struct[is_unknown] = 0.50

    w_depth = np.clip(1.0 - boundary, 0.05, 1.0).astype(np.float32)
    w_depth[~valid_mask] = 0.0
    w_depth[is_bldg] = 0.05

    z_struct[bldg_halo] = ground[bldg_halo]
    w_depth[bldg_halo] = 0.05
    w_depth[is_road] = 0.05
    w_depth[is_ground] = 0.10

    print("[SURFACE_FUSION] Performing confidence-weighted fusion...")
    numerator = (w_struct * z_struct) + (w_depth * depth)
    denominator = w_struct + w_depth + 1e-6
    z_fused = (numerator / denominator).astype(np.float32)
    del numerator, denominator, w_struct, w_depth
    clean_memory()

    z_fused[is_bldg] = z_struct[is_bldg]
    z_fused[is_road] = ground[is_road]
    z_fused[is_ground] = ground[is_ground]
    z_fused[bldg_halo] = ground[bldg_halo]
    z_fused[~valid_mask] = np.nan

    surface_confidence = np.clip((0.5 + 0.5 * (1.0 - boundary)), 0.0, 1.0).astype(np.float32)
    surface_confidence[~valid_mask] = 0.0

    log_memory_usage("Surface Fusion", {
        "z_fused": z_fused,
        "surface_confidence": surface_confidence
    })

    # Save Output Arrays as float32
    np.save(os.path.join(output_dir, "final_relative_surface.npy"), z_fused)
    np.save(os.path.join(output_dir, "surface_confidence_map.npy"), surface_confidence)

    # Save Visualizations (Memory-Safe)
    invalid_mask_img = ((~valid_mask) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "invalid_mask.png"), invalid_mask_img)

    p05 = float(np.nanpercentile(z_fused, 5))
    p95 = float(np.nanpercentile(z_fused, 95))
    save_colormap_safe(z_fused, os.path.join(output_dir, "final_relative_surface.png"), cmap="turbo", vmin=p05, vmax=p95)

    relative_height_above_ground = np.maximum(0.0, ground - z_fused).astype(np.float32)
    relative_height_above_ground[~valid_mask] = 0.0
    max_h_vis = float(np.percentile(relative_height_above_ground[valid_mask], 98)) if np.any(valid_mask) else 1.0
    save_colormap_safe(relative_height_above_ground, os.path.join(output_dir, "surface_visualization.png"), cmap="magma", vmin=0.0, vmax=max_h_vis)
    save_colormap_safe(surface_confidence, os.path.join(output_dir, "surface_confidence_map.png"), cmap="viridis", vmin=0.0, vmax=1.0)

    # Metadata JSON
    stats = {
        "config": config,
        "image_dimensions": {"width": w, "height": h},
        "valid_surface_pixel_percentage": float(round((np.sum(valid_mask)/(h*w))*100, 2)),
        "invalid_pixel_percentage": float(round((np.sum(~valid_mask)/(h*w))*100, 2)),
        "surface_depth_stats_m": {
            "min": float(np.nanmin(z_fused)),
            "max": float(np.nanmax(z_fused)),
            "mean": float(np.nanmean(z_fused)),
            "median": float(np.nanmedian(z_fused)),
            "std_dev": float(np.nanstd(z_fused))
        },
        "mean_surface_confidence": float(round(np.mean(surface_confidence[valid_mask]), 3))
    }

    with open(os.path.join(output_dir, "surface_fusion_metadata.json"), "w") as f:
        json.dump(stats, f, indent=4)

    del depth, ground, seg_mask, boundary, bldg_h, bldg_conf, z_fused, surface_confidence
    clean_memory()
    print(f"[SURFACE_FUSION] Finished successfully. All outputs written to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Surface Fusion Stage")
    parser.add_argument("--depth", required=True, help="Path to cleaned depth.npy")
    parser.add_argument("--ground", required=True, help="Path to ground_surface.npy")
    parser.add_argument("--segmentation", required=True, help="Path to segmentation_mask.npy")
    parser.add_argument("--boundary", required=True, help="Path to boundary_confidence.npy")
    parser.add_argument("--building_height", required=True, help="Path to building_height_map.npy")
    parser.add_argument("--building_conf", required=True, help="Path to building_confidence.npy")
    parser.add_argument("--output", required=True, help="Output directory path")

    parser.add_argument("--boundary_thresh", type=float, default=0.35, help="Boundary confidence threshold for sharp edges")

    args = parser.parse_args()

    config = {
        "boundary_thresh": args.boundary_thresh
    }

    fuse_surface(
        args.depth, args.ground, args.segmentation, args.boundary,
        args.building_height, args.building_conf, args.output, config
    )
