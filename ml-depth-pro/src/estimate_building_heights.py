import os
import sys
import json
import argparse
import numpy as np
import cv2
from memory_utils import to_float32_depth, save_colormap_safe, log_memory_usage, clean_memory

CLASS_BUILDING = 1

def estimate_building_heights(depth_path, ground_path, seg_path, boundary_path, output_dir, config):
    print(f"[BUILDING_HEIGHTS] Loading input maps...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Inputs (Enforce float32 & uint8)
    depth = to_float32_depth(np.load(depth_path))
    ground = to_float32_depth(np.load(ground_path))
    seg_mask = np.load(seg_path).astype(np.uint8, copy=False)
    boundary = np.load(boundary_path).astype(np.float32, copy=False)

    h, w = depth.shape
    valid_mask = np.isfinite(depth) & (depth > 0)

    # Raw height above ground across the entire scene
    height_above_ground_raw = np.maximum(0.0, ground - depth).astype(np.float32)
    height_above_ground_raw[~valid_mask] = 0.0

    all_hag_valid = height_above_ground_raw[valid_mask]
    if len(all_hag_valid) > 0 and np.max(all_hag_valid) > 1e-6:
        adaptive_min_bldg_h = float(np.percentile(all_hag_valid, 50))
        p99_hag = float(np.percentile(all_hag_valid, 99))
        adaptive_min_bldg_h = max(0.04, min(adaptive_min_bldg_h, p99_hag * 0.25))
    else:
        adaptive_min_bldg_h = config['min_building_height']

    print(f"[BUILDING_HEIGHTS] Adaptive min_building_height: {adaptive_min_bldg_h:.6f}m")

    # 2. Extract and Sever Building Candidates
    building_mask = (seg_mask == CLASS_BUILDING) & valid_mask
    strong_boundary = (boundary >= config['boundary_thresh'])
    kernel_bound = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_boundary = cv2.dilate(strong_boundary.astype(np.uint8), kernel_bound, iterations=1).astype(bool)

    severed_building_mask = building_mask & ~dilated_boundary
    del strong_boundary, dilated_boundary
    clean_memory()

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(severed_building_mask.astype(np.uint8), connectivity=8)

    building_height_map = np.zeros((h, w), dtype=np.float32)
    building_confidence = np.zeros((h, w), dtype=np.float32)
    refined_building_mask = np.zeros((h, w), dtype=np.uint8)

    building_records = []

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        instance_pixels = (labels == i)
        raw_h_vals = height_above_ground_raw[instance_pixels]

        if len(raw_h_vals) == 0:
            continue

        p05 = float(np.percentile(raw_h_vals, 5))
        p50 = float(np.median(raw_h_vals))
        p95 = float(np.percentile(raw_h_vals, 95))
        mean_h = float(np.mean(raw_h_vals))
        std_h = float(np.std(raw_h_vals))

        if area < config['min_building_area']:
            continue

        if p50 < adaptive_min_bldg_h and p95 < adaptive_min_bldg_h:
            continue

        h_instance = np.clip(raw_h_vals, p05, p95).astype(np.float32)

        if std_h <= config['flat_roof_std_thresh']:
            filtered_h = np.full_like(h_instance, p50, dtype=np.float32)
        else:
            filtered_h = np.where(np.abs(h_instance - p50) > 1.5 * std_h, p50, h_instance).astype(np.float32)

        building_height_map[instance_pixels] = filtered_h
        refined_building_mask[instance_pixels] = 255

        area_score = min(1.0, area / 300.0)
        consistency_score = max(0.0, 1.0 - (std_h / (p50 + 1e-3)))
        conf_score = float(round(0.5 * area_score + 0.5 * consistency_score, 3))
        building_confidence[instance_pixels] = conf_score

        building_records.append({
            "building_id": i,
            "area_pixels": area,
            "centroid": [float(centroids[i][0]), float(centroids[i][1])],
            "min_height_m": round(p05, 3),
            "max_height_m": round(p95, 3),
            "median_height_m": round(p50, 3),
            "mean_height_m": round(mean_h, 3),
            "std_dev_m": round(std_h, 3),
            "confidence": conf_score
        })

    del labels, stats, centroids
    clean_memory()

    valid_buildings_count = len(building_records)
    print(f"[BUILDING_HEIGHTS] Validated {valid_buildings_count} distinct building structures.")

    log_memory_usage("Estimate Building Heights", {
        "building_height_map": building_height_map,
        "building_confidence": building_confidence,
        "height_above_ground_raw": height_above_ground_raw
    })

    # Save Array Outputs
    np.save(os.path.join(output_dir, "building_height_map.npy"), building_height_map)
    np.save(os.path.join(output_dir, "building_confidence.npy"), building_confidence)
    np.save(os.path.join(output_dir, "height_above_ground.npy"), height_above_ground_raw)

    # Save Visualizations (Memory-Safe)
    cv2.imwrite(os.path.join(output_dir, "building_mask_refined.png"), refined_building_mask)

    max_h_vis = float(np.percentile(building_height_map[building_height_map > 0], 98)) if np.any(building_height_map > 0) else 1.0
    save_colormap_safe(building_height_map, os.path.join(output_dir, "building_height_map.png"), cmap="magma", vmin=0.0, vmax=max_h_vis)

    max_hag_vis = float(np.percentile(height_above_ground_raw[valid_mask], 98)) if np.any(valid_mask) else 1.0
    save_colormap_safe(height_above_ground_raw, os.path.join(output_dir, "height_above_ground_visualization.png"), cmap="plasma", vmin=0.0, vmax=max_hag_vis)
    save_colormap_safe(building_confidence, os.path.join(output_dir, "building_confidence.png"), cmap="viridis", vmin=0.0, vmax=1.0)

    # Save Metadata JSON
    metadata = {
        "config": config,
        "image_dimensions": {"width": w, "height": h},
        "total_building_structures_detected": valid_buildings_count,
        "max_building_height_m": float(np.max(building_height_map)) if valid_buildings_count > 0 else 0.0,
        "mean_building_height_m": float(np.mean(building_height_map[building_height_map > 0])) if valid_buildings_count > 0 else 0.0,
        "buildings": building_records
    }

    with open(os.path.join(output_dir, "building_height_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    del depth, ground, seg_mask, boundary, building_height_map, building_confidence, height_above_ground_raw
    clean_memory()
    print(f"[BUILDING_HEIGHTS] Finished successfully. All outputs written to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Building / Structure Height Estimation Stage")
    parser.add_argument("--depth", required=True, help="Path to cleaned depth.npy")
    parser.add_argument("--ground", required=True, help="Path to ground_surface.npy")
    parser.add_argument("--segmentation", required=True, help="Path to segmentation_mask.npy")
    parser.add_argument("--boundary", required=True, help="Path to boundary_confidence.npy")
    parser.add_argument("--output", required=True, help="Output directory path")

    parser.add_argument("--boundary_thresh", type=float, default=0.35, help="Boundary confidence threshold for separating adjacent buildings")
    parser.add_argument("--min_building_area", type=int, default=50, help="Minimum pixel footprint area for building candidates")
    parser.add_argument("--min_building_height", type=float, default=0.2, help="Minimum height above ground to qualify as building (m)")
    parser.add_argument("--flat_roof_std_thresh", type=float, default=0.15, help="Max height std-dev to treat as flat roof (m)")

    args = parser.parse_args()

    config = {
        "boundary_thresh": args.boundary_thresh,
        "min_building_area": args.min_building_area,
        "min_building_height": args.min_building_height,
        "flat_roof_std_thresh": args.flat_roof_std_thresh
    }

    estimate_building_heights(args.depth, args.ground, args.segmentation, args.boundary, args.output, config)
