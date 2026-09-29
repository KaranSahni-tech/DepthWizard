import os
import sys
import json
import argparse
import numpy as np
import cv2
from memory_utils import to_uint8_image, to_float32_depth, log_memory_usage, clean_memory

# Define Segmentation Class Constants
CLASS_UNKNOWN = 0
CLASS_BUILDING = 1
CLASS_ROAD = 2
CLASS_GROUND = 3
CLASS_VEGETATION = 4

CLASS_NAMES = {
    0: "UNKNOWN",
    1: "BUILDING",
    2: "ROAD",
    3: "GROUND",
    4: "VEGETATION"
}

# Visualization Color Palette (RGB format)
CLASS_COLORS = {
    0: [40, 40, 40],       # Unknown: Dark Gray
    1: [230, 57, 70],      # Building: Red / Coral
    2: [69, 123, 157],     # Road: Blue Gray
    3: [233, 196, 106],    # Ground: Sand / Warm Yellow
    4: [42, 157, 143]      # Vegetation: Teal / Green
}

def segment_scene(image_path, depth_path, boundary_path, output_dir, config):
    print(f"[SEGMENTATION] Memory-safe segmentation for {image_path}...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Inputs (Enforce uint8 RGB and float32 depth)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[ERROR] Could not load RGB image: {image_path}")
        sys.exit(1)
    
    img_rgb = to_uint8_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    depth = to_float32_depth(np.load(depth_path))
    boundary = np.load(boundary_path).astype(np.float32, copy=False)

    h, w = depth.shape

    if img_rgb.shape[:2] != (h, w):
        print(f"[WARNING] Resizing RGB image from {img_rgb.shape[:2]} to match depth {(h, w)}")
        img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_AREA)
        img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)

    if boundary.shape != (h, w):
        boundary = cv2.resize(boundary, (w, h), interpolation=cv2.INTER_LINEAR)

    valid_mask = np.isfinite(depth) & (depth > 0)
    valid_depths = depth[valid_mask]

    if len(valid_depths) == 0:
        print(f"[ERROR] Depth map contains no valid positive depth values!")
        sys.exit(1)

    # 2. Relative Elevation Calculation (float32)
    ground_ref = float(np.percentile(valid_depths, config['ground_percentile']))
    relative_height = np.maximum(0.0, ground_ref - depth).astype(np.float32)
    relative_height[~valid_mask] = 0.0

    print(f"[SEGMENTATION] Ground reference depth: {ground_ref:.3f}m | Max relative height: {np.max(relative_height):.3f}m")

    rel_valid = relative_height[valid_mask]
    if len(rel_valid) > 0:
        rel_p40 = float(np.percentile(rel_valid, 40))
        rel_p50 = float(np.percentile(rel_valid, 50))
        rel_p60 = float(np.percentile(rel_valid, 60))
        rel_p65 = float(np.percentile(rel_valid, 65))
    else:
        rel_p40, rel_p50, rel_p60, rel_p65 = 0.2, 0.3, 0.5, 0.55

    building_min_h = max(0.04, rel_p65)
    ground_max_h = rel_p50
    road_max_h = rel_p40

    min_building_area = max(10, int((w * h) * 0.0001))

    # A. Vegetation Features (ExG + HSV green)
    r, g, b = img_rgb[:, :, 0].astype(np.float32), img_rgb[:, :, 1].astype(np.float32), img_rgb[:, :, 2].astype(np.float32)
    exg = 2.0 * g - r - b
    del r, g, b
    clean_memory()

    exg_thresh = max(0.01, float(np.percentile(exg[valid_mask], 75)))

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]
    del hsv
    clean_memory()

    green_hsv = (hue >= config['veg_hue_min']) & (hue <= config['veg_hue_max']) & (sat >= config['veg_sat_min'])
    del hue

    depth_mean = cv2.blur(depth, (5, 5))
    depth_sq_mean = cv2.blur(depth**2, (5, 5))
    depth_var = np.maximum(depth_sq_mean - depth_mean**2, 0)
    depth_roughness = np.sqrt(depth_var).astype(np.float32)
    del depth_mean, depth_sq_mean, depth_var
    clean_memory()

    roughness_thresh = max(0.04, float(np.percentile(depth_roughness[valid_mask], 75)))

    is_vegetation = (exg > exg_thresh) | green_hsv
    is_vegetation |= (relative_height > ground_max_h) & (depth_roughness > roughness_thresh) & (sat > 30)
    del exg, green_hsv

    k_veg = max(3, int(min(w, h) * 0.008) | 1)
    kernel_veg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_veg, k_veg))
    is_vegetation = cv2.morphologyEx(is_vegetation.astype(np.uint8), cv2.MORPH_CLOSE, kernel_veg).astype(bool)

    # B. Road Surface Candidates
    is_low_elevation = (relative_height <= ground_max_h)
    is_low_sat = (sat <= config['road_sat_max'])
    is_smooth = (depth_roughness <= config['road_roughness_max'])
    del sat, depth_roughness
    clean_memory()

    is_road = (relative_height <= road_max_h) & is_low_sat & is_smooth & ~is_vegetation & valid_mask
    k_road = max(3, int(min(w, h) * 0.012) | 1)
    kernel_road = cv2.getStructuringElement(cv2.MORPH_RECT, (k_road, k_road))
    is_road = cv2.morphologyEx(is_road.astype(np.uint8), cv2.MORPH_OPEN, kernel_road).astype(bool)

    # C. Ground / Open Surface Candidates
    is_ground = is_low_elevation & ~is_road & ~is_vegetation & valid_mask

    # D. Building / Structure Candidates
    is_building_candidate = (relative_height > building_min_h) & ~is_vegetation & valid_mask

    boundary_thresh = max(0.20, float(np.percentile(boundary[valid_mask], 75)))
    strong_boundary = (boundary >= boundary_thresh)
    kernel_bound = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_boundary = cv2.dilate(strong_boundary.astype(np.uint8), kernel_bound, iterations=1).astype(bool)
    del strong_boundary

    severed_buildings = is_building_candidate & ~dilated_boundary
    del dilated_boundary
    clean_memory()

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(severed_buildings.astype(np.uint8), connectivity=8)
    del severed_buildings

    is_building = np.zeros((h, w), dtype=bool)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_building_area:
            is_building[labels == i] = True
    del labels, stats, centroids

    is_building = cv2.dilate(is_building.astype(np.uint8), kernel_bound, iterations=2).astype(bool)
    is_building &= is_building_candidate
    del is_building_candidate

    # 5. Assemble Final Multi-Class Segmentation Mask (uint8)
    seg_mask = np.full((h, w), CLASS_UNKNOWN, dtype=np.uint8)
    seg_mask[is_ground] = CLASS_GROUND
    seg_mask[is_road] = CLASS_ROAD
    seg_mask[is_vegetation] = CLASS_VEGETATION
    seg_mask[is_building] = CLASS_BUILDING
    seg_mask[~valid_mask] = CLASS_UNKNOWN
    del is_ground, is_road, is_vegetation, is_building

    seg_mask = cv2.medianBlur(seg_mask, 3)

    log_memory_usage("Segment Scene", {
        "seg_mask": seg_mask,
        "img_rgb": img_rgb
    })

    # 6. Generate Individual Masks
    cv2.imwrite(os.path.join(output_dir, "building_mask.png"), ((seg_mask == CLASS_BUILDING) * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, "road_mask.png"), ((seg_mask == CLASS_ROAD) * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, "ground_mask.png"), ((seg_mask == CLASS_GROUND) * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, "vegetation_mask.png"), ((seg_mask == CLASS_VEGETATION) * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, "unknown_mask.png"), ((seg_mask == CLASS_UNKNOWN) * 255).astype(np.uint8))

    np.save(os.path.join(output_dir, "segmentation_mask.npy"), seg_mask)

    # 7. Create Color-Coded Segmentation Image
    color_seg = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in CLASS_COLORS.items():
        color_seg[seg_mask == cid] = color

    cv2.imwrite(os.path.join(output_dir, "segmentation_mask.png"), cv2.cvtColor(color_seg, cv2.COLOR_RGB2BGR))

    # Overlay
    alpha = config['overlay_alpha']
    overlay = cv2.addWeighted(img_rgb, 1.0 - alpha, color_seg, alpha, 0)
    cv2.imwrite(os.path.join(output_dir, "segmentation_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    del color_seg, overlay

    # Metadata JSON
    total_pixels = h * w
    class_stats = {}
    for cid, name in CLASS_NAMES.items():
        count = int(np.sum(seg_mask == cid))
        percentage = float((count / total_pixels) * 100)
        class_stats[name] = {
            "class_id": cid,
            "pixel_count": count,
            "percentage": round(percentage, 2)
        }

    metadata = {
        "config": config,
        "image_dimensions": {"width": w, "height": h},
        "ground_reference_depth_m": float(ground_ref),
        "max_relative_height_m": float(np.max(relative_height)),
        "classes": class_stats
    }

    with open(os.path.join(output_dir, "segmentation_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    del depth, boundary, seg_mask, img_rgb, img_bgr
    clean_memory()
    print(f"[SEGMENTATION] Finished successfully. All outputs written to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structure-Aware Scene Segmentation Stage")
    parser.add_argument("--image", required=True, help="Path to input RGB image")
    parser.add_argument("--depth", required=True, help="Path to cleaned relative depth.npy")
    parser.add_argument("--boundary", required=True, help="Path to boundary_confidence.npy")
    parser.add_argument("--output", required=True, help="Output directory path")

    parser.add_argument("--ground_percentile", type=float, default=95.0, help="Percentile depth for ground reference")
    parser.add_argument("--building_min_h", type=float, default=1.0, help="Min relative elevation for buildings (m)")
    parser.add_argument("--ground_max_h", type=float, default=0.8, help="Max relative elevation for ground (m)")
    parser.add_argument("--road_max_h", type=float, default=0.6, help="Max relative elevation for roads (m)")
    parser.add_argument("--road_sat_max", type=int, default=80, help="Max HSV saturation for roads (0-255)")
    parser.add_argument("--road_roughness_max", type=float, default=0.08, help="Max depth roughness for smooth roads")
    parser.add_argument("--exg_thresh", type=float, default=0.04, help="Excess Green Index threshold")
    parser.add_argument("--veg_hue_min", type=int, default=25, help="Min HSV hue for green vegetation (0-180)")
    parser.add_argument("--veg_hue_max", type=int, default=85, help="Max HSV hue for green vegetation (0-180)")
    parser.add_argument("--veg_sat_min", type=int, default=30, help="Min HSV saturation for vegetation (0-255)")
    parser.add_argument("--roughness_thresh", type=float, default=0.12, help="Depth roughness threshold for foliage")
    parser.add_argument("--boundary_thresh", type=float, default=0.35, help="Boundary confidence threshold for splitting structures")
    parser.add_argument("--min_building_area", type=int, default=80, help="Minimum pixel area for building candidate regions")
    parser.add_argument("--overlay_alpha", type=float, default=0.45, help="Alpha transparency for segmentation overlay")

    args = parser.parse_args()

    config = {
        "ground_percentile": args.ground_percentile,
        "building_min_h": args.building_min_h,
        "ground_max_h": args.ground_max_h,
        "road_max_h": args.road_max_h,
        "road_sat_max": args.road_sat_max,
        "road_roughness_max": args.road_roughness_max,
        "exg_thresh": args.exg_thresh,
        "veg_hue_min": args.veg_hue_min,
        "veg_hue_max": args.veg_hue_max,
        "veg_sat_min": args.veg_sat_min,
        "roughness_thresh": args.roughness_thresh,
        "boundary_thresh": args.boundary_thresh,
        "min_building_area": args.min_building_area,
        "overlay_alpha": args.overlay_alpha
    }

    segment_scene(args.image, args.depth, args.boundary, args.output, config)
