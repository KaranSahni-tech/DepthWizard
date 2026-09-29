import os
import sys
import numpy as np
import cv2
from memory_utils import to_float32_depth, save_colormap_safe, log_memory_usage, clean_memory

def detect_boundaries(image_path, depth_npy_path, output_dir):
    print(f"[BOUNDARY] Memory-safe structure & boundary detection for {image_path}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not load RGB image: {image_path}")
        sys.exit(1)
        
    depth = to_float32_depth(np.load(depth_npy_path))
    h, w = depth.shape
    
    # Verify alignment
    if img.shape[:2] != (h, w):
        print(f"[WARNING] Dimension mismatch, resizing RGB to match Depth. RGB: {img.shape[:2]} Depth: {(h, w)}")
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        
    # A. RGB Edge Detection (uint8 -> float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    del img
    clean_memory()

    blurred_gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    del gray
    clean_memory()

    edges_rgb = cv2.Canny(blurred_gray, threshold1=30, threshold2=100)
    del blurred_gray
    clean_memory()

    edges_rgb_norm = (edges_rgb.astype(np.float32) / 255.0)
    del edges_rgb
    clean_memory()

    # B. Depth Gradient Detection (float32, CV_32F)
    valid_mask = np.isfinite(depth) & (depth > 0)
    valid_depths = depth[valid_mask]
    med_val = float(np.median(valid_depths)) if len(valid_depths) > 0 else 2.5

    depth_clean = np.copy(depth)
    depth_clean[~valid_mask] = med_val

    # Use CV_32F to avoid float64 memory allocation
    grad_x = cv2.Scharr(depth_clean, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(depth_clean, cv2.CV_32F, 0, 1)
    del depth_clean
    clean_memory()

    grad_mag = cv2.magnitude(grad_x, grad_y)
    del grad_x, grad_y
    clean_memory()

    grad_mag[~valid_mask] = 0.0
    
    p98_grad = float(np.percentile(grad_mag[valid_mask], 98)) if np.any(valid_mask) else 1.0
    if p98_grad == 0: p98_grad = 1.0
    
    edges_depth_norm = np.clip(grad_mag / p98_grad, 0.0, 1.0).astype(np.float32)
    del grad_mag
    clean_memory()

    # C. Combined Boundary Confidence Map (float32)
    confidence_map = (edges_depth_norm * 0.7) + (edges_rgb_norm * edges_depth_norm * 0.3)
    confidence_map = np.clip(confidence_map, 0.0, 1.0).astype(np.float32)

    log_memory_usage("Detect Boundaries", {
        "edges_rgb_norm": edges_rgb_norm,
        "edges_depth_norm": edges_depth_norm,
        "confidence_map": confidence_map
    })

    # Save visualization previews using OpenCV (memory-safe, uint8, zero Matplotlib float64 RGBA overhead)
    save_colormap_safe(edges_rgb_norm, os.path.join(output_dir, "boundary_rgb_edges.png"), cmap="gray")
    save_colormap_safe(edges_depth_norm, os.path.join(output_dir, "boundary_depth_gradient.png"), cmap="magma")
    save_colormap_safe(confidence_map, os.path.join(output_dir, "boundary_combined.png"), cmap="plasma")
    save_colormap_safe(confidence_map, os.path.join(output_dir, "boundary_confidence.png"), cmap="jet")
    
    # Save combined map as float32 NPY
    np.save(os.path.join(output_dir, "boundary_confidence.npy"), confidence_map)
    
    del edges_rgb_norm, edges_depth_norm
    clean_memory()
    print(f"[BOUNDARY] Complete. Memory-safe outputs saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input RGB image")
    parser.add_argument("--depth", required=True, help="Path to cleaned depth.npy")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    
    detect_boundaries(args.image, args.depth, args.output)
