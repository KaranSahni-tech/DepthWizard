import sys
import os
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
import depth_pro

def main():
    if len(sys.argv) >= 3:
        input_path = sys.argv[1]
        output_dir = sys.argv[2]
    else:
        input_path = "data/satellite.jpg"
        output_dir = "output/satellite_output"
    
    if not os.path.exists(input_path):
        print(f"ERROR: Cannot find {input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    model, transform = depth_pro.create_model_and_transforms(precision=torch.half)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    image, _, f_px = depth_pro.load_rgb(input_path)
    image_tensor = transform(image).to(device)
    
    print(f"Input width\n{image.shape[1]}")
    print(f"Input height\n{image.shape[0]}")
    print(f"Input channels\n{image.shape[2] if len(image.shape) > 2 else 1}")

    start_time = time.time()
    
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = model.infer(image_tensor, f_px=f_px)
        
    inference_time = time.time() - start_time
    
    # 2. PRESERVE THE RAW DEPTH
    depth = prediction["depth"].cpu().numpy()  
    
    print(f"Depth width\n{depth.shape[1]}")
    print(f"Depth height\n{depth.shape[0]}")

    if isinstance(f_px, torch.Tensor):
        f_px = f_px.item()
    elif prediction["focallength_px"] is not None:
        focallength_px = prediction["focallength_px"]
        if isinstance(focallength_px, torch.Tensor):
            f_px = focallength_px.item()
        else:
            f_px = float(focallength_px)

    # 3. VERIFY THE DEPTH VALUES
    valid = np.isfinite(depth)
    valid_depth = depth[valid]
    
    nan_count = int(np.isnan(depth).sum())
    inf_count = int(np.isinf(depth).sum())
    valid_count = int(valid.sum())
    
    min_depth = float(np.min(valid_depth))
    max_depth = float(np.max(valid_depth))
    mean_depth = float(np.mean(valid_depth))
    median_depth = float(np.median(valid_depth))
    std_depth = float(np.std(valid_depth))
    
    p02 = float(np.percentile(valid_depth, 2))
    p05 = float(np.percentile(valid_depth, 5))
    p95 = float(np.percentile(valid_depth, 95))
    p98 = float(np.percentile(valid_depth, 98))
    
    # Save raw npy
    npy_path = os.path.join(output_dir, "depth.npy")
    np.save(npy_path, depth)

    # 7. SAVE FLOATING-POINT TIFF
    tiff_path = os.path.join(output_dir, "depth.tiff")
    tifffile.imwrite(tiff_path, depth.astype(np.float32))

    # 4. CREATE A HIGH-QUALITY DEPTH VISUALIZATION
    visual = np.clip(depth, p02, p98)
    visual = (visual - p02) / (p98 - p02 + 1e-8)
    
    visual_inv = 1.0 - visual
    cmap = plt.get_cmap("turbo")
    vis_colored = (cmap(visual_inv)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(vis_colored).save(os.path.join(output_dir, "depth_visualization.png"))

    # 6. CREATE A GRAYSCALE DEPTH IMAGE TOO
    vis_gray = (visual_inv * 255).astype(np.uint8)
    Image.fromarray(vis_gray, mode='L').save(os.path.join(output_dir, "depth_grayscale.png"))

    # 11 & 12. CREATE metadata.json
    metadata = {
        "input_image": os.path.basename(input_path),
        "model": "Apple Depth Pro",
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "depth_width": int(depth.shape[1]),
        "depth_height": int(depth.shape[0]),
        "depth_shape": list(depth.shape),
        "depth_dtype": str(depth.dtype),
        "depth_units": "meters",
        "min_depth": min_depth,
        "max_depth": max_depth,
        "mean_depth": mean_depth,
        "median_depth": median_depth,
        "std_depth": std_depth,
        "p02_depth": p02,
        "p05_depth": p05,
        "p95_depth": p95,
        "p98_depth": p98,
        "valid_pixel_count": valid_count,
        "nan_count": nan_count,
        "infinite_count": inf_count,
        "focal_length_px": f_px if f_px is not None else None,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "depth_file": "depth.npy",
        "depth_tiff": "depth.tiff",
        "visualization_file": "depth_visualization.png",
        "grayscale_file": "depth_grayscale.png",
        "depth_quality": {
            "depth_range": max_depth - min_depth,
            "p05_p95_range": p95 - p05,
            "standard_deviation": std_depth,
            "relative_variation": (p95 - p05) / (median_depth + 1e-6)
        }
    }
    
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("========================================")
    print("DEPTH PRO SATELLITE TEST")
    print("========================================")
    print(f"\nInput:\n{os.path.basename(input_path)}")
    print(f"\nDepth shape:\n{depth.shape}")
    print(f"\nDepth dtype:\n{depth.dtype}")
    print(f"\nMin:\n{min_depth:.3f}")
    print(f"\nMax:\n{max_depth:.3f}")
    print(f"\nMean:\n{mean_depth:.3f}")
    print(f"\nMedian:\n{median_depth:.3f}")
    print(f"\nStd:\n{std_depth:.3f}")
    print(f"\nP05:\n{p05:.3f}")
    print(f"\nP95:\n{p95:.3f}")
    print(f"\nP95-P05:\n{(p95-p05):.3f}")
    print(f"\nValid pixels:\n{valid_count}")
    print(f"\nDevice:\n{metadata['device']}")
    print("========================================")

if __name__ == "__main__":
    main()
