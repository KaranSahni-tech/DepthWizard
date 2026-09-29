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
    input_path = "image2.jpg"
    output_dir = "output/image2_output"

    if not os.path.exists(input_path):
        print(f"ERROR: Could not find input image at {input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print("Loading Depth Pro model...")
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Loading image from {input_path}...")
    image, _, f_px = depth_pro.load_rgb(input_path)
    image_tensor = transform(image).to(device)

    # Print input info
    print("Input width")
    print(image.shape[1])
    print("Input height")
    print(image.shape[0])
    channels = image.shape[2] if len(image.shape) > 2 else 1
    print("Input channels")
    print(channels)

    print(f"Running inference on {device}...")
    start_time = time.time()
    
    # Run inference
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = model.infer(image_tensor, f_px=f_px)
        
    inference_time = time.time() - start_time
    
    # 2. PRESERVE THE RAW DEPTH
    depth = prediction["depth"].cpu().numpy()  
    
    print("Depth width")
    print(depth.shape[1])
    print("Depth height")
    print(depth.shape[0])

    focallength_px = prediction.get("focallength_px", None)
    if isinstance(focallength_px, torch.Tensor):
        focallength_px = float(focallength_px.item())
    elif focallength_px is not None:
        focallength_px = float(focallength_px)

    # 3. VERIFY THE DEPTH VALUES
    valid = np.isfinite(depth)
    valid_depth = depth[valid]
    
    nan_count = int(np.isnan(depth).sum())
    inf_count = int(np.isinf(depth).sum())
    valid_count = int(valid.sum())
    
    # Calculate stats
    min_val = float(np.min(valid_depth)) if valid_count > 0 else 0.0
    max_val = float(np.max(valid_depth)) if valid_count > 0 else 0.0
    mean_val = float(np.mean(valid_depth)) if valid_count > 0 else 0.0
    median_val = float(np.median(valid_depth)) if valid_count > 0 else 0.0
    std_val = float(np.std(valid_depth)) if valid_count > 0 else 0.0
    
    p02 = float(np.percentile(valid_depth, 2)) if valid_count > 0 else 0.0
    p05 = float(np.percentile(valid_depth, 5)) if valid_count > 0 else 0.0
    p95 = float(np.percentile(valid_depth, 95)) if valid_count > 0 else 0.0
    p98 = float(np.percentile(valid_depth, 98)) if valid_count > 0 else 0.0

    print("Shape:", depth.shape)
    print("Dtype:", depth.dtype)
    print("Min:", min_val)
    print("Max:", max_val)
    print("Mean:", mean_val)
    print("Median:", median_val)
    print("Std:", std_val)
    print("NaN:", nan_count)
    print("Inf:", inf_count)

    # Save raw depth
    npy_path = os.path.join(output_dir, "depth.npy")
    np.save(npy_path, depth)

    # 7. SAVE FLOATING-POINT TIFF
    tiff_path = os.path.join(output_dir, "depth.tiff")
    tifffile.imwrite(tiff_path, depth.astype(np.float32))

    # 4. CREATE A HIGH-QUALITY DEPTH VISUALIZATION
    # We use p02 and p98 for robust visualization
    visual = np.clip(depth, p02, p98)
    # Normalize to 0-1
    if p98 - p02 > 1e-6:
        visual = (visual - p02) / (p98 - p02)
    else:
        visual = np.zeros_like(visual)

    # Note: Depth pro produces metric depth. Often, closer is smaller (if metric distance),
    # or closer is larger (if disparity). Usually, it's metric depth (meters). 
    # The user asked for "high-quality depth visualization" - turbo colormap is good.
    cmap = plt.get_cmap("turbo")
    # For depth visualization, typically near=bright, far=dark. 
    # Or just use the colormap directly.
    # Inverse it so near is bright, or just map directly. 
    # The user just said:
    # visual = np.clip(depth, p2, p98)
    # visual = (visual - p2) / (p98 - p2)
    # I will strictly follow the user's formula.
    vis_colored = (cmap(visual)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(vis_colored).save(os.path.join(output_dir, "depth_visualization.png"))

    # 6. CREATE A GRAYSCALE DEPTH IMAGE TOO
    vis_gray = (visual * 255).astype(np.uint8)
    Image.fromarray(vis_gray, mode='L').save(os.path.join(output_dir, "depth_grayscale.png"))

    # 11 & 12. CREATE metadata.json
    depth_range = max_val - min_val
    p05_p95_range = p95 - p05
    relative_variation = p05_p95_range / (median_val + 1e-6)

    if p05_p95_range < 0.1 or std_val < 0.05:
        print("WARNING: The model is producing low depth variation. The scene might be interpreted as mostly flat.")

    metadata = {
        "input_image": os.path.basename(input_path),
        "model": "Apple Depth Pro",
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "depth_width": depth.shape[1],
        "depth_height": depth.shape[0],
        "depth_shape": list(depth.shape),
        "depth_dtype": str(depth.dtype),
        "depth_units": "meters",
        "min_depth": min_val,
        "max_depth": max_val,
        "mean_depth": mean_val,
        "median_depth": median_val,
        "std_depth": std_val,
        "p02_depth": p02,
        "p05_depth": p05,
        "p95_depth": p95,
        "p98_depth": p98,
        "valid_pixel_count": valid_count,
        "nan_count": nan_count,
        "infinite_count": inf_count,
        "focal_length_px": focallength_px,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "depth_file": "depth.npy",
        "depth_tiff": "depth.tiff",
        "visualization_file": "depth_visualization.png",
        "grayscale_file": "depth_grayscale.png",
        "depth_quality": {
            "depth_range": depth_range,
            "p05_p95_range": p05_p95_range,
            "standard_deviation": std_val,
            "relative_variation": relative_variation
        }
    }
    
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # 15. COMPARE THE RESULT
    print("")
    print("========================================")
    print("DEPTH PRO SATELLITE TEST")
    print("========================================")
    print("")
    print("Input:")
    print(os.path.basename(input_path))
    print("")
    print("Depth shape:")
    print(depth.shape)
    print("")
    print("Depth dtype:")
    print(depth.dtype)
    print("")
    print("Min:")
    print(f"{min_val}")
    print("")
    print("Max:")
    print(f"{max_val}")
    print("")
    print("Mean:")
    print(f"{mean_val}")
    print("")
    print("Median:")
    print(f"{median_val}")
    print("")
    print("Std:")
    print(f"{std_val}")
    print("")
    print("P05:")
    print(f"{p05}")
    print("")
    print("P95:")
    print(f"{p95}")
    print("")
    print("P95-P05:")
    print(f"{p05_p95_range}")
    print("")
    print("Valid pixels:")
    print(f"{valid_count}")
    print("")
    print("Device:")
    print(metadata["device"])
    print("")
    print("========================================")

if __name__ == "__main__":
    main()
