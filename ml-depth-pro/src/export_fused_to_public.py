import os
import json
import numpy as np

def main():
    surface_path = "output/image3_output/surface_fusion_test/final_relative_surface.npy"
    public_json_path = "point-cloud-viewer/public/depth.json"

    if not os.path.exists(surface_path):
        print(f"[ERROR] {surface_path} does not exist.")
        return

    surface = np.load(surface_path)
    h, w = surface.shape

    valid = np.isfinite(surface) & (surface > 0)
    valid_vals = surface[valid]

    f_px = 1342.898681640625

    # Prepare depth.json structure expected by frontend
    depth_data = {
        "image_width": int(w),
        "image_height": int(h),
        "depth_width": int(w),
        "depth_height": int(h),
        "min_depth": float(np.min(valid_vals)),
        "max_depth": float(np.max(valid_vals)),
        "focal_length": f_px,
        "principal_point": [w / 2.0, h / 2.0],
        "depth_array": surface.tolist()
    }

    print(f"Writing structure-aware relative surface model to {public_json_path}...")
    with open(public_json_path, "w") as f:
        json.dump(depth_data, f)
    print("Export complete.")

if __name__ == "__main__":
    main()
