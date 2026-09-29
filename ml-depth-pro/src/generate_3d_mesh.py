import os
import sys
import json
import uuid
import numpy as np
from PIL import Image
import trimesh

# Configuration
MAX_DEPTH_DISCONTINUITY = 1.0  # Max ceiling for adaptive threshold
MAX_EDGE_LENGTH = 10.0
MIN_COMPONENT_FACES = 50

def update_progress(progress_file, status, progress_pct, stage, message, generation_id):
    data = {
        "status": status,
        "progress": progress_pct,
        "stage": stage,
        "message": message,
        "generation_id": generation_id
    }
    with open(progress_file, "w") as f:
        json.dump(data, f)
    print(f"[3D] {stage} ({progress_pct}%): {message}")

def generate_3d_model(image_path, depth_path, heatmap_path, focal_length_px, output_dir, progress_file, generation_id, downsample=1):
    try:
        # Create directory structure
        models_dir = os.path.join(output_dir, "models")
        textures_dir = os.path.join(output_dir, "textures")
        metadata_dir = os.path.join(output_dir, "metadata")
        previews_dir = os.path.join(output_dir, "previews")
        cache_dir = os.path.join(output_dir, "cache")
        
        for d in [models_dir, textures_dir, metadata_dir, previews_dir, cache_dir]:
            os.makedirs(d, exist_ok=True)
            
        update_progress(progress_file, "running", 2, "Loading input data", "Loading RGB and depth", generation_id)
        
        img = Image.open(image_path).convert("RGB")
        original_w, original_h = img.size
        
        heatmap = Image.open(heatmap_path).convert("RGB") if os.path.exists(heatmap_path) else None
        
        depth = np.load(depth_path)
        if depth.shape != (original_h, original_w):
            print(f"[3D] Resizing depth shape {depth.shape} to match image shape {(original_h, original_w)}")
            depth_img = Image.fromarray(depth)
            depth_img = depth_img.resize((original_w, original_h), Image.Resampling.BILINEAR)
            depth = np.array(depth_img)
            
        update_progress(progress_file, "running", 5, "Validating depth", "Checking pixel integrity", generation_id)
        
        # Automatic resolution capping for memory safety & WebGL performance
        max_dim = max(original_w, original_h)
        target_downsample = downsample
        if max_dim > 1400 and target_downsample == 1:
            target_downsample = int(np.ceil(max_dim / 1400.0))
            print(f"[3D] Auto-downsampling high-res input (max_dim={max_dim}) by factor of {target_downsample}")

        if target_downsample > 1:
            new_w = original_w // target_downsample
            new_h = original_h // target_downsample
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            if heatmap:
                heatmap = heatmap.resize((new_w, new_h), Image.Resampling.LANCZOS)
            depth_img = Image.fromarray(depth)
            depth = np.array(depth_img.resize((new_w, new_h), Image.Resampling.BILINEAR))
            focal_length_px /= target_downsample
            
        h, w = depth.shape
        
        valid = np.isfinite(depth)
        valid_count = valid.sum()
        if valid_count == 0:
            raise ValueError("No valid depth pixels found.")
            
        if focal_length_px <= 0:
            focal_length_px = 1342.898681640625
            
        Z = depth

        # Load or compute ground surface
        ground_path = os.path.join(output_dir, "ground_surface.npy")
        if os.path.exists(ground_path):
            ground = np.load(ground_path)
            if ground.shape != (h, w):
                g_img = Image.fromarray(ground).resize((w, h), Image.Resampling.BILINEAR)
                ground = np.array(g_img)
        else:
            from scipy.ndimage import gaussian_filter
            p95 = np.percentile(Z[valid], 95)
            ground = np.full_like(Z, p95)

        # Relative height above ground baseline (0 = ground, >0 = structures/trees)
        height_above_ground = np.maximum(0.0, ground - Z)
        
        update_progress(progress_file, "running", 25, "Generating 3D vertices", f"Processing {valid_count} valid pixels", generation_id)
        
        cx = w / 2.0
        cy = h / 2.0
        fx = fy = focal_length_px
        
        uu, vv = np.meshgrid(np.arange(w), np.arange(h))
        
        # 3D Ortho-rectified camera space:
        # X: Left -> Right
        # Y: Height UPWARDS (+Y is elevated above ground)
        # Z: Top -> Bottom (Terrain grid)
        scale_spatial = 1.0  # Meter metric scale
        X_coord = (uu - cx) * ground / fx
        Z_coord = (vv - cy) * ground / fy
        Y_coord = height_above_ground * scale_spatial

        vertex_indices = np.full((h, w), -1, dtype=int)
        vertex_indices[valid] = np.arange(valid_count)
        
        vertices = np.zeros((valid_count, 3), dtype=np.float32)
        vertices[:, 0] = X_coord[valid]
        vertices[:, 1] = Y_coord[valid]
        vertices[:, 2] = Z_coord[valid]
        
        if np.any(np.isnan(vertices)) or np.any(np.isinf(vertices)):
            raise ValueError("Vertices contain NaN or Infinity")
            
        update_progress(progress_file, "running", 40, "Building 3D surface mesh", "Edge-aware triangulation & wall generation", generation_id)
        
        v_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
        
        idx_A = vertex_indices[:-1, :-1][v_valid]
        idx_B = vertex_indices[:-1, 1:][v_valid]
        idx_C = vertex_indices[1:, :-1][v_valid]
        idx_D = vertex_indices[1:, 1:][v_valid]
        
        h_A = height_above_ground[:-1, :-1][v_valid]
        h_B = height_above_ground[:-1, 1:][v_valid]
        h_C = height_above_ground[1:, :-1][v_valid]
        h_D = height_above_ground[1:, 1:][v_valid]
        
        dh_AB = np.abs(h_A - h_B)
        dh_CD = np.abs(h_C - h_D)
        dh_AC = np.abs(h_A - h_C)
        dh_BD = np.abs(h_B - h_D)
        dh_AD = np.abs(h_A - h_D)
        dh_BC = np.abs(h_B - h_C)
        
        # Max height jump across a single grid cell to allow continuous terrain surface (e.g. 0.8m)
        max_h_step = 0.8
        
        choose_AD = dh_AD < dh_BC
        
        triangles = []
        
        # Option 1: AD is diagonal -> Triangles: A-B-D and A-D-C
        mask_AD = choose_AD
        valid_ABD = mask_AD & (dh_AB < max_h_step) & (dh_BD < max_h_step) & (dh_AD < max_h_step)
        if np.any(valid_ABD):
            triangles.append(np.column_stack((idx_A[valid_ABD], idx_B[valid_ABD], idx_D[valid_ABD])))
            
        valid_ADC = mask_AD & (dh_AD < max_h_step) & (dh_CD < max_h_step) & (dh_AC < max_h_step)
        if np.any(valid_ADC):
            triangles.append(np.column_stack((idx_A[valid_ADC], idx_D[valid_ADC], idx_C[valid_ADC])))
            
        # Option 2: BC is diagonal -> Triangles: A-B-C and B-D-C
        mask_BC = ~choose_AD
        valid_ABC = mask_BC & (dh_AB < max_h_step) & (dh_BC < max_h_step) & (dh_AC < max_h_step)
        if np.any(valid_ABC):
            triangles.append(np.column_stack((idx_A[valid_ABC], idx_B[valid_ABC], idx_C[valid_ABC])))
            
        valid_BDC = mask_BC & (dh_BD < max_h_step) & (dh_CD < max_h_step) & (dh_BC < max_h_step)
        if np.any(valid_BDC):
            triangles.append(np.column_stack((idx_B[valid_BDC], idx_D[valid_BDC], idx_C[valid_BDC])))
            
        if len(triangles) == 0:
            raise ValueError("No valid triangles formed after height step filtering.")
            
        faces = np.vstack(triangles)
        max_possible_triangles = np.sum(v_valid) * 2
        rejected_triangles = max_possible_triangles - len(faces)
        
        update_progress(progress_file, "running", 60, "Optimizing mesh geometry", "Processing connected components", generation_id)
        
        u_tex = uu[valid] / float(w - 1)
        v_tex = vv[valid] / float(h - 1)
        uvs = np.column_stack((u_tex, v_tex)).astype(np.float32)
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        
        # Ensure correct normal orientation (+Y is UP)
        if len(mesh.vertex_normals) > 0:
            mean_ny = np.mean(mesh.vertex_normals[:, 1])
            if mean_ny < 0:
                mesh.invert() # Flips faces so top surfaces face UP (+Y)
            
        update_progress(progress_file, "running", 80, "Applying textures & UV maps", "Generating materials", generation_id)
        
        # Save textures
        rgb_tex_path = os.path.join(textures_dir, "texture_rgb.png")
        img.save(rgb_tex_path)
        
        if heatmap:
            heat_tex_path = os.path.join(textures_dir, "texture_heatmap.png")
            heatmap.save(heat_tex_path)

        # Attach texture visuals with exact UVs
        mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, image=img)
        
        update_progress(progress_file, "running", 90, "Exporting GLB", "Writing 3D GLB model", generation_id)
        glb_path = os.path.join(models_dir, "model.glb")
        mesh.export(glb_path)
        
        if not os.path.exists(glb_path) or os.path.getsize(glb_path) == 0:
            raise ValueError("Exported GLB file is missing or empty.")
            
        bounding_box = mesh.bounding_box.bounds
        height_range = float(bounding_box[1][1] - bounding_box[0][1])
        component_count = 1
        
        metadata = {
            "model_type": "monocular_depth_derived_3d_surface",
            "source_image": os.path.basename(image_path),
            "depth_source": os.path.basename(depth_path),
            "heatmap_source": os.path.basename(heatmap_path) if heatmap_path else None,
            "width": int(w),
            "height": int(h),
            "vertex_count": int(len(mesh.vertices)),
            "triangle_count": int(len(mesh.faces)),
            "rejected_triangles": int(rejected_triangles),
            "component_count": int(component_count),
            "adaptive_threshold": float(max_h_step),
            "median_depth_gradient": float(max_h_step),
            "height_range": height_range,
            "metric_scale": False,
            "absolute_elevation": False,
            "georeferenced": False,
            "crs": None,
            "height_exaggeration": 1.0,
            "mesh_downsample": downsample,
            "generation_id": generation_id
        }
        
        meta_path = os.path.join(metadata_dir, "model_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        preview_path = os.path.join(previews_dir, "model_preview.png")
        img.save(preview_path) 
            
        update_progress(progress_file, "completed", 100, "Complete", "3D model generated successfully", generation_id)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        update_progress(progress_file, "error", 0, "Failed", str(e), generation_id)

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage: python generate_3d_mesh.py <image> <depth> <heatmap> <focal_length> <output_dir> <progress_file> <generation_id> [downsample]")
        sys.exit(1)
        
    downsample = int(sys.argv[8]) if len(sys.argv) > 8 else 1
    
    generate_3d_model(
        image_path=sys.argv[1],
        depth_path=sys.argv[2],
        heatmap_path=sys.argv[3],
        focal_length_px=float(sys.argv[4]),
        output_dir=sys.argv[5],
        progress_file=sys.argv[6],
        generation_id=sys.argv[7],
        downsample=downsample
    )
