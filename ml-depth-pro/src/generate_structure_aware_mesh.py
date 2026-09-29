import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
import trimesh

CLASS_UNKNOWN = 0
CLASS_BUILDING = 1
CLASS_ROAD = 2
CLASS_GROUND = 3
CLASS_VEGETATION = 4

def generate_structure_aware_mesh(surface_path, boundary_path, seg_path, metadata_path, image_path, output_dir, config):
    print(f"[STRUCTURE_MESH] Generating structure-aware 3D mesh with RGB texture...")
    
    models_dir = os.path.join(output_dir, "models")
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # 1. Load Inputs
    surface = np.load(surface_path)
    boundary = np.load(boundary_path)
    seg_mask = np.load(seg_path)

    # Load RGB image for texture
    img_rgb = None
    if image_path and os.path.exists(image_path):
        img_rgb = Image.open(image_path).convert("RGB")

    focal_length_px = 1342.898681640625
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
            if "focal_length_px" in meta:
                focal_length_px = float(meta["focal_length_px"])
            elif "focal_length" in meta:
                focal_length_px = float(meta["focal_length"])

    h, w = surface.shape
    valid = np.isfinite(surface) & (surface > 0)
    valid_count = int(np.sum(valid))

    if valid_count == 0:
        print("[ERROR] No valid surface points found!")
        sys.exit(1)

    print(f"[STRUCTURE_MESH] Surface dimensions: {w}x{h} | Valid vertices: {valid_count} | Focal length: {focal_length_px:.2f}px")

    # 2. 3D Coordinate Generation (Perspective Projection)
    cx = w / 2.0
    cy = h / 2.0
    fx = fy = focal_length_px

    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    
    Z = surface
    X = (uu - cx) * Z / fx
    Y = -(vv - cy) * Z / fy  # Invert Y for standard 3D coordinate system
    Z_coord = -Z             # Invert Z for depth convention

    vertex_indices = np.full((h, w), -1, dtype=int)
    vertex_indices[valid] = np.arange(valid_count)

    vertices = np.zeros((valid_count, 3), dtype=np.float32)
    vertices[:, 0] = X[valid]
    vertices[:, 1] = Y[valid]
    vertices[:, 2] = Z_coord[valid]

    # UV Mapping: u = x / (w - 1), v = 1.0 - (y / (h - 1))
    u_tex = uu[valid] / float(w - 1)
    v_tex = 1.0 - (vv[valid] / float(h - 1))
    uvs = np.column_stack((u_tex, v_tex)).astype(np.float32)

    # 3. 6-Step Edge-Aware & Structure-Aware Triangulation
    print("[STRUCTURE_MESH] Performing 6-step triangle rejection filtering...")
    v_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]

    idx_A = vertex_indices[:-1, :-1][v_valid]
    idx_B = vertex_indices[:-1, 1:][v_valid]
    idx_C = vertex_indices[1:, :-1][v_valid]
    idx_D = vertex_indices[1:, 1:][v_valid]

    z_A = Z[:-1, :-1][v_valid]
    z_B = Z[:-1, 1:][v_valid]
    z_C = Z[1:, :-1][v_valid]
    z_D = Z[1:, 1:][v_valid]

    bound_A = boundary[:-1, :-1][v_valid]
    bound_B = boundary[:-1, 1:][v_valid]
    bound_C = boundary[1:, :-1][v_valid]
    bound_D = boundary[1:, 1:][v_valid]

    seg_A = seg_mask[:-1, :-1][v_valid]
    seg_B = seg_mask[:-1, 1:][v_valid]
    seg_C = seg_mask[1:, :-1][v_valid]
    seg_D = seg_mask[1:, 1:][v_valid]

    # Check depth continuity
    max_step = config['max_depth_step']
    d_AB = np.abs(z_A - z_B) < max_step
    d_CD = np.abs(z_C - z_D) < max_step
    d_AC = np.abs(z_A - z_C) < max_step
    d_BD = np.abs(z_B - z_D) < max_step
    d_AD = np.abs(z_A - z_D) < max_step
    d_BC = np.abs(z_B - z_C) < max_step

    # Check structural boundary crossing
    b_thresh = config['boundary_thresh']
    cross_AB = (bound_A >= b_thresh) & (bound_B >= b_thresh) & (seg_A != seg_B)
    cross_CD = (bound_C >= b_thresh) & (bound_D >= b_thresh) & (seg_C != seg_D)
    cross_AC = (bound_A >= b_thresh) & (bound_C >= b_thresh) & (seg_A != seg_C)
    cross_BD = (bound_B >= b_thresh) & (bound_D >= b_thresh) & (seg_B != seg_D)
    cross_AD = (bound_A >= b_thresh) & (bound_D >= b_thresh) & (seg_A != seg_D)
    cross_BC = (bound_B >= b_thresh) & (bound_C >= b_thresh) & (seg_B != seg_C)

    # 3D Edge length check
    pos_A = vertices[idx_A]
    pos_B = vertices[idx_B]
    pos_C = vertices[idx_C]
    pos_D = vertices[idx_D]

    len_AB = np.linalg.norm(pos_A - pos_B, axis=1) < config['max_edge_length']
    len_CD = np.linalg.norm(pos_C - pos_D, axis=1) < config['max_edge_length']
    len_AC = np.linalg.norm(pos_A - pos_C, axis=1) < config['max_edge_length']
    len_BD = np.linalg.norm(pos_B - pos_D, axis=1) < config['max_edge_length']
    len_AD = np.linalg.norm(pos_A - pos_D, axis=1) < config['max_edge_length']
    len_BC = np.linalg.norm(pos_B - pos_C, axis=1) < config['max_edge_length']

    edge_AB_valid = d_AB & ~cross_AB & len_AB
    edge_CD_valid = d_CD & ~cross_CD & len_CD
    edge_AC_valid = d_AC & ~cross_AC & len_AC
    edge_BD_valid = d_BD & ~cross_BD & len_BD
    edge_AD_valid = d_AD & ~cross_AD & len_AD
    edge_BC_valid = d_BC & ~cross_BC & len_BC

    diff_AD = np.abs(z_A - z_D)
    diff_BC = np.abs(z_B - z_C)

    # Choose diagonal with smaller depth jump if both are valid, or whichever diagonal is valid
    choose_AD = (edge_AD_valid & ~edge_BC_valid) | (edge_AD_valid & edge_BC_valid & (diff_AD <= diff_BC))
    choose_BC = (edge_BC_valid & ~edge_AD_valid) | (edge_AD_valid & edge_BC_valid & (diff_BC < diff_AD))

    triangles = []

    # Triangles with AD diagonal (counter-clockwise winding facing camera: ADB and ADC)
    valid_ABD = choose_AD & edge_AB_valid & edge_BD_valid & edge_AD_valid
    if np.any(valid_ABD):
        triangles.append(np.column_stack((idx_A[valid_ABD], idx_D[valid_ABD], idx_B[valid_ABD])))

    valid_ADC = choose_AD & edge_AD_valid & edge_CD_valid & edge_AC_valid
    if np.any(valid_ADC):
        triangles.append(np.column_stack((idx_A[valid_ADC], idx_C[valid_ADC], idx_D[valid_ADC])))

    # Triangles with BC diagonal (counter-clockwise winding facing camera: ACB and BDC)
    valid_ABC = choose_BC & edge_AB_valid & edge_BC_valid & edge_AC_valid
    if np.any(valid_ABC):
        triangles.append(np.column_stack((idx_A[valid_ABC], idx_C[valid_ABC], idx_B[valid_ABC])))

    valid_BDC = choose_BC & edge_BD_valid & edge_CD_valid & edge_BC_valid
    if np.any(valid_BDC):
        triangles.append(np.column_stack((idx_B[valid_BDC], idx_C[valid_BDC], idx_D[valid_BDC])))

    if len(triangles) == 0:
        print("[ERROR] No valid triangles formed after structure filtering!")
        sys.exit(1)

    faces = np.vstack(triangles)
    max_possible = np.sum(v_valid) * 2
    rejected_count = max_possible - len(faces)
    print(f"[STRUCTURE_MESH] Formed {len(faces)} triangles (rejected {rejected_count} boundary/discontinuity triangles).")

    # 4. Connected Component Analysis with Texture UV preservation
    visual = trimesh.visual.TextureVisuals(uv=uvs, image=img_rgb) if img_rgb is not None else trimesh.visual.TextureVisuals(uv=uvs)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    components = mesh.split(only_watertight=False)
    valid_components = [c for c in components if len(c.faces) >= config['min_component_faces']]

    if len(valid_components) == 0:
        valid_components = [max(components, key=lambda c: len(c.faces))]

    clean_mesh = trimesh.util.concatenate(valid_components)
    clean_mesh.metadata["name"] = "StructureAwareMesh"
    component_count = len(valid_components)

    # Recompute vertex normals and orient toward camera (+Z)
    clean_mesh.fix_normals()
    mean_nz = np.mean(clean_mesh.vertex_normals[:, 2])
    if mean_nz < 0:
        clean_mesh.invert()

    # Center and normalize mesh bounds
    bounding_box = clean_mesh.bounding_box.bounds
    min_b, max_b = bounding_box[0], bounding_box[1]
    center = (min_b + max_b) / 2.0
    clean_mesh.vertices -= center

    # Ensure UVs are properly set (either preserved or recomputed from ground truth projection)
    if not (hasattr(clean_mesh.visual, "uv") and clean_mesh.visual.uv is not None and len(clean_mesh.visual.uv) == len(clean_mesh.vertices)):
        orig_Z = np.maximum(1e-4, -clean_mesh.vertices[:, 2] - center[2])
        orig_X = clean_mesh.vertices[:, 0] + center[0]
        orig_Y = clean_mesh.vertices[:, 1] + center[1]

        final_u = orig_X * fx / orig_Z + cx
        final_v = -orig_Y * fy / orig_Z + cy

        final_u_tex = np.clip(final_u / float(w - 1), 0.0, 1.0)
        # In OpenGL/Trimesh convention, v=1 is top, v=0 is bottom; Trimesh flips Y when exporting to glTF
        final_v_tex = 1.0 - np.clip(final_v / float(h - 1), 0.0, 1.0)
        final_uvs = np.column_stack((final_u_tex, final_v_tex)).astype(np.float32)

        if img_rgb is not None:
            clean_mesh.visual = trimesh.visual.TextureVisuals(uv=final_uvs, image=img_rgb)
    elif img_rgb is not None and getattr(clean_mesh.visual, "material", None) is not None:
        clean_mesh.visual.material.image = img_rgb

    # Export GLB Model
    glb_path = os.path.join(models_dir, "model.glb")
    clean_mesh.export(glb_path)

    print(f"[STRUCTURE_MESH] Exported GLB model with RGB texture to {glb_path} ({os.path.getsize(glb_path)/1024:.1f} KB)")

    # 5. Metadata JSON
    metadata = {
        "model_type": "structure_aware_relative_surface_3d_mesh",
        "has_rgb_texture": img_rgb is not None,
        "vertex_count": int(len(clean_mesh.vertices)),
        "triangle_count": int(len(clean_mesh.faces)),
        "rejected_triangles": int(rejected_count),
        "component_count": int(component_count),
        "focal_length_px": float(focal_length_px),
        "bounding_box_size": [
            float(max_b[0] - min_b[0]),
            float(max_b[1] - min_b[1]),
            float(max_b[2] - min_b[2])
        ],
        "config": config
    }

    with open(os.path.join(metadata_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[STRUCTURE_MESH] Finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structure-Aware 3D Mesh Generator Stage")
    parser.add_argument("--surface", required=True, help="Path to final_relative_surface.npy")
    parser.add_argument("--boundary", required=True, help="Path to boundary_confidence.npy")
    parser.add_argument("--segmentation", required=True, help="Path to segmentation_mask.npy")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--image", required=False, help="Path to input RGB image for texture")
    parser.add_argument("--output", required=True, help="Output directory path")

    parser.add_argument("--max_depth_step", type=float, default=0.5, help="Max depth step across triangle edge (m)")
    parser.add_argument("--max_edge_length", type=float, default=5.0, help="Max 3D edge length (m)")
    parser.add_argument("--boundary_thresh", type=float, default=0.35, help="Boundary confidence threshold for splitting edges")
    parser.add_argument("--min_component_faces", type=int, default=30, help="Min connected faces to keep component")

    args = parser.parse_args()

    config = {
        "max_depth_step": args.max_depth_step,
        "max_edge_length": args.max_edge_length,
        "boundary_thresh": args.boundary_thresh,
        "min_component_faces": args.min_component_faces
    }

    generate_structure_aware_mesh(args.surface, args.boundary, args.segmentation, args.metadata, args.image, args.output, config)
