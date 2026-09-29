import os
import sys
import subprocess
import json
import time
import zipfile
import shutil
import re
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'), static_folder=os.path.join(BASE_DIR, 'static'))
CORS(app)

app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload limit

# Resolve correct Python executable for subprocesses
venv_python = os.path.join(BASE_DIR, '.venv', 'Scripts', 'python.exe')
python_exe = venv_python if os.path.exists(venv_python) else sys.executable

print(f"[SERVER] Starting DepthWizard Backend...")
print(f"[SERVER] Base Directory: {BASE_DIR}")
print(f"[SERVER] Output Folder: {OUTPUT_FOLDER}")
print(f"[SERVER] Python Engine: {python_exe}")

# Check CUDA availability at startup
CUDA_AVAILABLE = False
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    print(f"[SERVER] PyTorch CUDA Status: {CUDA_AVAILABLE} ({torch.cuda.get_device_name(0) if CUDA_AVAILABLE else 'CPU'})")
except Exception as e:
    print(f"[SERVER] PyTorch CUDA check note: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "success": True,
        "status": "online",
        "service": "DepthWizard 3D Reconstruction Platform",
        "version": "1.0.0",
        "cuda_available": CUDA_AVAILABLE,
        "output_folder": OUTPUT_FOLDER
    })

@app.route('/api/<path:subpath>', methods=['OPTIONS'])
@app.route('/api/', methods=['OPTIONS'])
def handle_options_preflight(subpath=''):
    response = jsonify({"success": True, "message": "Preflight OK"})
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    return response, 200

@app.route('/api/analyze', methods=['POST'])
@app.route('/api/infer', methods=['POST'])
@app.route('/api/3d/generate', methods=['POST'])
def start_reconstruction():
    try:
        preset_name = request.form.get('preset') or request.args.get('preset')
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")

        input_path = None
        original_filename = "image.jpg"

        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            raw_filename = os.path.basename(file.filename)
            ext = os.path.splitext(raw_filename)[1].lower()
            allowed_exts = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp']
            if ext not in allowed_exts:
                return jsonify({
                    "success": False,
                    "error": f"Unsupported file extension '{ext}'. Allowed image formats: {', '.join(allowed_exts)}"
                }), 400

            # Sanitize filename: replace spaces and weird characters
            safe_name = re.sub(r'[^a-zA-Z0-9_\.-]', '_', raw_filename)
            base_name = os.path.splitext(safe_name)[0]
            if not base_name:
                base_name = "image"
            
            session_id = f"{base_name}_{timestamp_str}"
            session_dir = os.path.join(OUTPUT_FOLDER, session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            input_path = os.path.join(session_dir, safe_name)
            file.save(input_path)
            original_filename = safe_name

        elif preset_name:
            clean_preset = re.sub(r'[^a-zA-Z0-9_\.-]', '_', preset_name)
            session_id = f"{clean_preset}_{timestamp_str}"
            session_dir = os.path.join(OUTPUT_FOLDER, session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            candidates = [
                preset_name,
                os.path.join(BASE_DIR, preset_name),
                os.path.join(BASE_DIR, f"{preset_name}.jpg"),
                os.path.join(BASE_DIR, f"{preset_name}.png"),
                os.path.join(BASE_DIR, "image2.jpg"),
                os.path.join(BASE_DIR, "satellite.jpg"),
                os.path.join(BASE_DIR, "example.jpg")
            ]
            for cand in candidates:
                if os.path.exists(cand) and os.path.isfile(cand):
                    original_filename = os.path.basename(cand)
                    input_path = os.path.join(session_dir, original_filename)
                    shutil.copy2(cand, input_path)
                    break
        else:
            # Fallback default image
            session_id = f"default_{timestamp_str}"
            session_dir = os.path.join(OUTPUT_FOLDER, session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            for default_img in ["image2.jpg", "satellite.jpg", "example.jpg"]:
                cand = os.path.join(BASE_DIR, default_img)
                if os.path.exists(cand):
                    original_filename = default_img
                    input_path = os.path.join(session_dir, original_filename)
                    shutil.copy2(cand, input_path)
                    break

        if not input_path or not os.path.exists(input_path):
            return jsonify({
                "success": False,
                "error": "No image file provided or the uploaded payload was empty."
            }), 400

        progress_file = os.path.join(session_dir, "progress.json")
        log_file_path = os.path.join(session_dir, "pipeline.log")

        # Initialize progress JSON
        initial_progress = {
            "status": "running",
            "step_index": 1,
            "step_name": "Loading image",
            "stage": "Step 1/8: Loading image",
            "progress": 10,
            "message": "Uploading payload and preparing reconstruction pipeline...",
            "session_id": session_id,
            "generation_id": session_id,
            "timestamp": time.time()
        }
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(initial_progress, f, indent=2)

        # Launch export_depth_data.py in background subprocess
        pipeline_script = os.path.join(BASE_DIR, "src", "export_depth_data.py")
        cmd = [
            python_exe,
            pipeline_script,
            input_path,
            session_dir,
            progress_file
        ]

        print(f"[RECONSTRUCTION] Launching job '{session_id}' with command:")
        print(f"               {' '.join(cmd)}")

        log_file = open(log_file_path, "w", encoding="utf-8")
        subprocess.Popen(cmd, stdout=log_file, stderr=log_file, cwd=BASE_DIR)

        return jsonify({
            "success": True,
            "status": "started",
            "job_id": session_id,
            "session_id": session_id,
            "generation_id": session_id,
            "message": "Reconstruction pipeline started successfully."
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Failed to start reconstruction: {str(e)}"
        }), 500

@app.route('/api/progress', methods=['GET'])
@app.route('/api/infer/progress', methods=['GET'])
@app.route('/api/3d/progress', methods=['GET'])
def get_progress():
    session_id = request.args.get('session_id') or request.args.get('id') or request.args.get('gen_id')
    if not session_id:
        return jsonify({
            "success": False,
            "status": "error",
            "error": "Missing session_id parameter."
        }), 400

    session_dir = os.path.join(OUTPUT_FOLDER, session_id)
    if not os.path.exists(session_dir):
        return jsonify({
            "success": False,
            "status": "error",
            "error": f"Session folder '{session_id}' not found."
        }), 404

    progress_file = os.path.join(session_dir, "progress.json")
    log_file_path = os.path.join(session_dir, "pipeline.log")

    if not os.path.exists(progress_file):
        return jsonify({
            "success": True,
            "status": "running",
            "step_index": 1,
            "step_name": "Loading image",
            "stage": "Step 1/8: Loading image",
            "progress": 10,
            "message": "Starting processing...",
            "session_id": session_id,
            "generation_id": session_id
        })

    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check if pipeline.log has an unhandled traceback if status is not already completed or error
        if data.get("status") == "running" and os.path.exists(log_file_path):
            try:
                with open(log_file_path, "r", encoding="utf-8", errors="replace") as lf:
                    log_tail = lf.read()
                    if "Traceback (most recent call last):" in log_tail and "Error" in log_tail:
                        err_lines = [l for l in log_tail.splitlines() if l.strip()]
                        last_err = err_lines[-1] if err_lines else "Unknown pipeline crash"
                        data["status"] = "error"
                        data["error"] = f"Fatal error in subprocess: {last_err}"
            except Exception:
                pass

        data["success"] = (data.get("status") != "error")
        return jsonify(data)
    except Exception as e:
        return jsonify({
            "success": True,
            "status": "running",
            "step_index": 1,
            "step_name": "Loading image",
            "progress": 15,
            "message": f"Reading progress ({str(e)})...",
            "session_id": session_id,
            "generation_id": session_id
        })

@app.route('/api/results', methods=['GET'])
@app.route('/api/details', methods=['GET'])
def get_results():
    session_id = request.args.get('session_id') or request.args.get('id') or request.args.get('gen_id')
    if not session_id:
        return jsonify({
            "success": False,
            "status": "error",
            "error": "Missing session_id parameter."
        }), 400

    session_dir = os.path.join(OUTPUT_FOLDER, session_id)
    if not os.path.exists(session_dir):
        return jsonify({
            "success": False,
            "status": "error",
            "error": f"Session directory '{session_id}' not found."
        }), 404

    # Search for metadata in data/ or root
    meta_candidates = [
        os.path.join(session_dir, "data", "metadata.json"),
        os.path.join(session_dir, "metadata.json"),
        os.path.join(session_dir, "analysis_report.json")
    ]
    target_meta_file = None
    for cand in meta_candidates:
        if os.path.exists(cand):
            target_meta_file = cand
            break

    if not target_meta_file:
        return jsonify({
            "success": False,
            "status": "error",
            "error": f"Report metadata file not found for session '{session_id}'."
        }), 404

    try:
        with open(target_meta_file, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)

        # Buildings JSON
        buildings_data = {"buildings": []}
        bldg_candidates = [
            os.path.join(session_dir, "data", "buildings.json"),
            os.path.join(session_dir, "buildings.json")
        ]
        for bc in bldg_candidates:
            if os.path.exists(bc):
                try:
                    with open(bc, 'r', encoding='utf-8') as bf:
                        buildings_data = json.load(bf)
                    break
                except Exception:
                    pass

        # Statistics JSON
        stats_data = {}
        stats_candidates = [
            os.path.join(session_dir, "data", "statistics.json"),
            os.path.join(session_dir, "statistics.json")
        ]
        for sc in stats_candidates:
            if os.path.exists(sc):
                try:
                    with open(sc, 'r', encoding='utf-8') as sf:
                        stats_data = json.load(sf)
                    break
                except Exception:
                    pass

        ts = int(time.time())
        map_urls = {}

        # 12 Standard Map Filenames
        standard_maps = {
            "rgb": ["maps/rgb.png", "rgb.png", "image.jpg", "original_image.jpg"],
            "depth": ["maps/depth_map.png", "depth_map.png", "depth_heatmap.png", "depth.png"],
            "rdsm": ["maps/rdsm.png", "rdsm.png", "height_above_ground.npy"],
            "elevation_heatmap": ["maps/elevation_heatmap.png", "elevation_heatmap.png", "heatmap.png"],
            "building_detection": ["maps/building_detection.png", "building_detection.png", "building_mask.png"],
            "building_height": ["maps/building_height_map.png", "building_height_map.png"],
            "edges": ["maps/edges.png", "edges.png"],
            "building_boundaries": ["maps/building_boundaries.png", "building_boundaries.png"],
            "slope": ["maps/slope_map.png", "slope_map.png"],
            "terrain_relief": ["maps/terrain_relief.png", "terrain_relief.png"],
            "semantic": ["maps/semantic_map.png", "semantic_map.png", "segmentation_mask.png"],
            "error": ["maps/error_map.png", "error_map.png", "confidence_map.png"]
        }

        # Resolve URLs for each map
        for key, candidates in standard_maps.items():
            for cand in candidates:
                fpath = os.path.join(session_dir, cand)
                if os.path.exists(fpath):
                    map_urls[key] = f"/assets/{session_id}/{cand.replace(os.sep, '/')}?t={ts}"
                    break

        return jsonify({
            "success": True,
            "status": "completed",
            "session_id": session_id,
            "generation_id": session_id,
            "metadata": meta_data,
            "buildings": buildings_data,
            "statistics": stats_data,
            "report": meta_data,
            "map_urls": map_urls
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "status": "error",
            "error": f"Failed reading analysis data: {str(e)}"
        }), 500

@app.route('/api/download_all', methods=['GET'])
def download_all():
    session_id = request.args.get('session_id') or request.args.get('id') or request.args.get('gen_id')
    if not session_id:
        return jsonify({"success": False, "error": "Missing session_id parameter."}), 400

    session_dir = os.path.join(OUTPUT_FOLDER, session_id)
    if not os.path.exists(session_dir):
        return jsonify({"success": False, "error": f"Session folder '{session_id}' not found."}), 404

    zip_filename = f"{session_id}_results.zip"
    zip_filepath = os.path.join(session_dir, zip_filename)

    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(session_dir):
            for file in files:
                if file.endswith('.zip') or file.endswith('.log'):
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, session_dir)
                zipf.write(full_path, arcname=rel_path)

    return send_file(zip_filepath, as_attachment=True, download_name=zip_filename)

# Serve both /output/ and /assets/ routes with correct content types
@app.route('/output/<path:filename>')
@app.route('/assets/<path:filename>')
def serve_output_file(filename):
    mimetypes_custom = {
        '.glb': 'model/gltf-binary',
        '.gltf': 'model/gltf+json',
        '.json': 'application/json',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.tif': 'image/tiff',
        '.tiff': 'image/tiff',
        '.npy': 'application/octet-stream'
    }
    ext = os.path.splitext(filename)[1].lower()
    mimetype = mimetypes_custom.get(ext)
    return send_from_directory(OUTPUT_FOLDER, filename, mimetype=mimetype)

@app.after_request
def after_request_callback(response):
    if request.path.startswith('/api/'):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        if 'application/json' not in response.headers.get('Content-Type', ''):
            response.headers['Content-Type'] = 'application/json'
    return response

@app.errorhandler(404)
def not_found_handler(e):
    if request.path.startswith('/api/') or 'application/json' in request.headers.get('Accept', ''):
        return jsonify({"success": False, "error": f"API endpoint not found: {request.path}"}), 404
    return jsonify({"success": False, "error": f"Resource not found: {request.path}"}), 404

@app.errorhandler(500)
def internal_error_handler(e):
    if request.path.startswith('/api/') or 'application/json' in request.headers.get('Accept', ''):
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500
    return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    print("[SERVER] Starting Flask on 0.0.0.0:5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
