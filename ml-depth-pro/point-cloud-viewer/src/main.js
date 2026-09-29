import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import GUI from 'lil-gui';
import { loadDepthData, loadImageToCanvas } from './depthLoader.js';
import { setupCameraAndControls, resetCamera, fitModel, setTopView, setSideView, setPerspectiveView, setAerialView } from './camera.js';
import { createSceneHelpers } from './sceneHelpers.js';
import './style.css';

let scene, camera, controls, renderer;
let currentCitySceneGroup = null;
let groundMeshObject = null;
let buildingMeshObjects = [];
let selectedBuildingMesh = null;
let buildingsMetadataMap = new Map();

let depthData = null;
let currentGenId = "image3_output";
let raycaster, mouse;

const config = {
    mode: 'RGB Texture',
    zScale: 1.0,
    showGround: true,
    showBuildings: true,
    fitModel: () => fitModel(camera, controls, currentCitySceneGroup),
    resetCamera: () => resetCamera(camera, controls, currentCitySceneGroup),
    topView: () => setTopView(camera, controls, currentCitySceneGroup),
    sideView: () => setSideView(camera, controls, currentCitySceneGroup),
    perspectiveView: () => setPerspectiveView(camera, controls, currentCitySceneGroup),
    aerialView: () => setAerialView(camera, controls, currentCitySceneGroup)
};

const RECONSTRUCTION_DEBUG_MODES = [
    'RGB Texture',
    'Building Footprints',
    'Building Height',
    'Relative Depth',
    'Cleaned Depth',
    'Ground Surface',
    'Heatmap',
    'Solid Mesh',
    'Wireframe',
    'Boundary Map',
    'Segmentation',
    'Confidence',
    'Validity Mask',
    'Ground Only',
    'Buildings Only',
    'Normals'
];

async function init() {
    console.log("[1] Application initializing (Structured 3D City Architecture)...");
    const canvas = document.querySelector('#webgl-canvas');
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.setClearColor(0x0d0e12);

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d0e12);

    const camSetup = setupCameraAndControls(canvas);
    camera = camSetup.camera;
    controls = camSetup.controls;

    // Professional Lighting Setup
    const hemiLight = new THREE.HemisphereLight(0xf0f4f8, 0x22242a, 0.85);
    hemiLight.position.set(0, 50, 0);
    scene.add(hemiLight);

    const sunLight = new THREE.DirectionalLight(0xfffaed, 1.4);
    sunLight.position.set(-20, 40, 30);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 2048;
    sunLight.shadow.mapSize.height = 2048;
    scene.add(sunLight);

    const fillLight = new THREE.DirectionalLight(0xa5c0e0, 0.4);
    fillLight.position.set(20, 20, -20);
    scene.add(fillLight);

    // Raycaster for Building Selection
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    window.addEventListener('pointerdown', handleBuildingClick);
    window.addEventListener('resize', onWindowResize);

    setupRightPanelUI();
    setupUpload();
    checkBackendHealth();
    setInterval(checkBackendHealth, 15000);

    // Load initial 3D City Model if available
    loadStructuredCityModel("image3_output");

    // Render loop
    renderer.setAnimationLoop(() => {
        controls.update();
        renderer.render(scene, camera);
    });
}

async function safeFetchJson(url, options = {}) {
    let response;
    try {
        response = await fetch(url, options);
    } catch (networkErr) {
        throw new Error(`Cannot connect to reconstruction backend (port 5000): ${networkErr.message}. Make sure python app.py is running.`);
    }

    const rawText = await response.text();

    if (!response.ok) {
        let errMessage = `HTTP ${response.status} ${response.statusText}`;
        try {
            const errJson = JSON.parse(rawText);
            if (errJson.error || errJson.message) {
                errMessage = errJson.error || errJson.message;
            }
        } catch (_) {
            if (rawText.trim()) {
                errMessage += `: ${rawText.slice(0, 300)}`;
            }
        }
        throw new Error(errMessage);
    }

    if (!rawText.trim()) {
        throw new Error(`Backend returned an empty response (HTTP ${response.status}) from ${url}.`);
    }

    let data;
    try {
        data = JSON.parse(rawText);
    } catch (parseErr) {
        throw new Error(`Backend returned invalid JSON from ${url}: ${rawText.slice(0, 300)}`);
    }

    return data;
}

async function checkBackendHealth() {
    const statusText = document.getElementById('server-status-text');
    const statusDot = document.querySelector('.status-dot');
    try {
        const res = await fetch('/api/health');
        if (res.ok) {
            const raw = await res.text();
            if (raw.trim()) {
                const data = JSON.parse(raw);
                if (statusText) statusText.innerText = data.cuda_available ? "Server Online (CUDA)" : "Server Online (CPU)";
                if (statusDot) {
                    statusDot.classList.remove('offline');
                    statusDot.classList.add('online');
                }
                return true;
            }
        }
    } catch (_) {}
    if (statusText) statusText.innerText = "Backend Offline (Port 5000)";
    if (statusDot) {
        statusDot.classList.remove('online');
        statusDot.classList.add('offline');
    }
    return false;
}

async function loadStructuredCityModel(genId) {
    currentGenId = genId;
    console.log(`[3D_CITY] Loading structured 3D city for genId: ${genId}...`);
    
    const timestamp = new Date().getTime();
    const glbUrl = `/assets/${genId}/models/model.glb?t=${timestamp}`;
    const buildingsJsonUrl = `/assets/${genId}/buildings_3d.json?t=${timestamp}`;
    
    // Load metadata JSON safely
    try {
        const bRes = await fetch(buildingsJsonUrl);
        if (bRes.ok) {
            const raw = await bRes.text();
            if (raw.trim()) {
                const bData = JSON.parse(raw);
                buildingsMetadataMap.clear();
                if (bData.buildings) {
                    bData.buildings.forEach(b => {
                        buildingsMetadataMap.set(b.name || `Building_${b.building_id}`, b);
                        buildingsMetadataMap.set(`building_${b.building_id}`, b);
                    });
                }
                if (document.getElementById('info-bldgs-count')) {
                    document.getElementById('info-bldgs-count').innerText = bData.total_buildings_extruded || bData.buildings.length || 0;
                }
                if (document.getElementById('info-ground-size') && bData.ground_plane) {
                    document.getElementById('info-ground-size').innerText = `${bData.ground_plane.width_m}m × ${bData.ground_plane.depth_m}m`;
                }
            }
        }
    } catch (e) {
        console.warn("[3D_CITY] Metadata fetch note:", e);
    }

    // Load GLTF Model
    const loader = new GLTFLoader();
    loader.load(
        glbUrl,
        (gltf) => {
            if (currentCitySceneGroup) {
                scene.remove(currentCitySceneGroup);
            }

            currentCitySceneGroup = gltf.scene;
            groundMeshObject = null;
            buildingMeshObjects = [];

            let meshIdx = 0;
            currentCitySceneGroup.traverse((child) => {
                if (child.isMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                    
                    if (child.name === "GroundPlane" || meshIdx === 0) {
                        groundMeshObject = child;
                        child.name = "GroundPlane";
                        if (child.material) {
                            child.material.side = THREE.DoubleSide;
                        }
                    } else {
                        if (!child.name || child.name.startsWith("Mesh")) {
                            const bIdStr = String(buildingMeshObjects.length + 1).padStart(3, '0');
                            child.name = `Building_${bIdStr}`;
                        }
                        buildingMeshObjects.push(child);
                        
                        // Store original material (which contains rooftop RGB UV texture)
                        child.userData.originalMaterial = child.material;
                    }
                    meshIdx++;
                }
            });

            scene.add(currentCitySceneGroup);

            // Fit camera perspective
            const box = new THREE.Box3().setFromObject(currentCitySceneGroup);
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.z, 2.0);
            camera.position.set(maxDim * 0.8, maxDim * 0.8, maxDim * 0.9);
            controls.target.set(0, 0, 0);
            controls.update();

            // Unhide UI controls
            document.getElementById('right-panel')?.classList.remove('hidden');
            document.getElementById('bottom-thumbnail-bar')?.classList.remove('hidden');
            
            updateVisualization();
        },
        (progress) => {
            console.log(`[3D_CITY] GLTF loading progress: ${Math.round((progress.loaded / (progress.total || 1)) * 100)}%`);
        },
        (err) => {
            console.error("[3D_CITY] GLTF Load error:", err);
            const errBox = document.getElementById('loading-error');
            const errTxt = document.getElementById('loading-error-text');
            const spinner = document.getElementById('loading-spinner');
            if (errBox && errTxt) {
                errTxt.innerText = `Failed to load 3D scene (${glbUrl}): ${err.message || 'Model file not found or corrupted.'}`;
                errBox.classList.remove('hidden');
                if (spinner) spinner.classList.add('hidden');
            }
        }
    );
}

function handleBuildingClick(event) {
    if (event.button !== 0 || buildingMeshObjects.length === 0) return;

    const canvas = document.querySelector('#webgl-canvas');
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / canvas.clientWidth) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / canvas.clientHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(buildingMeshObjects);

    if (intersects.length > 0) {
        const hitBuilding = intersects[0].object;
        selectBuilding(hitBuilding);
    }
}

function selectBuilding(buildingMesh) {
    if (selectedBuildingMesh) {
        if (selectedBuildingMesh.userData.originalMaterial) {
            selectedBuildingMesh.material = selectedBuildingMesh.userData.originalMaterial;
        } else if (selectedBuildingMesh.material && selectedBuildingMesh.material.color) {
            selectedBuildingMesh.material.color.setHex(0x38bdf8);
        }
    }

    selectedBuildingMesh = buildingMesh;
    if (selectedBuildingMesh) {
        selectedBuildingMesh.material = new THREE.MeshStandardMaterial({
            color: 0xf59e0b,
            roughness: 0.2,
            metalness: 0.1,
            emissive: 0x331a00,
            side: THREE.DoubleSide
        });
    }

    const bName = buildingMesh.name || "Building_001";
    const meta = buildingsMetadataMap.get(bName) || buildingsMetadataMap.get(bName.toLowerCase());

    const card = document.getElementById('building-info-card');
    if (card) {
        card.classList.remove('hidden');
        document.getElementById('bldg-card-title').innerText = bName;
        document.getElementById('bldg-val-id').innerText = bName;
        
        if (meta) {
            document.getElementById('bldg-val-height').innerText = `${meta.height_m}m`;
            document.getElementById('bldg-val-dims').innerText = `${meta.dimensions_3d[0]}m × ${meta.dimensions_3d[1]}m × ${meta.dimensions_3d[2]}m`;
            document.getElementById('bldg-val-center').innerText = `${meta.center_3d[0]}, ${meta.center_3d[1]}, ${meta.center_3d[2]}`;
            document.getElementById('bldg-val-area').innerText = `${meta.area_px} px`;
            document.getElementById('bldg-val-conf').innerText = `${Math.round(meta.confidence * 100)}%`;
        } else {
            const box = new THREE.Box3().setFromObject(buildingMesh);
            const sz = box.getSize(new THREE.Vector3());
            const ctr = box.getCenter(new THREE.Vector3());
            document.getElementById('bldg-val-height').innerText = `${sz.y.toFixed(2)}m`;
            document.getElementById('bldg-val-dims').innerText = `${sz.x.toFixed(2)}m × ${sz.y.toFixed(2)}m × ${sz.z.toFixed(2)}m`;
            document.getElementById('bldg-val-center').innerText = `${ctr.x.toFixed(2)}, ${ctr.y.toFixed(2)}, ${ctr.z.toFixed(2)}`;
            document.getElementById('bldg-val-area').innerText = `Extruded Geometry`;
            document.getElementById('bldg-val-conf').innerText = `100%`;
        }
    }
}

function onWindowResize() {
    const canvas = document.querySelector('#webgl-canvas');
    if (!canvas) return;
    camera.aspect = canvas.clientWidth / canvas.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
}

function setupRightPanelUI() {
    // Mode Pills
    const modePills = document.querySelectorAll('.mode-pill');
    modePills.forEach(pill => {
        pill.addEventListener('click', () => {
            const m = pill.getAttribute('data-mode');
            config.mode = m;
            syncModeUI(m);
            updateVisualization();
        });
    });

    // Extended Debug Selector
    const debugSelect = document.getElementById('debug-mode-select');
    if (debugSelect) {
        debugSelect.addEventListener('change', (e) => {
            config.mode = e.target.value;
            syncModeUI(config.mode);
            updateVisualization();
        });
    }

    // Layer Toggles
    const toggleGroundBtn = document.getElementById('btn-toggle-ground');
    const toggleBldgsBtn = document.getElementById('btn-toggle-buildings');

    if (toggleGroundBtn) {
        toggleGroundBtn.addEventListener('click', () => {
            config.showGround = !config.showGround;
            if (groundMeshObject) groundMeshObject.visible = config.showGround;
            toggleGroundBtn.classList.toggle('active', config.showGround);
            toggleGroundBtn.style.background = config.showGround ? '#10b981' : '#4b5563';
            toggleGroundBtn.innerText = `Ground: ${config.showGround ? 'ON' : 'OFF'}`;
        });
    }

    if (toggleBldgsBtn) {
        toggleBldgsBtn.addEventListener('click', () => {
            config.showBuildings = !config.showBuildings;
            buildingMeshObjects.forEach(b => b.visible = config.showBuildings);
            toggleBldgsBtn.classList.toggle('active', config.showBuildings);
            toggleBldgsBtn.style.background = config.showBuildings ? '#3b82f6' : '#4b5563';
            toggleBldgsBtn.innerText = `Buildings: ${config.showBuildings ? 'ON' : 'OFF'}`;
        });
    }

    // View Controls
    document.getElementById('btn-reset')?.addEventListener('click', () => setCameraPreset('reset'));
    document.getElementById('btn-fit')?.addEventListener('click', () => setCameraPreset('fit'));
    document.getElementById('btn-top')?.addEventListener('click', () => setCameraPreset('top'));
    document.getElementById('btn-iso')?.addEventListener('click', () => setCameraPreset('iso'));
    document.getElementById('btn-persp')?.addEventListener('click', () => setCameraPreset('persp'));

    // Building Card Close
    document.getElementById('bldg-card-close')?.addEventListener('click', () => {
        document.getElementById('building-info-card')?.classList.add('hidden');
        if (selectedBuildingMesh) {
            if (selectedBuildingMesh.userData.originalMaterial) {
                selectedBuildingMesh.material = selectedBuildingMesh.userData.originalMaterial;
            } else if (selectedBuildingMesh.material && selectedBuildingMesh.material.color) {
                selectedBuildingMesh.material.color.setHex(0x38bdf8);
            }
            selectedBuildingMesh = null;
        }
    });

    // Height Exaggeration Slider
    const heightSlider = document.getElementById('height-slider');
    if (heightSlider) {
        heightSlider.addEventListener('input', (e) => {
            updateHeightExaggeration(e.target.value);
        });
    }
}

function setCameraPreset(preset) {
    if (!currentCitySceneGroup) return;
    const box = new THREE.Box3().setFromObject(currentCitySceneGroup);
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.z, 2.0);

    if (preset === 'reset' || preset === 'persp') {
        camera.position.set(maxDim * 0.9, maxDim * 0.9, maxDim * 0.9);
    } else if (preset === 'top') {
        camera.position.set(0, maxDim * 1.8, 0);
    } else if (preset === 'iso') {
        camera.position.set(maxDim * 1.2, maxDim * 1.2, maxDim * 1.2);
    } else if (preset === 'fit') {
        camera.position.set(0, maxDim * 0.8, maxDim * 1.2);
    }
    controls.target.set(0, 0, 0);
    controls.update();
}

function syncModeUI(modeVal) {
    const modePills = document.querySelectorAll('.mode-pill');
    modePills.forEach(pill => {
        const pm = pill.getAttribute('data-mode');
        if (pm === modeVal) {
            pill.classList.add('active');
        } else {
            pill.classList.remove('active');
        }
    });

    const debugSelect = document.getElementById('debug-mode-select');
    if (debugSelect && debugSelect.value !== modeVal) {
        debugSelect.value = modeVal;
    }
}

function updateHeightExaggeration(newScale) {
    config.zScale = parseFloat(newScale);
    const heightDisplay = document.getElementById('height-val-display');
    if (heightDisplay) heightDisplay.innerText = `${config.zScale.toFixed(1)}×`;

    // Scale building extrusion height ONLY (ground plane remains flat at Y=0)
    buildingMeshObjects.forEach(b => {
        b.scale.y = config.zScale;
    });
}

function updateVisualization() {
    if (!currentCitySceneGroup) return;

    const textureLoader = new THREE.TextureLoader();
    const timestamp = new Date().getTime();

    if (config.mode === 'Ground Only') {
        if (groundMeshObject) groundMeshObject.visible = true;
        buildingMeshObjects.forEach(b => b.visible = false);
        return;
    } else if (config.mode === 'Buildings Only') {
        if (groundMeshObject) groundMeshObject.visible = false;
        buildingMeshObjects.forEach(b => b.visible = true);
        return;
    }

    if (groundMeshObject) groundMeshObject.visible = config.showGround;
    buildingMeshObjects.forEach(b => b.visible = config.showBuildings);

    const texMap = {
        'RGB Texture': 'texture_rgb.png',
        'Building Footprints': 'texture_building_footprints.png',
        'Building Height': 'building_height_map.png',
        'Heatmap': 'heatmap.png',
        'Cleaned Depth': 'depth_cleaned.png',
        'Ground Surface': 'ground_surface.png',
        'Boundary Map': 'boundary_combined.png',
        'Segmentation': 'segmentation_mask.png',
        'Confidence': 'building_confidence.png',
        'Validity Mask': 'validity_mask.png'
    };

    if (texMap[config.mode] && groundMeshObject) {
        const texUrl = `/assets/${currentGenId}/${texMap[config.mode]}?t=${timestamp}`;
        const tex = textureLoader.load(texUrl);
        tex.colorSpace = THREE.SRGBColorSpace;
        groundMeshObject.material = new THREE.MeshStandardMaterial({
            map: tex,
            side: THREE.DoubleSide,
            roughness: 0.6
        });
    }

    buildingMeshObjects.forEach(b => {
        if (config.mode === 'Wireframe') {
            b.material = new THREE.MeshBasicMaterial({
                color: 0x00ffcc,
                wireframe: true
            });
        } else if (config.mode === 'Normals') {
            b.material = new THREE.MeshNormalMaterial();
        } else if (config.mode === 'Building Height') {
            b.material = new THREE.MeshStandardMaterial({
                color: 0xf59e0b,
                roughness: 0.3,
                metalness: 0.1
            });
        } else {
            b.material = b.userData.originalMaterial || new THREE.MeshStandardMaterial({
                color: 0x38bdf8,
                roughness: 0.35,
                metalness: 0.1,
                side: THREE.DoubleSide
            });
        }
    });
}

function setupUpload() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');

    uploadArea?.addEventListener('click', () => fileInput.click());

    uploadArea?.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#34d399';
    });

    uploadArea?.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '#10b981';
    });

    uploadArea?.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#10b981';
        if (e.dataTransfer.files.length) {
            handleUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput?.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleUpload(e.target.files[0]);
        }
    });
}

async function handleUpload(file) {
    if (!file) return;

    const loading = document.getElementById('loading');
    const loadingText = document.getElementById('loading-text');
    const loadingSubtext = document.getElementById('loading-subtext');
    const stepBadge = document.getElementById('loading-step-badge');
    const progressBar = document.getElementById('loading-progress-bar');
    const loadingSpinner = document.getElementById('loading-spinner');
    const loadingError = document.getElementById('loading-error');
    const errorText = document.getElementById('loading-error-text');
    const retryBtn = document.getElementById('loading-retry-btn');
    
    document.getElementById('upload-overlay').classList.add('hidden');
    loading.classList.remove('hidden');
    if (loadingSpinner) loadingSpinner.classList.remove('hidden');
    if (loadingError) loadingError.classList.add('hidden');

    function updateProgress(stepIdx, stepName, pct, subText) {
        if (stepBadge) stepBadge.innerText = `STEP ${stepIdx}/8`;
        if (loadingText) loadingText.innerText = stepName;
        if (progressBar) progressBar.style.width = `${pct}%`;
        if (loadingSubtext) loadingSubtext.innerText = `${pct}% ${subText || ''}`;
        
        if (document.getElementById('top-progress-status')) document.getElementById('top-progress-status').innerText = stepName;
        if (document.getElementById('top-progress-pct')) document.getElementById('top-progress-pct').innerText = `${pct}%`;
    }

    if (retryBtn) {
        retryBtn.onclick = () => {
            loading.classList.add('hidden');
            if (loadingError) loadingError.classList.add('hidden');
            document.getElementById('upload-overlay').classList.remove('hidden');
        };
    }

    updateProgress(1, "Loading image", 10, "Uploading payload...");

    const formData = new FormData();
    formData.append('image', file);

    try {
        const data = await safeFetchJson('/api/infer', {
            method: 'POST',
            body: formData
        });

        if (data.error) throw new Error(data.error);

        const genId = data.generation_id || data.session_id || data.job_id;
        if (!genId) throw new Error("Backend response did not include a valid session / generation ID.");
        
        let done = false;
        let pollFailures = 0;
        while (!done) {
            await new Promise(r => setTimeout(r, 800));
            let progData;
            try {
                progData = await safeFetchJson(`/api/infer/progress?id=${encodeURIComponent(genId)}`);
                pollFailures = 0;
            } catch (pErr) {
                pollFailures++;
                if (pollFailures > 4) {
                    throw new Error(`Connection lost while tracking reconstruction: ${pErr.message}`);
                }
                continue;
            }
            
            if (progData.status === 'running') {
                const sIdx = progData.step_index || 2;
                const sName = progData.step_name || progData.stage || progData.message || "Processing...";
                const pct = progData.progress || 20;
                const detail = progData.message || "";
                updateProgress(sIdx, sName, pct, detail);
            } else if (progData.status === 'done' || progData.status === 'completed') {
                done = true;
            } else if (progData.status === 'error') {
                const stageStr = progData.step_name || progData.stage ? `[${progData.step_name || progData.stage}] ` : "";
                const errStr = progData.error || progData.message || "Reconstruction pipeline failure occurred.";
                const causeStr = progData.cause ? ` — ${progData.cause}` : "";
                throw new Error(`${stageStr}${errStr}${causeStr}`);
            }
        }

        updateProgress(8, "Loading 3D City Model", 95, "Rendering GLTF buildings...");
        
        // Load the new 3D City Model
        await loadStructuredCityModel(genId);

        // Top bar updates
        if (document.getElementById('top-img-name')) document.getElementById('top-img-name').innerText = file.name || "image.jpg";
        if (document.getElementById('info-src-img')) document.getElementById('info-src-img').innerText = file.name || "image.jpg";

        // Update Thumbnails
        const heatmapUrl = `/assets/${genId}/depth.png`;
        const rgbUrl = `/assets/${genId}/image.jpg`;
        const thumbHeatmapImg = document.getElementById('thumb-img-heatmap');
        const thumbDepthImg = document.getElementById('thumb-img-depth');
        const thumbRgbImg = document.getElementById('thumb-img-rgb');
        if (thumbHeatmapImg) thumbHeatmapImg.src = heatmapUrl;
        if (thumbDepthImg) thumbDepthImg.src = `/assets/${genId}/depth_cleaned.png` || heatmapUrl;
        if (thumbRgbImg) thumbRgbImg.src = rgbUrl;

        updateProgress(8, "Ready", 100, "Complete");
        await new Promise(r => setTimeout(r, 400));
        loading.classList.add('hidden');

    } catch (err) {
        console.error("Reconstruction error:", err);
        if (loadingSpinner) loadingSpinner.classList.add('hidden');
        if (loadingError) {
            loadingError.classList.remove('hidden');
            if (errorText) errorText.innerText = err.message || "An unexpected error occurred.";
        } else {
            alert(`Error: ${err.message}`);
            loading.classList.add('hidden');
            document.getElementById('upload-overlay').classList.remove('hidden');
        }
    }
}

init();

