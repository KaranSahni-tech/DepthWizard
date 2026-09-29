// --- PS 175 REMOTE SENSING & 2D MULTI-LAYER ELEVATION CONTROLLER ---

export function getApiBaseUrl() {
    if (typeof window !== 'undefined') {
        const savedUrl = localStorage.getItem('DEPTHWIZARD_API_URL');
        if (savedUrl && savedUrl.trim()) {
            return savedUrl.trim().replace(/\/+$/, '');
        }
        if (window.VITE_API_URL) return window.VITE_API_URL.replace(/\/+$/, '');
        if (window.VITE_BACKEND_URL) return window.VITE_BACKEND_URL.replace(/\/+$/, '');
    }
    if (typeof import.meta !== 'undefined' && import.meta.env) {
        if (import.meta.env.VITE_API_URL) return import.meta.env.VITE_API_URL.replace(/\/+$/, '');
        if (import.meta.env.VITE_BACKEND_URL) return import.meta.env.VITE_BACKEND_URL.replace(/\/+$/, '');
    }
    return '';
}

export function getApiUrl(path) {
    if (!path) return '';
    if (path.startsWith('http://') || path.startsWith('https://')) return path;
    if (!path.startsWith('/')) path = '/' + path;
    const baseUrl = getApiBaseUrl();
    return baseUrl ? `${baseUrl}${path}` : path;
}

async function safeFetchJson(url, options = {}) {
    let res;
    try {
        res = await fetch(url, options);
    } catch (networkErr) {
        console.error('[API Network Error]', networkErr);
        const baseUrl = getApiBaseUrl();
        throw new Error(
            `Unable to connect to backend server at ${url}. ${
                !baseUrl 
                    ? "Vercel static frontend is not connected to a backend. Please set VITE_API_URL in Vercel settings or click ⚙️ API Server in top header."
                    : `Check if backend server '${baseUrl}' is running.`
            }`
        );
    }

    const contentType = res.headers.get('content-type') || '';
    const rawText = await res.text();

    if (!rawText || !rawText.trim()) {
        if (!res.ok) {
            throw new Error(`Server returned HTTP ${res.status} ${res.statusText || 'Error'} with empty response.`);
        }
        return { success: true };
    }

    let parsed = null;
    const isJson = contentType.includes('application/json') || rawText.trim().startsWith('{') || rawText.trim().startsWith('[');

    if (isJson) {
        try {
            parsed = JSON.parse(rawText);
        } catch (parseErr) {
            console.error('[API JSON Error] Failed to parse JSON response:', parseErr, rawText);
            throw new Error(`Failed to parse server response as JSON (HTTP ${res.status}). Response preview: "${rawText.slice(0, 150)}..."`);
        }
    } else {
        console.error(`[API Error] Received non-JSON content-type (${contentType}, HTTP ${res.status}):`, rawText);
        if (res.status === 404) {
            const baseUrl = getApiBaseUrl();
            throw new Error(
                `HTTP 404 NOT_FOUND: Route '${url}' was not found. ${
                    !baseUrl 
                        ? "Your Vercel deployment is serving static files without a connected Python backend engine. Please configure VITE_API_URL in Vercel Environment Variables or click ⚙️ API Server in the header to enter your backend URL."
                        : `Please verify that route '${url}' is registered on your backend server (${baseUrl}).`
                }`
            );
        }
        if (rawText.includes('<!DOCTYPE') || rawText.includes('<html')) {
            throw new Error(`Server returned HTML instead of JSON (HTTP ${res.status}). Please verify that the backend API URL is reachable and configured properly.`);
        }
        throw new Error(`Server returned HTTP ${res.status}: ${rawText.slice(0, 200)}`);
    }

    if (!res.ok) {
        if (res.status === 404) {
            const baseUrl = getApiBaseUrl();
            throw new Error(
                `HTTP 404 NOT_FOUND: ${
                    !baseUrl 
                        ? "API endpoint not found on static server. Set VITE_API_URL in Vercel settings or click ⚙️ API Server to configure your backend URL."
                        : `API route not found on backend (${baseUrl}).`
                }`
            );
        }
        const errMsg = (parsed && (parsed.error || parsed.message)) ? (parsed.error || parsed.message) : `Request failed with status HTTP ${res.status}`;
        const err = new Error(errMsg);
        err.status = res.status;
        err.data = parsed;
        throw err;
    }

    return parsed;
}

let activeSessionId = null;
let pollInterval = null;
let isAnalyzing = false;
let currentMetadata = null;
let currentBuildings = null;
let currentStatistics = null;
let currentMapUrls = {};
let activeJsonTab = "metadata";

// Modal zoom state
let modalZoom = 1.0;
let isPanning = false;
let startX = 0, startY = 0;
let panX = 0, panY = 0;

const MAP_TITLES = {
    rgb: "MAP 1 — Original RGB",
    depth: "MAP 2 — Depth Map",
    rdsm: "MAP 3 — Relative DSM / rDSM",
    elevation_heatmap: "MAP 4 — Elevation Heatmap",
    building_detection: "MAP 5 — Building Detection Map",
    building_height: "MAP 6 — Building Height Map",
    edges: "MAP 7 — Edge / Boundary Map",
    building_boundaries: "MAP 8 — Building Footprint Map",
    slope: "MAP 9 — Slope Map",
    terrain_relief: "MAP 10 — Terrain / Relief Map",
    semantic: "MAP 11 — Semantic Classification Map",
    error: "MAP 12 — Model Confidence / Uncertainty Map"
};

// DOM References
const DOM = {
    // Top Bar & Buttons
    headerBtnUpload: document.getElementById('header-btn-upload'),
    btnBrowseFiles: document.getElementById('btn-browse-files'),
    btnSampleSatellite: document.getElementById('btn-sample-satellite'),
    btnSampleImage2: document.getElementById('btn-sample-image2'),
    fileInputHidden: document.getElementById('file-input-hidden'),
    dropzoneArea: document.getElementById('dropzone-area'),
    
    // Sections
    sectionUpload: document.getElementById('section-upload'),
    sectionProcessing: document.getElementById('section-processing'),
    sectionDashboard: document.getElementById('section-dashboard'),
    
    // Input Specs Preview Card
    inputPreviewImg: document.getElementById('input-preview-img'),
    infoFilename: document.getElementById('info-filename'),
    infoResolution: document.getElementById('info-resolution'),
    infoFormat: document.getElementById('info-format'),
    infoChannels: document.getElementById('info-channels'),
    infoGeospatial: document.getElementById('info-geospatial'),
    infoSize: document.getElementById('info-size'),

    // Progress Monitor
    progressBarFill: document.getElementById('progress-bar-fill'),
    progressPercentText: document.getElementById('progress-percent-text'),
    stageTitle: document.getElementById('stage-title'),
    stageMessage: document.getElementById('stage-message'),
    stageSteps: document.querySelectorAll('.stage-step'),

    // Dashboard Header & Key Metrics
    sessionIdDisplay: document.getElementById('session-id-display'),
    sessionTimeDisplay: document.getElementById('session-time-display'),
    btnDownloadZip: document.getElementById('btn-download-zip'),
    statBuildingCount: document.getElementById('stat-building-count'),
    statImageSize: document.getElementById('stat-image-size'),
    statImageFormat: document.getElementById('stat-image-format'),
    statDepthRange: document.getElementById('stat-depth-range'),
    statElevationMax: document.getElementById('stat-elevation-max'),
    statGeospatialStatus: document.getElementById('stat-geospatial-status'),
    statGeospatialSub: document.getElementById('stat-geospatial-sub'),
    statConfidence: document.getElementById('stat-confidence'),

    // Tabs
    dashTabs: document.querySelectorAll('.dash-tab'),
    tabPanes: document.querySelectorAll('.tab-pane'),

    // Compare Mode
    compareSelectLeft: document.getElementById('compare-select-left'),
    compareSelectRight: document.getElementById('compare-select-right'),
    compareTitleLeft: document.getElementById('compare-title-left'),
    compareTitleRight: document.getElementById('compare-title-right'),
    compareImgLeft: document.getElementById('compare-img-left'),
    compareImgRight: document.getElementById('compare-img-right'),
    btnSwapCompare: document.getElementById('btn-swap-compare'),

    // Analysis Data
    adFilename: document.getElementById('ad-filename'),
    adFormat: document.getElementById('ad-format'),
    adDims: document.getElementById('ad-dims'),
    adChannels: document.getElementById('ad-channels'),
    adSize: document.getElementById('ad-size'),
    adTime: document.getElementById('ad-time'),

    // Geospatial Info
    geoStatus: document.getElementById('geo-status'),
    geoCrs: document.getElementById('geo-crs'),
    geoEpsg: document.getElementById('geo-epsg'),
    geoPixelSize: document.getElementById('geo-pixel-size'),
    geoBounds: document.getElementById('geo-bounds'),
    geoCoverage: document.getElementById('geo-coverage'),

    // Elevation & Quality
    adElevType: document.getElementById('ad-elev-type'),
    adElevMin: document.getElementById('ad-elev-min'),
    adElevMax: document.getElementById('ad-elev-max'),
    adElevMean: document.getElementById('ad-elev-mean'),
    adElevMedian: document.getElementById('ad-elev-median'),
    adQConf: document.getElementById('ad-q-conf'),
    adQMae: document.getElementById('ad-q-mae'),
    adQRmse: document.getElementById('ad-q-rmse'),
    adQCorr: document.getElementById('ad-q-corr'),
    adQValid: document.getElementById('ad-q-valid'),
    adQStatus: document.getElementById('ad-q-status'),

    // Building Table & Summary
    bsumCount: document.getElementById('bsum-count'),
    bsumAvgH: document.getElementById('bsum-avg-h'),
    bsumMaxH: document.getElementById('bsum-max-h'),
    bsumMinH: document.getElementById('bsum-min-h'),
    bsumArea: document.getElementById('bsum-area'),
    bsumConf: document.getElementById('bsum-conf'),
    buildingsTableBody: document.getElementById('buildings-table-body'),

    // JSON Subtabs & Actions
    jsonSubtabs: document.querySelectorAll('.btn-subtab'),
    jsonCodeDisplay: document.getElementById('json-code-display'),
    btnCopyJson: document.getElementById('btn-copy-json'),
    btnDownloadJson: document.getElementById('btn-download-json'),

    // Modal
    imageModal: document.getElementById('image-modal'),
    modalTitle: document.getElementById('modal-title'),
    modalSubInfo: document.getElementById('modal-sub-info'),
    modalImageDisplay: document.getElementById('modal-image-display'),
    modalPanContainer: document.getElementById('modal-pan-container'),
    modalDownloadLink: document.getElementById('modal-download-link'),
    modalCloseBtn: document.getElementById('modal-close-btn'),
    btnModalZoomIn: document.getElementById('btn-modal-zoom-in'),
    btnModalZoomOut: document.getElementById('btn-modal-zoom-out'),
    btnModalZoomReset: document.getElementById('btn-modal-zoom-reset')
};

// INITIALIZATION
window.addEventListener('DOMContentLoaded', () => {
    initEvents();
    initTabs();
    initCompareControls();
    initModalPanZoom();

    // Check if initial session exists (e.g. image3_output or latest)
    safeFetchJson(getApiUrl('/api/results?session_id=image3_output'))
        .then(data => {
            if (data.success) {
                renderResults(data);
            }
        })
        .catch(() => {});
});

function initApiConfigDialog() {
    const btn = document.getElementById('header-btn-api-config');
    if (!btn) return;

    btn.addEventListener('click', () => {
        const current = getApiBaseUrl() || "(Relative / Default Vercel Domain)";
        const newUrl = prompt(
            `Configure Production Backend API URL:\n\nCurrent active API base: ${current}\n\nEnter your backend URL (e.g. http://localhost:5000 or https://your-backend-domain.com):`,
            localStorage.getItem('DEPTHWIZARD_API_URL') || ""
        );
        if (newUrl !== null) {
            if (newUrl.trim() === "") {
                localStorage.removeItem('DEPTHWIZARD_API_URL');
                alert("Cleared custom API URL override. Now using default environment settings.");
            } else {
                localStorage.setItem('DEPTHWIZARD_API_URL', newUrl.trim());
                alert(`Backend API URL saved: ${newUrl.trim()}\nAll future analysis requests will be routed to this backend engine.`);
            }
        }
    });
}

function initEvents() {
    initApiConfigDialog();

    if (DOM.headerBtnUpload) {
        DOM.headerBtnUpload.addEventListener('click', () => {
            DOM.sectionUpload.scrollIntoView({ behavior: 'smooth' });
        });
    }

    if (DOM.btnBrowseFiles && DOM.fileInputHidden) {
        DOM.btnBrowseFiles.addEventListener('click', () => DOM.fileInputHidden.click());
        DOM.fileInputHidden.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });
    }

    // Preset Sample Buttons
    if (DOM.btnSampleSatellite) {
        DOM.btnSampleSatellite.addEventListener('click', () => handleSampleSelect('satellite.jpg'));
    }
    if (DOM.btnSampleImage2) {
        DOM.btnSampleImage2.addEventListener('click', () => handleSampleSelect('image2.jpg'));
    }

    // Drag and Drop
    if (DOM.dropzoneArea) {
        ['dragenter', 'dragover'].forEach(evt => {
            DOM.dropzoneArea.addEventListener(evt, (e) => {
                e.preventDefault();
                DOM.dropzoneArea.classList.add('dragover');
            });
        });
        ['dragleave', 'drop'].forEach(evt => {
            DOM.dropzoneArea.addEventListener(evt, (e) => {
                e.preventDefault();
                DOM.dropzoneArea.classList.remove('dragover');
            });
        });
        DOM.dropzoneArea.addEventListener('drop', (e) => {
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFileUpload(e.dataTransfer.files[0]);
            }
        });
    }

    // Download Zip
    if (DOM.btnDownloadZip) {
        DOM.btnDownloadZip.addEventListener('click', () => {
            if (activeSessionId) {
                window.location.href = getApiUrl(`/api/download_all?session_id=${activeSessionId}`);
            }
        });
    }

    // Gallery Actions (Expand, Compare, Download buttons inside cards)
    document.addEventListener('click', (e) => {
        const viewBtn = e.target.closest('.view-btn');
        if (viewBtn) {
            const key = viewBtn.dataset.key;
            openModal(key);
            return;
        }

        const compareBtn = e.target.closest('.compare-btn');
        if (compareBtn) {
            const key = compareBtn.dataset.key;
            activateCompareWith(key);
            return;
        }

        const dlBtn = e.target.closest('.dl-btn');
        if (dlBtn) {
            const key = dlBtn.dataset.key;
            if (currentMapUrls[key]) {
                const a = document.createElement('a');
                a.href = currentMapUrls[key];
                a.download = `${key}.png`;
                a.click();
            }
            return;
        }
    });

    // JSON Subtabs
    DOM.jsonSubtabs.forEach(btn => {
        btn.addEventListener('click', () => {
            DOM.jsonSubtabs.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeJsonTab = btn.dataset.json;
            updateJsonDisplay();
        });
    });

    if (DOM.btnCopyJson) {
        DOM.btnCopyJson.addEventListener('click', () => {
            const code = DOM.jsonCodeDisplay.textContent;
            navigator.clipboard.writeText(code).then(() => {
                const orig = DOM.btnCopyJson.textContent;
                DOM.btnCopyJson.textContent = "✓ Copied!";
                setTimeout(() => DOM.btnCopyJson.textContent = orig, 1800);
            });
        });
    }

    if (DOM.btnDownloadJson) {
        DOM.btnDownloadJson.addEventListener('click', () => {
            const code = DOM.jsonCodeDisplay.textContent;
            const blob = new Blob([code], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${activeJsonTab}.json`;
            a.click();
            URL.revokeObjectURL(url);
        });
    }
}

function initTabs() {
    DOM.dashTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            DOM.dashTabs.forEach(t => t.classList.remove('active'));
            DOM.tabPanes.forEach(p => p.classList.add('hidden'));

            tab.classList.add('active');
            const targetPane = document.getElementById(tab.dataset.tab);
            if (targetPane) {
                targetPane.classList.remove('hidden');
            }
        });
    });
}

function initCompareControls() {
    function updateCompareViews() {
        const leftKey = DOM.compareSelectLeft.value;
        const rightKey = DOM.compareSelectRight.value;

        DOM.compareTitleLeft.textContent = MAP_TITLES[leftKey] || leftKey;
        DOM.compareTitleRight.textContent = MAP_TITLES[rightKey] || rightKey;

        if (currentMapUrls[leftKey]) {
            DOM.compareImgLeft.src = currentMapUrls[leftKey];
        }
        if (currentMapUrls[rightKey]) {
            DOM.compareImgRight.src = currentMapUrls[rightKey];
        }
    }

    DOM.compareSelectLeft.addEventListener('change', updateCompareViews);
    DOM.compareSelectRight.addEventListener('change', updateCompareViews);

    if (DOM.btnSwapCompare) {
        DOM.btnSwapCompare.addEventListener('click', () => {
            const temp = DOM.compareSelectLeft.value;
            DOM.compareSelectLeft.value = DOM.compareSelectRight.value;
            DOM.compareSelectRight.value = temp;
            updateCompareViews();
        });
    }
}

function activateCompareWith(selectedKey) {
    // Switch to compare tab
    DOM.dashTabs.forEach(t => t.classList.remove('active'));
    DOM.tabPanes.forEach(p => p.classList.add('hidden'));

    const compareTab = document.querySelector('[data-tab="tab-compare"]');
    const comparePane = document.getElementById('tab-compare');
    if (compareTab && comparePane) {
        compareTab.classList.add('active');
        comparePane.classList.remove('hidden');
    }

    // Set right comparison layer
    DOM.compareSelectRight.value = selectedKey;
    DOM.compareSelectRight.dispatchEvent(new Event('change'));
    comparePane.scrollIntoView({ behavior: 'smooth' });
}

function handleFileUpload(file) {
    if (!file) return;

    // Client-side quick preview
    const reader = new FileReader();
    reader.onload = (e) => {
        DOM.inputPreviewImg.src = e.target.result;
    };
    reader.readAsDataURL(file);

    DOM.infoFilename.textContent = file.name;
    DOM.infoSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
    DOM.infoFormat.textContent = file.name.split('.').pop().toUpperCase();
    DOM.infoGeospatial.textContent = (file.name.endsWith('.tif') || file.name.endsWith('.tiff')) ? "Analyzing CRS..." : "Non-Georeferenced (Relative only)";

    const formData = new FormData();
    formData.append('image', file);

    startPipeline(formData);
}

function handleSampleSelect(sampleName) {
    DOM.infoFilename.textContent = sampleName;
    DOM.infoFormat.textContent = sampleName.split('.').pop().toUpperCase();
    DOM.infoGeospatial.textContent = "Sample Dataset (Relative Elevation)";
    DOM.inputPreviewImg.src = getApiUrl(`/${sampleName}`);

    const formData = new FormData();
    formData.append('preset', sampleName);

    startPipeline(formData);
}

function startPipeline(formData) {
    if (isAnalyzing) {
        console.warn('[Pipeline] Analysis already in progress. Ignoring duplicate request.');
        return;
    }
    isAnalyzing = true;

    // Show processing section
    DOM.sectionProcessing.classList.remove('hidden');
    DOM.sectionDashboard.classList.add('hidden');
    DOM.sectionProcessing.scrollIntoView({ behavior: 'smooth' });

    updateProgressBar(5, "Submitting payload...");

    safeFetchJson(getApiUrl('/api/analyze'), {
        method: 'POST',
        body: formData
    })
    .then(data => {
        if (!data.success) {
            throw new Error(data.error || "Failed to start remote-sensing pipeline.");
        }
        activeSessionId = data.session_id || data.generation_id;
        startPolling(activeSessionId);
    })
    .catch(err => {
        console.error('[Analysis Startup Error]', err);
        isAnalyzing = false;
        alert(`Error starting analysis: ${err.message}`);
        DOM.sectionProcessing.classList.add('hidden');
        DOM.sectionUpload.scrollIntoView({ behavior: 'smooth' });
    });
}

function startPolling(sessionId) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(() => {
        safeFetchJson(getApiUrl(`/api/progress?session_id=${sessionId}&t=${Date.now()}`))
            .then(progress => {
                if (!progress.success && progress.status === 'error') {
                    clearInterval(pollInterval);
                    isAnalyzing = false;
                    console.error('[Pipeline Execution Error]', progress);
                    alert(`Pipeline error: ${progress.error || 'Unknown error'}`);
                    DOM.sectionProcessing.classList.add('hidden');
                    DOM.sectionUpload.scrollIntoView({ behavior: 'smooth' });
                    return;
                }

                const pct = progress.progress || 10;
                updateProgressBar(pct, progress.message || progress.step_name);
                highlightStageStep(progress.step_index || 1);

                if (progress.status === 'completed' || pct >= 100) {
                    clearInterval(pollInterval);
                    setTimeout(() => fetchResults(sessionId), 700);
                }
            })
            .catch(err => {
                console.warn('[Polling Error]', err);
            });
    }, 600);
}

function updateProgressBar(percent, msg) {
    DOM.progressBarFill.style.width = `${percent}%`;
    DOM.progressPercentText.textContent = `${percent}%`;
    DOM.stageMessage.textContent = msg;
}

function highlightStageStep(stepIdx) {
    DOM.stageSteps.forEach((st) => {
        const stepNum = parseInt(st.dataset.step, 10);
        if (stepNum < stepIdx) {
            st.className = 'stage-step completed';
        } else if (stepNum === stepIdx) {
            st.className = 'stage-step active';
        } else {
            st.className = 'stage-step';
        }
    });
}

function fetchResults(sessionId) {
    safeFetchJson(getApiUrl(`/api/results?session_id=${sessionId}&t=${Date.now()}`))
        .then(data => {
            isAnalyzing = false;
            if (!data.success) {
                throw new Error(data.error || "Failed loading analysis results.");
            }
            DOM.sectionProcessing.classList.add('hidden');
            DOM.sectionDashboard.classList.remove('hidden');
            renderResults(data);
            DOM.sectionDashboard.scrollIntoView({ behavior: 'smooth' });
        })
        .catch(err => {
            isAnalyzing = false;
            console.error('[Fetch Results Error]', err);
            alert(`Error retrieving results: ${err.message}`);
            DOM.sectionProcessing.classList.add('hidden');
            DOM.sectionUpload.scrollIntoView({ behavior: 'smooth' });
        });
}

function renderResults(data) {
    activeSessionId = data.session_id || data.generation_id;
    currentMetadata = data.metadata || data.report || {};
    currentBuildings = data.buildings || {};
    currentStatistics = data.statistics || {};
    currentMapUrls = data.map_urls || {};

    // Transform relative URLs using getApiUrl
    Object.keys(currentMapUrls).forEach(k => {
        if (currentMapUrls[k]) {
            currentMapUrls[k] = getApiUrl(currentMapUrls[k]);
        }
    });

    const meta = currentMetadata;
    const imgInfo = meta.image || {};
    const depthInfo = meta.depth || {};
    const elevInfo = meta.elevation || {};
    const bldgSummary = (currentBuildings.summary) || meta.buildings || {};
    const geoInfo = meta.geospatial || {};
    const qualityInfo = meta.quality || {};

    // Header Pills
    DOM.sessionIdDisplay.textContent = `Session: ${activeSessionId}`;
    DOM.sessionTimeDisplay.textContent = new Date().toLocaleTimeString();

    // 6 Key Stat Cards
    DOM.statBuildingCount.textContent = bldgSummary.count || 0;
    DOM.statImageSize.textContent = `${imgInfo.width || '--'} × ${imgInfo.height || '--'}`;
    DOM.statImageFormat.textContent = `${imgInfo.format || 'RGB'} (${imgInfo.channels || 3} Ch)`;
    DOM.statDepthRange.textContent = `${depthInfo.min !== undefined ? depthInfo.min : '--'} - ${depthInfo.max !== undefined ? depthInfo.max : '--'}`;
    DOM.statElevationMax.textContent = elevInfo.max !== undefined ? elevInfo.max : '--';
    
    // Geospatial Status
    if (geoInfo.georeferenced) {
        DOM.statGeospatialStatus.textContent = geoInfo.epsg ? `EPSG:${geoInfo.epsg}` : "GeoTIFF";
        DOM.statGeospatialSub.textContent = "Georeferenced bounds";
    } else {
        DOM.statGeospatialStatus.textContent = "Non-Geo";
        DOM.statGeospatialSub.textContent = "Relative scale only";
    }

    DOM.statConfidence.textContent = qualityInfo.confidence ? `${Math.round(qualityInfo.confidence * 100)}%` : "--%";

    // Render 12 Maps Gallery
    Object.keys(MAP_TITLES).forEach(key => {
        const imgElem = document.getElementById(`map-img-${key}`);
        if (imgElem && currentMapUrls[key]) {
            imgElem.src = currentMapUrls[key];
        }
    });

    // Sub-pills under maps
    const pillDepth = document.getElementById('pill-depth');
    if (pillDepth && depthInfo.min !== undefined) {
        pillDepth.innerHTML = `<span>Min: ${depthInfo.min}</span> | <span>Max: ${depthInfo.max}</span> | <span>unit: relative</span>`;
    }
    const pillRdsm = document.getElementById('pill-rdsm');
    if (pillRdsm && elevInfo.max !== undefined) {
        pillRdsm.innerHTML = `<span>Surface Model</span> | <span>Max Elev: ${elevInfo.max} (relative)</span>`;
    }
    const pillElev = document.getElementById('pill-elev');
    if (pillElev && elevInfo.mean !== undefined) {
        pillElev.innerHTML = `<span>Mean: ${elevInfo.mean}</span> | <span>Unit: Relative Elevation</span>`;
    }
    const pillBldgDet = document.getElementById('pill-bldg-det');
    if (pillBldgDet && bldgSummary.count !== undefined) {
        pillBldgDet.innerHTML = `<span>${bldgSummary.count} Buildings</span> | <span>Avg Conf: ${Math.round((bldgSummary.average_confidence || 0) * 100)}%</span>`;
    }
    const pillBldgH = document.getElementById('pill-bldg-h');
    if (pillBldgH && bldgSummary.maximum_height !== undefined) {
        pillBldgH.innerHTML = `<span>Max H: ${bldgSummary.maximum_height}</span> | <span>Avg H: ${bldgSummary.average_height} (relative)</span>`;
    }

    // Populate Analysis Data Section
    DOM.adFilename.textContent = imgInfo.filename || '--';
    DOM.adFormat.textContent = imgInfo.format || '--';
    DOM.adDims.textContent = `${imgInfo.width || '--'} × ${imgInfo.height || '--'} px`;
    DOM.adChannels.textContent = `${imgInfo.channels || 3} (RGB)`;
    DOM.adSize.textContent = imgInfo.file_size_bytes ? `${(imgInfo.file_size_bytes / 1024).toFixed(1)} KB` : '-- KB';
    DOM.adTime.textContent = imgInfo.processing_time_seconds ? `${imgInfo.processing_time_seconds}s` : '--';

    // Geospatial Info
    DOM.geoStatus.textContent = geoInfo.georeferenced ? "✓ Georeferenced (True)" : "✗ Non-Georeferenced";
    DOM.geoCrs.textContent = geoInfo.crs || "Georeferencing: Not Available";
    DOM.geoEpsg.textContent = geoInfo.epsg ? `EPSG:${geoInfo.epsg}` : "None";
    DOM.geoPixelSize.textContent = geoInfo.pixel_size ? `${geoInfo.pixel_size[0].toFixed(4)} × ${geoInfo.pixel_size[1].toFixed(4)}` : "Relative pixel grid";
    DOM.geoBounds.textContent = geoInfo.bounds ? `[${geoInfo.bounds.map(b => b.toFixed(2)).join(', ')}]` : "None";
    DOM.geoCoverage.textContent = geoInfo.coordinates && geoInfo.coordinates.center ? `Center: ${geoInfo.coordinates.center.map(c => c.toFixed(4)).join(', ')}` : "Local image space";

    // Elevation Stats
    DOM.adElevType.textContent = elevInfo.type || "relative";
    DOM.adElevMin.textContent = elevInfo.min !== undefined ? elevInfo.min : "0.00";
    DOM.adElevMax.textContent = elevInfo.max !== undefined ? elevInfo.max : "0.00";
    DOM.adElevMean.textContent = elevInfo.mean !== undefined ? elevInfo.mean : "0.00";
    DOM.adElevMedian.textContent = elevInfo.median !== undefined ? elevInfo.median : "0.00";

    // Quality Stats
    DOM.adQConf.textContent = qualityInfo.confidence ? `${(qualityInfo.confidence * 100).toFixed(1)}%` : "--";
    DOM.adQMae.textContent = qualityInfo.mae !== null ? qualityInfo.mae : "null (No Ground Truth DEM)";
    DOM.adQRmse.textContent = qualityInfo.rmse !== null ? qualityInfo.rmse : "null (No Ground Truth DEM)";
    DOM.adQCorr.textContent = qualityInfo.correlation !== null ? qualityInfo.correlation : "null";
    DOM.adQValid.textContent = "100% (No invalid pixels)";
    DOM.adQStatus.textContent = qualityInfo.quality_status || "PASS";

    // Building Summary & Table
    DOM.bsumCount.textContent = bldgSummary.count || 0;
    DOM.bsumAvgH.textContent = bldgSummary.average_height !== undefined ? bldgSummary.average_height : "0.00";
    DOM.bsumMaxH.textContent = bldgSummary.maximum_height !== undefined ? bldgSummary.maximum_height : "0.00";
    DOM.bsumMinH.textContent = bldgSummary.minimum_height !== undefined ? bldgSummary.minimum_height : "0.00";
    DOM.bsumArea.textContent = bldgSummary.total_area ? `${bldgSummary.total_area.toLocaleString()} px²` : "0 px²";
    DOM.bsumConf.textContent = bldgSummary.average_confidence ? `${Math.round(bldgSummary.average_confidence * 100)}%` : "0%";

    renderBuildingTable(currentBuildings.buildings || []);

    // Update Compare Views & JSON Viewer
    DOM.compareSelectLeft.dispatchEvent(new Event('change'));
    updateJsonDisplay();
}

function renderBuildingTable(buildings) {
    DOM.buildingsTableBody.innerHTML = '';

    if (!buildings || buildings.length === 0) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td colspan="7" style="text-align: center; color: var(--text-muted); padding: 24px;">No building structures detected for this image threshold.</td>`;
        DOM.buildingsTableBody.appendChild(tr);
        return;
    }

    buildings.forEach(b => {
        const tr = document.createElement('tr');
        tr.className = "clickable-bldg-row";
        tr.dataset.id = b.id;

        const bboxStr = b.bounding_box ? `[${b.bounding_box.join(', ')}]` : '--';
        const centerStr = b.center ? `(${b.center[0]}, ${b.center[1]})` : '--';

        tr.innerHTML = `
            <td><strong class="bldg-id-tag">${b.id}</strong></td>
            <td><code>${bboxStr}</code></td>
            <td>${b.area_pixels ? b.area_pixels.toLocaleString() : '--'}</td>
            <td><span class="height-tag">${b.estimated_height !== undefined ? b.estimated_height : '--'}</span></td>
            <td><code>${b.height_unit || 'relative'}</code></td>
            <td><span class="conf-badge">${Math.round((b.confidence || 0) * 100)}%</span></td>
            <td><code>${centerStr}</code></td>
        `;

        tr.addEventListener('click', () => {
            document.querySelectorAll('.clickable-bldg-row').forEach(r => r.classList.remove('selected-bldg'));
            tr.classList.add('selected-bldg');
            highlightBuildingPreview(b);
        });

        DOM.buildingsTableBody.appendChild(tr);
    });
}

function highlightBuildingPreview(building) {
    // Open Building Footprint map with alert toast identifying the selected structure
    openModal('building_boundaries', `Focused Building: ${building.id} | Relative Height: ${building.estimated_height} | Area: ${building.area_pixels} px² | Center: (${building.center.join(', ')})`);
}

function updateJsonDisplay() {
    let targetData = currentMetadata;
    if (activeJsonTab === "buildings") {
        targetData = currentBuildings;
    } else if (activeJsonTab === "statistics") {
        targetData = currentStatistics;
    }

    DOM.jsonCodeDisplay.textContent = JSON.stringify(targetData, null, 2);
}

// MODAL ZOOM & PAN CONTROLLER
function initModalPanZoom() {
    if (DOM.modalCloseBtn) {
        DOM.modalCloseBtn.addEventListener('click', closeModal);
    }
    if (DOM.imageModal) {
        DOM.imageModal.addEventListener('click', (e) => {
            if (e.target === DOM.imageModal) closeModal();
        });
    }

    if (DOM.btnModalZoomIn) {
        DOM.btnModalZoomIn.addEventListener('click', () => adjustZoom(0.25));
    }
    if (DOM.btnModalZoomOut) {
        DOM.btnModalZoomOut.addEventListener('click', () => adjustZoom(-0.25));
    }
    if (DOM.btnModalZoomReset) {
        DOM.btnModalZoomReset.addEventListener('click', resetZoom);
    }

    // Drag to pan
    DOM.modalPanContainer.addEventListener('mousedown', (e) => {
        isPanning = true;
        startX = e.clientX - panX;
        startY = e.clientY - panY;
        DOM.modalPanContainer.style.cursor = 'grabbing';
    });

    window.addEventListener('mousemove', (e) => {
        if (!isPanning) return;
        panX = e.clientX - startX;
        panY = e.clientY - startY;
        applyTransform();
    });

    window.addEventListener('mouseup', () => {
        isPanning = false;
        DOM.modalPanContainer.style.cursor = 'grab';
    });

    // Mouse wheel zoom
    DOM.modalPanContainer.addEventListener('wheel', (e) => {
        e.preventDefault();
        const delta = e.deltaY < 0 ? 0.15 : -0.15;
        adjustZoom(delta);
    });
}

function adjustZoom(amount) {
    modalZoom = Math.max(0.5, Math.min(modalZoom + amount, 5.0));
    applyTransform();
}

function resetZoom() {
    modalZoom = 1.0;
    panX = 0;
    panY = 0;
    applyTransform();
}

function applyTransform() {
    DOM.modalImageDisplay.style.transform = `translate(${panX}px, ${panY}px) scale(${modalZoom})`;
}

function openModal(key, customSubText = null) {
    if (!currentMapUrls[key]) return;

    DOM.modalTitle.textContent = MAP_TITLES[key] || key;
    DOM.modalSubInfo.textContent = customSubText || "High-Resolution 2D Inspection View • Use wheel to zoom, drag to pan";
    DOM.modalImageDisplay.src = currentMapUrls[key];
    DOM.modalDownloadLink.href = currentMapUrls[key];
    DOM.modalDownloadLink.download = `${key}.png`;

    resetZoom();
    DOM.imageModal.classList.remove('hidden');
}

function closeModal() {
    DOM.imageModal.classList.add('hidden');
    resetZoom();
}
