// --- DEPTHWIZARD — SAAS DASHBOARD CONTROLLER (SIH PS 175) ---

function getApiBaseUrl() {
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

function getApiUrl(path) {
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
let selectedFile = null;

// Modal zoom state
let modalZoom = 1.0;
let isPanning = false;
let startX = 0, startY = 0;
let panX = 0, panY = 0;

const MAP_TITLES = {
    rgb: "MAP 1 — Original RGB Image",
    depth: "MAP 2 — Depth Map",
    rdsm: "MAP 3 — Relative DSM (rDSM)",
    elevation_heatmap: "MAP 4 — Elevation Heatmap",
    building_detection: "MAP 5 — Building Detection Map",
    building_height: "MAP 6 — Building Height Map",
    edges: "MAP 7 — Edge & Boundary Map",
    building_boundaries: "MAP 8 — Building Footprint Map",
    slope: "MAP 9 — Slope Map",
    terrain_relief: "MAP 10 — Terrain Relief / Hillshade",
    semantic: "MAP 11 — Semantic Classification Map",
    error: "MAP 12 — Model Uncertainty / Confidence Map"
};

// INITIALIZATION
window.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initEvents();
    initCompareControls();
    initModalPanZoom();
    checkApiConnection();

    // Check initial session
    safeFetchJson(getApiUrl('/api/results?session_id=image3_output'))
        .then(data => {
            if (data && data.success) {
                renderResults(data);
            }
        })
        .catch(() => {});
});

// NAVIGATION CONTROLLER FOR 11 VIEWS
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const viewSections = document.querySelectorAll('.view-section');
    const pageTitle = document.getElementById('header-page-title');
    const pageSub = document.getElementById('header-page-sub');

    function switchView(targetId, titleText, subText) {
        navItems.forEach(item => item.classList.remove('active'));
        viewSections.forEach(sec => sec.classList.remove('active'));

        const targetNav = document.querySelector(`.nav-item[data-target="${targetId}"]`);
        const targetSection = document.getElementById(targetId);

        if (targetNav) targetNav.classList.add('active');
        if (targetSection) targetSection.classList.add('active');

        if (pageTitle && titleText) pageTitle.textContent = titleText;
        if (pageSub && subText) pageSub.textContent = subText;

        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const target = item.dataset.target;
            const title = item.dataset.title;
            const sub = item.dataset.sub;
            switchView(target, title, sub);
        });
    });

    // Mobile Sidebar Toggle
    const sidebar = document.getElementById('app-sidebar');
    const toggleBtn = document.getElementById('sidebar-toggle-btn');
    const closeBtn = document.getElementById('sidebar-close-btn');

    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', () => sidebar.classList.add('open'));
    }
    if (closeBtn && sidebar) {
        closeBtn.addEventListener('click', () => sidebar.classList.remove('open'));
    }

    // Quick Action Card Links
    const qaUpload = document.getElementById('qa-upload');
    const qaLayers = document.getElementById('qa-layers');
    const qaBuildings = document.getElementById('qa-buildings');
    const qaReports = document.getElementById('qa-reports');
    const bannerStart = document.getElementById('banner-btn-start');
    const bannerLayers = document.getElementById('banner-btn-layers');
    const headerBtnNew = document.getElementById('header-btn-new-analysis');
    const dashViewAll = document.getElementById('dash-btn-view-all');

    if (qaUpload) qaUpload.addEventListener('click', () => switchView('view-new-analysis', 'New Analysis Workspace', 'Upload imagery and configure extraction.'));
    if (qaLayers) qaLayers.addEventListener('click', () => switchView('view-map-layers', '12 Thematic Map Layers', 'Explore AI-generated depth, rDSM, and slope maps.'));
    if (qaBuildings) qaBuildings.addEventListener('click', () => switchView('view-building-analysis', 'Building Analysis', 'Detected footprints and height profiles.'));
    if (qaReports) qaReports.addEventListener('click', () => switchView('view-reports', 'Reports & Exports', 'Download full ZIPs and metadata.'));
    if (bannerStart) bannerStart.addEventListener('click', () => switchView('view-new-analysis', 'New Analysis Workspace', 'Upload imagery and configure extraction.'));
    if (bannerLayers) bannerLayers.addEventListener('click', () => switchView('view-map-layers', '12 Thematic Map Layers', 'Explore AI-generated depth, rDSM, and slope maps.'));
    if (headerBtnNew) headerBtnNew.addEventListener('click', () => switchView('view-new-analysis', 'New Analysis Workspace', 'Upload imagery and configure extraction.'));
    if (dashViewAll) dashViewAll.addEventListener('click', () => switchView('view-history', 'Analysis History', 'Review past remote-sensing sessions.'));
}

function initEvents() {
    initApiConfigDialog();

    const fileInput = document.getElementById('file-input-hidden');
    const browseBtn = document.getElementById('btn-browse-files');
    const dropzone = document.getElementById('dropzone-area');
    const sampleSat = document.getElementById('btn-sample-satellite');
    const sampleImg2 = document.getElementById('btn-sample-image2');
    const startBtn = document.getElementById('btn-start-analysis');

    if (browseBtn && fileInput) {
        browseBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });
    }

    if (sampleSat) sampleSat.addEventListener('click', () => handleSampleSelect('satellite.jpg'));
    if (sampleImg2) sampleImg2.addEventListener('click', () => handleSampleSelect('image2.jpg'));
    
    // Sample buttons in dataset tab
    const dsSat = document.getElementById('ds-sample-satellite');
    const dsImg2 = document.getElementById('ds-sample-image2');
    if (dsSat) dsSat.addEventListener('click', () => handleSampleSelect('satellite.jpg'));
    if (dsImg2) dsImg2.addEventListener('click', () => handleSampleSelect('image2.jpg'));

    if (dropzone) {
        ['dragenter', 'dragover'].forEach(evt => {
            dropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropzone.classList.add('dragover');
            });
        });
        ['dragleave', 'drop'].forEach(evt => {
            dropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropzone.classList.remove('dragover');
            });
        });
        dropzone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFileUpload(e.dataTransfer.files[0]);
            }
        });
    }

    if (startBtn) {
        startBtn.addEventListener('click', () => {
            if (!selectedFile && !window.selectedPreset) {
                alert("Please select or drop an image file first.");
                return;
            }
            const formData = new FormData();
            if (selectedFile) {
                formData.append('image', selectedFile);
            } else if (window.selectedPreset) {
                formData.append('preset', window.selectedPreset);
            }
            startPipeline(formData);
        });
    }

    // Radio card mode toggles
    const optRel = document.getElementById('opt-relative-card');
    const optCal = document.getElementById('opt-calibrated-card');
    if (optRel && optCal) {
        optRel.addEventListener('click', () => {
            optRel.classList.add('selected');
            optCal.classList.remove('selected');
        });
        optCal.addEventListener('click', () => {
            optCal.classList.add('selected');
            optRel.classList.remove('selected');
        });
    }

    // Export buttons
    const btnZip = document.getElementById('btn-download-zip');
    const btnZipMain = document.getElementById('btn-export-zip-main');
    const handleZip = () => {
        if (activeSessionId) {
            window.location.href = getApiUrl(`/api/download_all?session_id=${activeSessionId}`);
        } else {
            alert("No active session loaded to export.");
        }
    };
    if (btnZip) btnZip.addEventListener('click', handleZip);
    if (btnZipMain) btnZipMain.addEventListener('click', handleZip);

    // JSON Export buttons
    const btnMeta = document.getElementById('btn-export-meta-json');
    const btnBldg = document.getElementById('btn-export-bldg-json');
    const btnStats = document.getElementById('btn-export-stats-json');
    if (btnMeta) btnMeta.addEventListener('click', () => downloadJsonFile(currentMetadata, 'metadata.json'));
    if (btnBldg) btnBldg.addEventListener('click', () => downloadJsonFile(currentBuildings, 'buildings.json'));
    if (btnStats) btnStats.addEventListener('click', () => downloadJsonFile(currentStatistics, 'statistics.json'));

    // Building Search
    const bldgSearch = document.getElementById('bldg-search-input');
    if (bldgSearch) {
        bldgSearch.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            const rows = document.querySelectorAll('#buildings-table-body tr');
            rows.forEach(r => {
                const text = r.textContent.toLowerCase();
                r.style.display = text.includes(query) ? '' : 'none';
            });
        });
    }

    // Settings save & test
    const settingsSave = document.getElementById('settings-btn-save');
    const settingsTest = document.getElementById('settings-btn-test');
    const settingsInput = document.getElementById('settings-api-url');
    if (settingsInput) settingsInput.value = localStorage.getItem('DEPTHWIZARD_API_URL') || '';

    if (settingsSave && settingsInput) {
        settingsSave.addEventListener('click', () => {
            const val = settingsInput.value.trim();
            if (val === '') {
                localStorage.removeItem('DEPTHWIZARD_API_URL');
                alert("Cleared custom API URL override.");
            } else {
                localStorage.setItem('DEPTHWIZARD_API_URL', val);
                alert(`Saved API Server URL: ${val}`);
            }
            checkApiConnection();
        });
    }

    if (settingsTest) {
        settingsTest.addEventListener('click', () => checkApiConnection(true));
    }

    // Gallery Actions
    document.addEventListener('click', (e) => {
        const viewBtn = e.target.closest('.view-btn');
        if (viewBtn) {
            openModal(viewBtn.dataset.key);
            return;
        }

        const compareBtn = e.target.closest('.compare-btn');
        if (compareBtn) {
            activateCompareWith(compareBtn.dataset.key);
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

        const reloadBtn = e.target.closest('.btn-reload-session, .btn-view-analysis-row');
        if (reloadBtn) {
            const sid = reloadBtn.dataset.session;
            if (sid) loadSession(sid);
            return;
        }
    });
}

function handleFileUpload(file) {
    if (!file) return;

    // Validate format
    const allowed = ['image/jpeg', 'image/png', 'image/tiff', 'image/webp'];
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['jpg', 'jpeg', 'png', 'tif', 'tiff', 'webp'].includes(ext)) {
        alert(`Unsupported file format '.${ext}'. Please upload JPG, PNG, TIFF, or GeoTIFF.`);
        return;
    }

    selectedFile = file;
    window.selectedPreset = null;

    const reader = new FileReader();
    reader.onload = (e) => {
        const img = document.getElementById('input-preview-img');
        if (img) img.src = e.target.result;
    };
    reader.readAsDataURL(file);

    document.getElementById('info-filename').textContent = file.name;
    document.getElementById('info-size').textContent = `${(file.size / 1024).toFixed(1)} KB`;
    document.getElementById('info-format').textContent = ext.toUpperCase();
    document.getElementById('info-geospatial').textContent = (ext === 'tif' || ext === 'tiff') ? "GeoTIFF (Metadata Check)" : "Non-Geo (Relative Grid)";
}

function handleSampleSelect(sampleName) {
    selectedFile = null;
    window.selectedPreset = sampleName;

    const img = document.getElementById('input-preview-img');
    if (img) img.src = getApiUrl(`/${sampleName}`);

    document.getElementById('info-filename').textContent = sampleName;
    document.getElementById('info-format').textContent = sampleName.split('.').pop().toUpperCase();
    document.getElementById('info-geospatial').textContent = "Sample Dataset (Relative Elevation)";
    document.getElementById('info-size').textContent = "Sample Payload";
}

function startPipeline(formData) {
    if (isAnalyzing) {
        console.warn('[Pipeline] Analysis already running.');
        return;
    }
    isAnalyzing = true;

    const procCard = document.getElementById('section-processing');
    if (procCard) procCard.classList.remove('hidden');

    updateProgressBar(5, "Submitting remote sensing payload to Python backend engine...");

    safeFetchJson(getApiUrl('/api/analyze'), {
        method: 'POST',
        body: formData
    })
    .then(data => {
        if (!data.success) {
            throw new Error(data.error || "Failed to start pipeline.");
        }
        activeSessionId = data.session_id || data.generation_id;
        startPolling(activeSessionId);
    })
    .catch(err => {
        console.error('[Analysis Startup Error]', err);
        isAnalyzing = false;
        alert(`Error starting analysis: ${err.message}`);
        if (procCard) procCard.classList.add('hidden');
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
                    alert(`Pipeline error: ${progress.error || 'Unknown error'}`);
                    const procCard = document.getElementById('section-processing');
                    if (procCard) procCard.classList.add('hidden');
                    return;
                }

                const pct = progress.progress || 10;
                updateProgressBar(pct, progress.message || progress.step_name);

                if (progress.status === 'completed' || pct >= 100) {
                    clearInterval(pollInterval);
                    setTimeout(() => fetchResults(sessionId), 600);
                }
            })
            .catch(err => {
                console.warn('[Polling Progress Warning]', err);
            });
    }, 600);
}

function updateProgressBar(percent, msg) {
    const fill = document.getElementById('progress-bar-fill');
    const txt = document.getElementById('progress-percent-text');
    const sub = document.getElementById('stage-message');

    if (fill) fill.style.width = `${percent}%`;
    if (txt) txt.textContent = `${percent}%`;
    if (sub) sub.textContent = msg;
}

function fetchResults(sessionId) {
    safeFetchJson(getApiUrl(`/api/results?session_id=${sessionId}&t=${Date.now()}`))
        .then(data => {
            isAnalyzing = false;
            if (!data.success) {
                throw new Error(data.error || "Failed loading analysis results.");
            }
            const procCard = document.getElementById('section-processing');
            if (procCard) procCard.classList.add('hidden');

            renderResults(data);

            // Switch view to 12 Map Layers
            const mapNav = document.querySelector('.nav-item[data-target="view-map-layers"]');
            if (mapNav) mapNav.click();
        })
        .catch(err => {
            isAnalyzing = false;
            console.error('[Fetch Results Error]', err);
            alert(`Error retrieving results: ${err.message}`);
        });
}

function loadSession(sessionId) {
    fetchResults(sessionId);
}

function renderResults(data) {
    activeSessionId = data.session_id || data.generation_id;
    currentMetadata = data.metadata || data.report || {};
    currentBuildings = data.buildings || {};
    currentStatistics = data.statistics || {};
    currentMapUrls = data.map_urls || {};

    // Transform map URLs using getApiUrl
    Object.keys(currentMapUrls).forEach(k => {
        if (currentMapUrls[k]) {
            currentMapUrls[k] = getApiUrl(currentMapUrls[k]);
        }
    });

    const meta = currentMetadata;
    const imgInfo = meta.image || {};
    const elevInfo = meta.elevation || {};
    const bldgSummary = (currentBuildings.summary) || meta.buildings || {};
    const geoInfo = meta.geospatial || {};

    // Update Dashboard Metrics
    const totalElem = document.getElementById('dash-stat-total');
    const imgsElem = document.getElementById('dash-stat-images');
    const bldgElem = document.getElementById('dash-stat-buildings');
    const expElem = document.getElementById('dash-stat-exports');
    if (totalElem) totalElem.textContent = "12";
    if (imgsElem) imgsElem.textContent = "12";
    if (bldgElem) bldgElem.textContent = bldgSummary.count !== undefined ? bldgSummary.count : 48;
    if (expElem) expElem.textContent = "12";

    // Update Session Pill
    const sessDisplay = document.getElementById('session-id-display');
    const timeDisplay = document.getElementById('session-time-display');
    if (sessDisplay) sessDisplay.textContent = `Session: ${activeSessionId}`;
    if (timeDisplay) timeDisplay.textContent = new Date().toLocaleTimeString();

    // Populate 12 Maps Gallery
    Object.keys(MAP_TITLES).forEach(key => {
        const imgElem = document.getElementById(`map-img-${key}`);
        if (imgElem && currentMapUrls[key]) {
            imgElem.src = currentMapUrls[key];
        }
    });

    // Populate Building Table & Stats
    const bCount = document.getElementById('bsum-count');
    const bAvgH = document.getElementById('bsum-avg-h');
    const bMaxH = document.getElementById('bsum-max-h');
    const bArea = document.getElementById('bsum-area');

    if (bCount) bCount.textContent = bldgSummary.count || 0;
    if (bAvgH) bAvgH.textContent = bldgSummary.average_height !== undefined ? bldgSummary.average_height : '--';
    if (bMaxH) bMaxH.textContent = bldgSummary.maximum_height !== undefined ? bldgSummary.maximum_height : '--';
    if (bArea) bArea.textContent = bldgSummary.total_area ? bldgSummary.total_area.toLocaleString() : '0';

    renderBuildingTable(currentBuildings.buildings || []);

    // Update Metadata View
    const geoStatus = document.getElementById('geo-status');
    const geoCrs = document.getElementById('geo-crs');
    const geoEpsg = document.getElementById('geo-epsg');
    const geoPixel = document.getElementById('geo-pixel-size');
    const geoBounds = document.getElementById('geo-bounds');

    if (geoStatus) geoStatus.textContent = geoInfo.georeferenced ? "✓ Georeferenced (GeoTIFF)" : "Relative local grid (Non-GeoTIFF)";
    if (geoCrs) geoCrs.textContent = geoInfo.crs || "None";
    if (geoEpsg) geoEpsg.textContent = geoInfo.epsg ? `EPSG:${geoInfo.epsg}` : "None";
    if (geoPixel) geoPixel.textContent = geoInfo.pixel_size ? `${geoInfo.pixel_size[0]} × ${geoInfo.pixel_size[1]}` : "1.0 × 1.0 px grid";
    if (geoBounds) geoBounds.textContent = geoInfo.bounds ? JSON.stringify(geoInfo.bounds) : "N/A";

    // Update JSON inspector
    const jsonDisp = document.getElementById('json-code-display');
    if (jsonDisp) jsonDisp.textContent = JSON.stringify(currentMetadata, null, 2);

    // Update Compare Views
    const compLeft = document.getElementById('compare-select-left');
    if (compLeft) compLeft.dispatchEvent(new Event('change'));
}

function renderBuildingTable(buildings) {
    const tbody = document.getElementById('buildings-table-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!buildings || buildings.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding: 24px; color: var(--text-muted);">No building structures detected for this image threshold.</td></tr>`;
        return;
    }

    buildings.forEach(b => {
        const tr = document.createElement('tr');
        const bboxStr = b.bounding_box ? `[${b.bounding_box.join(', ')}]` : '--';
        const centerStr = b.center ? `(${b.center[0]}, ${b.center[1]})` : '--';

        tr.innerHTML = `
            <td><strong>${b.id}</strong></td>
            <td><code>${bboxStr}</code></td>
            <td>${b.area_pixels ? b.area_pixels.toLocaleString() : '--'}</td>
            <td><strong style="color: var(--primary-blue);">${b.estimated_height !== undefined ? b.estimated_height : '--'}</strong></td>
            <td><code>${b.height_unit || 'relative'}</code></td>
            <td><span class="badge-tag badge-success">${Math.round((b.confidence || 0) * 100)}%</span></td>
            <td><code>${centerStr}</code></td>
        `;
        tbody.appendChild(tr);
    });
}

function initCompareControls() {
    const leftSel = document.getElementById('compare-select-left');
    const rightSel = document.getElementById('compare-select-right');
    const leftImg = document.getElementById('compare-img-left');
    const rightImg = document.getElementById('compare-img-right');
    const swapBtn = document.getElementById('btn-swap-compare');

    function updateCompare() {
        if (leftSel && leftImg && currentMapUrls[leftSel.value]) {
            leftImg.src = currentMapUrls[leftSel.value];
        }
        if (rightSel && rightImg && currentMapUrls[rightSel.value]) {
            rightImg.src = currentMapUrls[rightSel.value];
        }
    }

    if (leftSel) leftSel.addEventListener('change', updateCompare);
    if (rightSel) rightSel.addEventListener('change', updateCompare);

    if (swapBtn && leftSel && rightSel) {
        swapBtn.addEventListener('click', () => {
            const temp = leftSel.value;
            leftSel.value = rightSel.value;
            rightSel.value = temp;
            updateCompare();
        });
    }
}

function activateCompareWith(key) {
    const rightSel = document.getElementById('compare-select-right');
    if (rightSel) {
        rightSel.value = key;
        rightSel.dispatchEvent(new Event('change'));
    }
    const mapNav = document.querySelector('.nav-item[data-target="view-map-layers"]');
    if (mapNav) mapNav.click();
}

function checkApiConnection(showAlert = false) {
    const statusText = document.getElementById('api-status-text');
    const statusDot = document.getElementById('api-status-dot');
    const miniDot = document.getElementById('status-dot-mini');
    const miniText = document.getElementById('status-text-mini');

    safeFetchJson(getApiUrl('/api/health'))
        .then(data => {
            if (data && data.success) {
                if (statusText) statusText.textContent = "API Server (Online)";
                if (statusDot) statusDot.className = "status-dot online";
                if (miniDot) miniDot.className = "status-dot online";
                if (miniText) miniText.textContent = "Depth Engine Connected";
                if (showAlert) alert(`API Connection Successful!\nServer: ${getApiBaseUrl() || 'Local Origin'}\nStatus: Online`);
            } else {
                throw new Error("Invalid health check response");
            }
        })
        .catch(err => {
            if (statusText) statusText.textContent = "API Server (Disconnected)";
            if (statusDot) statusDot.className = "status-dot offline";
            if (miniDot) miniDot.className = "status-dot offline";
            if (miniText) miniText.textContent = "Engine Disconnected";
            if (showAlert) alert(`API Connection Failed:\n${err.message}`);
        });
}

function initApiConfigDialog() {
    const btn = document.getElementById('header-btn-api-config');
    if (!btn) return;

    btn.addEventListener('click', () => {
        const current = getApiBaseUrl() || "(Relative / Default Vercel Domain)";
        const newUrl = prompt(
            `Configure Production Backend API URL:\n\nCurrent active API base: ${current}\n\nEnter backend URL (e.g. http://localhost:5000 or https://your-backend-domain.com):`,
            localStorage.getItem('DEPTHWIZARD_API_URL') || ""
        );
        if (newUrl !== null) {
            if (newUrl.trim() === "") {
                localStorage.removeItem('DEPTHWIZARD_API_URL');
                alert("Cleared custom API URL override.");
            } else {
                localStorage.setItem('DEPTHWIZARD_API_URL', newUrl.trim());
                alert(`Saved API Server URL: ${newUrl.trim()}`);
            }
            checkApiConnection();
        }
    });
}

function downloadJsonFile(dataObj, filename) {
    const code = JSON.stringify(dataObj || {}, null, 2);
    const blob = new Blob([code], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// MODAL ZOOM & PAN
function initModalPanZoom() {
    const closeBtn = document.getElementById('modal-close-btn');
    const resetBtn = document.getElementById('btn-modal-zoom-reset');
    const modal = document.getElementById('image-modal');
    const viewport = document.getElementById('modal-pan-container');
    const displayImg = document.getElementById('modal-image-display');

    if (closeBtn) closeBtn.addEventListener('click', closeModal);
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeModal();
        });
    }
    if (resetBtn) resetBtn.addEventListener('click', resetZoom);

    if (viewport && displayImg) {
        viewport.addEventListener('mousedown', (e) => {
            isPanning = true;
            startX = e.clientX - panX;
            startY = e.clientY - panY;
            viewport.style.cursor = 'grabbing';
        });

        window.addEventListener('mousemove', (e) => {
            if (!isPanning) return;
            panX = e.clientX - startX;
            panY = e.clientY - startY;
            applyTransform();
        });

        window.addEventListener('mouseup', () => {
            isPanning = false;
            if (viewport) viewport.style.cursor = 'grab';
        });

        viewport.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY < 0 ? 0.15 : -0.15;
            modalZoom = Math.max(0.5, Math.min(modalZoom + delta, 5.0));
            applyTransform();
        });
    }
}

function resetZoom() {
    modalZoom = 1.0;
    panX = 0;
    panY = 0;
    applyTransform();
}

function applyTransform() {
    const displayImg = document.getElementById('modal-image-display');
    if (displayImg) {
        displayImg.style.transform = `translate(${panX}px, ${panY}px) scale(${modalZoom})`;
    }
}

function openModal(key) {
    const modal = document.getElementById('image-modal');
    const title = document.getElementById('modal-title');
    const displayImg = document.getElementById('modal-image-display');

    if (!currentMapUrls[key] || !modal) return;

    if (title) title.textContent = MAP_TITLES[key] || key;
    if (displayImg) displayImg.src = currentMapUrls[key];

    resetZoom();
    modal.classList.remove('hidden');
}

function closeModal() {
    const modal = document.getElementById('image-modal');
    if (modal) modal.classList.add('hidden');
    resetZoom();
}
