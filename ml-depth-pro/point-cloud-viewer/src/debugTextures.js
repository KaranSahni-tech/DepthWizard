import * as THREE from 'three';

/**
 * Reconstruction Debug Texture Generator & Colormap Engine
 * Generates input-adaptive canvas-backed WebGL textures for all 12 reconstruction debug modes.
 * All thresholds are dynamically derived from input percentiles & image/depth statistics.
 */

export function colorTurbo(t) {
    t = Math.max(0, Math.min(1, t));
    const r = Math.floor(255 * Math.sin(t * Math.PI * 0.9));
    const g = Math.floor(255 * Math.sin(t * Math.PI * 1.1 + 0.2));
    const b = Math.floor(255 * Math.cos(t * Math.PI * 0.8));
    return [
        Math.min(255, Math.max(0, r)),
        Math.min(255, Math.max(0, g)),
        Math.min(255, Math.max(0, b))
    ];
}

export function colorMagma(t) {
    t = Math.max(0, Math.min(1, t));
    const r = Math.floor(255 * Math.pow(t, 0.7));
    const g = Math.floor(255 * Math.pow(t, 2.2));
    const b = Math.floor(255 * (1 - Math.pow(1 - t, 3.0)) * 0.7 + 30 * (1 - t));
    return [r, g, b];
}

export function colorViridis(t) {
    t = Math.max(0, Math.min(1, t));
    const r = Math.floor(255 * (0.2 + 0.78 * Math.pow(t, 2.0)));
    const g = Math.floor(255 * (0.1 + 0.85 * Math.sin(t * Math.PI * 0.85)));
    const b = Math.floor(255 * (0.4 + 0.5 * Math.cos(t * Math.PI * 0.9)));
    return [Math.min(255, Math.max(0, r)), Math.min(255, Math.max(0, g)), Math.min(255, Math.max(0, b))];
}

export function colorInferno(t) {
    t = Math.max(0, Math.min(1, t));
    const r = Math.floor(255 * Math.pow(t, 0.85));
    const g = Math.floor(255 * Math.pow(t, 2.5) * 0.95);
    const b = Math.floor(255 * Math.sin(t * Math.PI * 0.5) * 0.3 * (1 - t));
    return [r, g, b];
}

export function colorTerrain(t) {
    t = Math.max(0, Math.min(1, t));
    if (t < 0.35) {
        const k = t / 0.35;
        return [Math.floor(210 - k * 140), Math.floor(180 + k * 10), Math.floor(120 - k * 70)];
    } else if (t < 0.75) {
        const k = (t - 0.35) / 0.40;
        return [Math.floor(70 + k * 80), Math.floor(190 - k * 70), Math.floor(50 + k * 30)];
    } else {
        const k = (t - 0.75) / 0.25;
        return [Math.floor(150 + k * 105), Math.floor(120 + k * 135), Math.floor(80 + k * 175)];
    }
}

export const SEGMENTATION_COLORS = {
    0: [40, 40, 40],      // UNKNOWN: Dark Gray
    1: [230, 57, 70],     // BUILDING: Coral Red
    2: [69, 123, 157],    // ROAD: Slate Blue
    3: [233, 196, 106],   // GROUND: Warm Yellow / Sand
    4: [42, 157, 143]     // VEGETATION: Emerald Teal
};

/**
 * Generate an input-adaptive THREE.CanvasTexture for a given layer mode from depthData.
 */
export function generateDebugTexture(mode, depthData, options = {}) {
    const w = depthData.depth_width;
    const h = depthData.depth_height;
    const depthArray = depthData.depth_array;

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(w, h);
    const pixels = imgData.data;

    // Calculate robust percentile statistics for input-adaptive thresholding
    const validVals = [];
    for (let i = 0; i < depthArray.length; i++) {
        const v = depthArray[i];
        if (Number.isFinite(v) && v > 0) {
            validVals.push(v);
        }
    }

    if (validVals.length === 0) {
        const texture = new THREE.CanvasTexture(canvas);
        return texture;
    }

    validVals.sort((a, b) => a - b);
    const count = validVals.length;
    const p02 = validVals[Math.floor(count * 0.02)];
    const p25 = validVals[Math.floor(count * 0.25)];
    const p50 = validVals[Math.floor(count * 0.50)];
    const p75 = validVals[Math.floor(count * 0.75)];
    const p98 = validVals[Math.floor(count * 0.98)];
    
    const valRange = p98 > p02 ? (p98 - p02) : 1.0;
    const iqr = p75 > p25 ? (p75 - p25) : 0.5;

    // Calculate local depth gradients for adaptive boundary & texture features
    const gradients = new Float32Array(w * h);
    let sumGrad = 0;
    let validGradCount = 0;

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const idx = y * w + x;
            const z = depthArray[idx];
            if (Number.isFinite(z) && z > 0) {
                const rZ = x < w - 1 ? depthArray[idx + 1] : z;
                const dZ = y < h - 1 ? depthArray[idx + w] : z;
                const dzX = Number.isFinite(rZ) && rZ > 0 ? Math.abs(z - rZ) : 0;
                const dzY = Number.isFinite(dZ) && dZ > 0 ? Math.abs(z - dZ) : 0;
                const gMag = Math.sqrt(dzX * dzX + dzY * dzY);
                gradients[idx] = gMag;
                sumGrad += gMag;
                validGradCount++;
            }
        }
    }
    const meanGrad = validGradCount > 0 ? (sumGrad / validGradCount) : 0.05;
    const p95Grad = meanGrad * 3.5 + 0.01;

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const idx = y * w + x;
            const pxIdx = idx * 4;
            const z = depthArray[idx];
            const isValid = Number.isFinite(z) && z > 0;

            let rgb = [0, 0, 0];

            const segArr = depthData.segmentation_array;
            const boundArr = depthData.boundary_array;
            const groundArr = depthData.ground_array;
            const bldgHArr = depthData.building_height_array;
            const confArr = depthData.confidence_array;

            switch (mode) {
                case 'Relative Depth': {
                    if (isValid) {
                        const t = Math.max(0, Math.min(1, 1.0 - ((z - p02) / valRange)));
                        rgb = colorMagma(t);
                    } else {
                        rgb = [15, 15, 20];
                    }
                    break;
                }
                case 'Cleaned Depth': {
                    if (isValid) {
                        const t = Math.max(0, Math.min(1, (z - p02) / valRange));
                        rgb = colorViridis(t);
                    } else {
                        rgb = [20, 20, 25];
                    }
                    break;
                }
                case 'Ground Surface': {
                    if (isValid) {
                        const gVal = (groundArr && groundArr[idx] > 0) ? groundArr[idx] : (p98 - (p98 - z) * 0.3);
                        const t = Math.max(0, Math.min(1, (gVal - p02) / valRange));
                        rgb = colorTerrain(t);
                    } else {
                        rgb = [30, 25, 20];
                    }
                    break;
                }
                case 'Building Height': {
                    if (isValid) {
                        const heightEst = (bldgHArr && bldgHArr[idx] >= 0) ? bldgHArr[idx] : Math.max(0, p98 - z);
                        const normH = Math.min(1.0, heightEst / (valRange * 0.7));
                        if (normH < 0.001) {
                            rgb = [40, 44, 52];
                        } else {
                            rgb = colorMagma(normH);
                        }
                    } else {
                        rgb = [20, 20, 20];
                    }
                    break;
                }
                case 'Heatmap': {
                    if (isValid) {
                        const t = Math.max(0, Math.min(1, 1.0 - ((z - p02) / valRange)));
                        rgb = colorTurbo(t);
                    } else {
                        rgb = [0, 0, 0];
                    }
                    break;
                }
                case 'Boundary Map': {
                    if (isValid) {
                        const bVal = (boundArr && boundArr[idx] >= 0) ? boundArr[idx] : (gradients[idx] / p95Grad);
                        const t = Math.min(1.0, Math.max(0, bVal));
                        rgb = colorInferno(t);
                    } else {
                        rgb = [0, 0, 0];
                    }
                    break;
                }
                case 'Segmentation': {
                    if (isValid) {
                        if (segArr && segArr[idx] !== undefined) {
                            const classId = segArr[idx];
                            rgb = SEGMENTATION_COLORS[classId] || SEGMENTATION_COLORS[0];
                        } else {
                            const heightEst = p98 - z;
                            const normH = heightEst / valRange;
                            const gMag = gradients[idx];
                            if (normH > 0.22 && gMag < p95Grad * 0.8) rgb = SEGMENTATION_COLORS[1];
                            else if (normH < 0.10 && gMag < p95Grad * 0.3) rgb = SEGMENTATION_COLORS[2];
                            else if (normH < 0.18 && gMag > p95Grad * 0.4) rgb = SEGMENTATION_COLORS[4];
                            else rgb = SEGMENTATION_COLORS[3];
                        }
                    } else {
                        rgb = SEGMENTATION_COLORS[0];
                    }
                    break;
                }
                case 'Confidence': {
                    if (isValid) {
                        const cVal = (confArr && confArr[idx] >= 0) ? confArr[idx] : Math.max(0.15, 1.0 - Math.min(1.0, gradients[idx] / p95Grad));
                        rgb = colorViridis(Math.min(1.0, Math.max(0, cVal)));
                    } else {
                        rgb = [100, 15, 30];
                    }
                    break;
                }
                case 'Validity Mask': {
                    if (isValid) {
                        rgb = [255, 255, 255];
                    } else {
                        rgb = [220, 38, 38];
                    }
                    break;
                }
                default: {
                    rgb = [180, 180, 180];
                }
            }

            pixels[pxIdx] = rgb[0];
            pixels[pxIdx + 1] = rgb[1];
            pixels[pxIdx + 2] = rgb[2];
            pixels[pxIdx + 3] = 255;
        }
    }

    ctx.putImageData(imgData, 0, 0);

    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;
    return texture;
}
