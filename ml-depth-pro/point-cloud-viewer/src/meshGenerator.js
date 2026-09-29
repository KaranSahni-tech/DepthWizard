import * as THREE from 'three';

/**
 * Stage A: Outlier Removal & Validation
 */
function preprocessDepthArray(depthArray, width, height, step) {
    const gridW = Math.floor(width / step);
    const gridH = Math.floor(height / step);
    const processed = new Float32Array(gridW * gridH).fill(-1);
    let invalidHeightCount = 0;
    
    for (let gy = 0; gy < gridH; gy++) {
        for (let gx = 0; gx < gridW; gx++) {
            const idx = (gy * step) * width + (gx * step);
            const z = depthArray[idx];
            if (Number.isFinite(z) && z > 0) {
                processed[gy * gridW + gx] = z;
            } else {
                invalidHeightCount++;
            }
        }
    }
    
    return { processed, gridW, gridH, invalidHeightCount };
}

export function createMesh(depthData, texture, options = {}) {
    const {
        depth_width,
        depth_height,
        depth_array,
        focal_length,
        principal_point,
    } = depthData;

    const step = Math.max(1, Math.floor(options.resolution || 1));
    const zScale = options.zScale || 1.0;
    
    // Compute adaptive maxDepthDiff from the actual depth distribution
    // This is critical for aerial images where all height differences are tiny
    let maxDepthDiff = options.maxDepthDiff || 0.15;
    const maxEdgeLengthRatio = options.maxEdgeLengthRatio || 2.5;

    // Compute depth statistics for adaptive threshold
    const validSample = [];
    for (let i = 0; i < depth_array.length; i += step) {
        const v = depth_array[i];
        if (Number.isFinite(v) && v > 0) validSample.push(v);
    }
    if (validSample.length > 0) {
        validSample.sort((a, b) => a - b);
        const n = validSample.length;
        const p05 = validSample[Math.floor(n * 0.05)];
        const p95 = validSample[Math.floor(n * 0.95)];
        const depthRange = p95 - p05;
        if (depthRange > 0) {
            // Adaptive threshold: 8% of total depth range — severs building walls, keeps flat surfaces
            const adaptiveThreshold = depthRange * 0.08;
            // Use the smaller of user-specified and adaptive, but never allow fixed value to overwhelm aerial data
            maxDepthDiff = Math.min(maxDepthDiff, adaptiveThreshold);
        }
    }

    const fx = focal_length || 1342.9;
    const fy = focal_length || 1342.9;
    const cx = (principal_point && principal_point[0]) || depth_width / 2.0;
    const cy = (principal_point && principal_point[1]) || depth_height / 2.0;

    // Stage 1: Preprocess depth array to grid & check invalid heights
    const { processed: gridDepth, gridW, gridH, invalidHeightCount } = preprocessDepthArray(depth_array, depth_width, depth_height, step);

    const totalGridPixels = gridW * gridH;
    const positions = [];
    const uvs = [];
    const indices = [];

    const vertexMap = new Int32Array(totalGridPixels).fill(-1);
    let vIdx = 0;
    let hasNaNOrInf = false;
    
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    const gridPoints = new Array(totalGridPixels).fill(null);
    const relativeHeights = [];

    // Stage 2: Perspective 3D Coordinate Generation
    for (let gy = 0; gy < gridH; gy++) {
        for (let gx = 0; gx < gridW; gx++) {
            let z = gridDepth[gy * gridW + gx];

            if (z > 0) {
                const x = gx * step;
                const y = gy * step;

                const px = (x - cx) * z / fx;
                const py = -(y - cy) * z / fy;
                const pz = -z;

                if (Number.isFinite(px) && Number.isFinite(py) && Number.isFinite(pz)) {
                    positions.push(px, py, pz);

                    const u = x / (depth_width - 1);
                    const v = 1.0 - (y / (depth_height - 1));
                    uvs.push(u, v);

                    const gridIdx = gy * gridW + gx;
                    vertexMap[gridIdx] = vIdx;
                    gridPoints[gridIdx] = new THREE.Vector3(px, py, pz);
                    vIdx++;

                    if (px < minX) minX = px; if (px > maxX) maxX = px;
                    if (py < minY) minY = py; if (py > maxY) maxY = py;
                    if (pz < minZ) minZ = pz; if (pz > maxZ) maxZ = pz;
                } else {
                    hasNaNOrInf = true;
                    gridDepth[gy * gridW + gx] = -1;
                }
            }
        }
    }

    if (vIdx === 0) {
        throw new Error("ERROR: No valid 3D vertices were generated.");
    }

    // Dynamic edge limit & giant edge calculation
    let totalEdgeLength = 0;
    let edgeSampleCount = 0;

    for (let gy = 0; gy < gridH - 1; gy++) {
        for (let gx = 0; gx < gridW - 1; gx++) {
            const pA = gridPoints[gy * gridW + gx];
            const pB = gridPoints[gy * gridW + (gx + 1)];
            const pC = gridPoints[(gy + 1) * gridW + gx];
            
            if (pA && pB) { totalEdgeLength += pA.distanceTo(pB); edgeSampleCount++; }
            if (pA && pC) { totalEdgeLength += pA.distanceTo(pC); edgeSampleCount++; }
        }
    }
    
    const avgEdgeLength = edgeSampleCount > 0 ? (totalEdgeLength / edgeSampleCount) : 0;
    const edgeLimit = avgEdgeLength * maxEdgeLengthRatio;

    let giantEdgeCount = 0;
    for (let gy = 0; gy < gridH - 1; gy++) {
        for (let gx = 0; gx < gridW - 1; gx++) {
            const pA = gridPoints[gy * gridW + gx];
            const pB = gridPoints[gy * gridW + (gx + 1)];
            const pC = gridPoints[(gy + 1) * gridW + gx];
            
            if (pA && pB && pA.distanceTo(pB) > edgeLimit) giantEdgeCount++;
            if (pA && pC && pA.distanceTo(pC) > edgeLimit) giantEdgeCount++;
        }
    }

    // Stage 3: 6-Step Structure-Aware Triangulation
    let validTriangles = 0;
    let rejectedTriangles = 0;

    const boundaryArray = depthData.boundary_array;
    const segArray = depthData.segmentation_array;
    const bThresh = 0.35;

    for (let gy = 0; gy < gridH - 1; gy++) {
        for (let gx = 0; gx < gridW - 1; gx++) {
            const idxA = gy * gridW + gx;
            const idxB = gy * gridW + (gx + 1);
            const idxC = (gy + 1) * gridW + gx;
            const idxD = (gy + 1) * gridW + (gx + 1);

            const iA = vertexMap[idxA];
            const iB = vertexMap[idxB];
            const iC = vertexMap[idxC];
            const iD = vertexMap[idxD];

            if (iA === -1 || iB === -1 || iC === -1 || iD === -1) {
                continue;
            }

            const pA = gridPoints[idxA];
            const pB = gridPoints[idxB];
            const pC = gridPoints[idxC];
            const pD = gridPoints[idxD];

            const edgeAC = pA.distanceTo(pC);
            const edgeCB = pC.distanceTo(pB);
            const edgeBA = pB.distanceTo(pA);
            
            const zA = gridDepth[idxA];
            const zB = gridDepth[idxB];
            const zC = gridDepth[idxC];
            
            const diffACB = Math.max(Math.abs(zA - zC), Math.abs(zA - zB), Math.abs(zC - zB));

            let crossBoundaryACB = false;
            if (boundaryArray && segArray) {
                const origA = (gy * step) * depth_width + (gx * step);
                const origB = (gy * step) * depth_width + ((gx + 1) * step);
                const origC = ((gy + 1) * step) * depth_width + (gx * step);
                
                const segA = segArray[origA], segB = segArray[origB], segC = segArray[origC];
                const bA = boundaryArray[origA], bB = boundaryArray[origB], bC = boundaryArray[origC];

                if ((bA >= bThresh && bB >= bThresh && segA !== segB) ||
                    (bA >= bThresh && bC >= bThresh && segA !== segC) ||
                    (bC >= bThresh && bB >= bThresh && segC !== segB)) {
                    crossBoundaryACB = true;
                }
            }
            
            if (diffACB <= maxDepthDiff && edgeAC < edgeLimit && edgeCB < edgeLimit && edgeBA < edgeLimit && !crossBoundaryACB) {
                indices.push(iA, iC, iB);
                validTriangles++;
            } else {
                rejectedTriangles++;
            }

            const zD = gridDepth[idxD];
            const edgeBC = pB.distanceTo(pC);
            const edgeCD = pC.distanceTo(pD);
            const edgeDB = pD.distanceTo(pB);
            
            const diffBCD = Math.max(Math.abs(zB - zC), Math.abs(zB - zD), Math.abs(zC - zD));

            let crossBoundaryBCD = false;
            if (boundaryArray && segArray) {
                const origB = (gy * step) * depth_width + ((gx + 1) * step);
                const origC = ((gy + 1) * step) * depth_width + (gx * step);
                const origD = ((gy + 1) * step) * depth_width + ((gx + 1) * step);
                
                const segB = segArray[origB], segC = segArray[origC], segD = segArray[origD];
                const bB = boundaryArray[origB], bC = boundaryArray[origC], bD = boundaryArray[origD];

                if ((bB >= bThresh && bC >= bThresh && segB !== segC) ||
                    (bB >= bThresh && bD >= bThresh && segB !== segD) ||
                    (bC >= bThresh && bD >= bThresh && segC !== segD)) {
                    crossBoundaryBCD = true;
                }
            }

            if (diffBCD <= maxDepthDiff && edgeBC < edgeLimit && edgeCD < edgeLimit && edgeDB < edgeLimit && !crossBoundaryBCD) {
                indices.push(iB, iC, iD);
                validTriangles++;
            } else {
                rejectedTriangles++;
            }
        }
    }

    // Connected Component Analysis
    const parent = new Int32Array(vIdx);
    for (let i = 0; i < vIdx; i++) parent[i] = i;

    function find(i) {
        let root = i;
        while (root !== parent[root]) root = parent[root];
        let curr = i;
        while (curr !== root) {
            let nxt = parent[curr];
            parent[curr] = root;
            curr = nxt;
        }
        return root;
    }

    function union(i, j) {
        const rootI = find(i);
        const rootJ = find(j);
        if (rootI !== rootJ) parent[rootI] = rootJ;
    }

    for (let i = 0; i < indices.length; i += 3) {
        union(indices[i], indices[i+1]);
        union(indices[i+1], indices[i+2]);
    }

    const compSizes = new Int32Array(vIdx);
    for (let i = 0; i < vIdx; i++) {
        compSizes[find(i)]++;
    }

    const minComponentSize = options.minComponentSize || 30;
    const finalIndices = [];
    let noiseTriangles = 0;

    for (let i = 0; i < indices.length; i += 3) {
        const root = find(indices[i]);
        if (compSizes[root] >= minComponentSize) {
            finalIndices.push(indices[i], indices[i+1], indices[i+2]);
        } else {
            noiseTriangles++;
        }
    }
    
    validTriangles = finalIndices.length / 3;

    // Calculate Component Count
    const uniqueCompRoots = new Set();
    for (let i = 0; i < vIdx; i++) {
        const root = find(i);
        if (compSizes[root] >= minComponentSize) {
            uniqueCompRoots.add(root);
        }
    }
    const componentCount = uniqueCompRoots.size;

    // Calculate Degenerate & Giant Triangles
    let degenerateTriangleCount = 0;
    let giantTriangleCount = 0;

    const vecA = new THREE.Vector3();
    const vecB = new THREE.Vector3();
    const vecC = new THREE.Vector3();
    const crossVec = new THREE.Vector3();

    for (let i = 0; i < finalIndices.length; i += 3) {
        const idxA = finalIndices[i] * 3;
        const idxB = finalIndices[i+1] * 3;
        const idxC = finalIndices[i+2] * 3;

        vecA.set(positions[idxA], positions[idxA+1], positions[idxA+2]);
        vecB.set(positions[idxB], positions[idxB+1], positions[idxB+2]);
        vecC.set(positions[idxC], positions[idxC+1], positions[idxC+2]);

        const e1 = vecA.distanceTo(vecB);
        const e2 = vecB.distanceTo(vecC);
        const e3 = vecC.distanceTo(vecA);

        crossVec.crossVectors(vecB.clone().sub(vecA), vecC.clone().sub(vecA));
        const triArea = 0.5 * crossVec.length();

        if (triArea < 1e-7 || e1 < 1e-6 || e2 < 1e-6 || e3 < 1e-6) {
            degenerateTriangleCount++;
        }

        const maxTriEdge = Math.max(e1, e2, e3);
        if (maxTriEdge > avgEdgeLength * 3.0) {
            giantTriangleCount++;
        }
    }

    // Relative Elevation Statistics (Depth-derived relative height above ground reference)
    // Note: Depth minZ is closest to camera (highest elevation), maxZ is furthest (ground base)
    const groundBaseZ = -minZ; // In inverted Z coordinates
    let sumRelElev = 0;
    let maxRelativeElevation = 0;
    let minRelativeElevation = Infinity;

    for (let i = 0; i < vIdx; i++) {
        const pz = positions[i * 3 + 2];
        const relElev = Math.max(0, (pz - minZ));
        relativeHeights.push(relElev);
        sumRelElev += relElev;
        if (relElev > maxRelativeElevation) maxRelativeElevation = relElev;
        if (relElev < minRelativeElevation) minRelativeElevation = relElev;
    }
    const avgRelativeElevation = vIdx > 0 ? (sumRelElev / vIdx) : 0;
    if (minRelativeElevation === Infinity) minRelativeElevation = 0;

    const validVertexPercentage = (vIdx / totalGridPixels) * 100.0;

    // --- AUTOMATIC MESH QUALITY VALIDATION STAGE ---
    let status = 'GOOD';
    const issues = [];

    // Check 1: NaN or Infinity in Geometry
    if (hasNaNOrInf) {
        status = 'ERROR';
        issues.push('CRITICAL: Geometry contains NaN or Infinity values.');
    }

    // Check 2: Vertex Count & Coverage
    if (vIdx < 100 || validVertexPercentage < 5.0) {
        status = 'ERROR';
        issues.push('CRITICAL: Valid depth coverage is extremely low (< 5%).');
    } else if (validVertexPercentage < 65.0) {
        if (status !== 'ERROR') status = 'WARNING';
        issues.push(`WARNING: Valid depth coverage is low (${validVertexPercentage.toFixed(1)}%).`);
    }

    // Check 3: Triangle Rejection Rate
    const totalPotentialTriangles = validTriangles + rejectedTriangles;
    const rejectionRate = totalPotentialTriangles > 0 ? (rejectedTriangles / totalPotentialTriangles) : 0;
    if (rejectionRate > 0.40) {
        if (status !== 'ERROR') status = 'WARNING';
        issues.push(`WARNING: High triangle rejection rate (${(rejectionRate * 100).toFixed(1)}%).`);
    }

    // Check 4: Enormous Vertical Spike Detection
    if (avgRelativeElevation > 0 && maxRelativeElevation > 5.5 * avgRelativeElevation) {
        if (status !== 'ERROR') status = 'WARNING';
        issues.push(`WARNING: Enormous vertical spike detected (Max ${maxRelativeElevation.toFixed(2)}m vs Avg ${avgRelativeElevation.toFixed(2)}m).`);
    }

    // Check 5: Giant Triangles & Edges
    if (giantTriangleCount > validTriangles * 0.05 || giantEdgeCount > 50) {
        if (status !== 'ERROR') status = 'WARNING';
        issues.push(`WARNING: High concentration of giant stretched triangles/edges (${giantTriangleCount} giant tris).`);
    }

    // Check 6: Mesh Fragmentation
    if (componentCount > 12) {
        if (status !== 'ERROR') status = 'WARNING';
        issues.push(`WARNING: Mesh is heavily fragmented into ${componentCount} disconnected components.`);
    }

    let statusMessage = "Mesh Quality Validation Passed";
    if (status === 'WARNING') {
        statusMessage = "Reconstruction Warning: Anomalies Detected";
    } else if (status === 'ERROR') {
        statusMessage = "Reconstruction Error: Invalid Geometry";
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    
    geometry.setIndex(finalIndices);
    geometry.computeVertexNormals();

    // Stage 4: Center and Normalize Model
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const centerZ = (minZ + maxZ) / 2;
    geometry.translate(-centerX, -centerY, -centerZ);

    const sizeX = maxX - minX;
    const sizeY = maxY - minY;
    const sizeZ = maxZ - minZ;
    const maxDim = Math.max(sizeX, sizeY, sizeZ);
    const scaleTarget = 10.0;
    const scaleFactor = maxDim > 0 ? (scaleTarget / maxDim) : 1;
    
    geometry.scale(scaleFactor, scaleFactor, scaleFactor);

    // Store unexaggerated base positions for fast, non-destructive Height Exaggeration updates
    const posAttr = geometry.attributes.position;
    const basePositions = new Float32Array(posAttr.array);
    geometry.userData.basePositions = basePositions;

    if (zScale !== 1.0) {
        const array = posAttr.array;
        for (let i = 2; i < array.length; i += 3) {
            array[i] = basePositions[i] * zScale;
        }
        posAttr.needsUpdate = true;
        geometry.computeVertexNormals();
    }

    if (texture) {
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.needsUpdate = true;
    }

    const materialOptions = {
        color: 0xffffff,
        roughness: 0.7,
        metalness: 0.1,
        side: THREE.DoubleSide
    };

    if (options.useTexture !== false && texture) {
        materialOptions.map = texture;
    } else if (options.wireframe) {
        materialOptions.wireframe = true;
        materialOptions.color = 0x00ffaa;
    } else {
        materialOptions.color = 0x99aab5;
    }

    const material = new THREE.MeshStandardMaterial(materialOptions);
    const mesh = new THREE.Mesh(geometry, material);
    
    return {
        mesh,
        stats: {
            vertexCount: vIdx,
            triangleCount: validTriangles,
            rejectedTriangles: rejectedTriangles,
            noiseTriangles: noiseTriangles,
            scaleFactor: scaleFactor,
            rawMinZ: minZ,
            rawMaxZ: maxZ
        },
        quality: {
            status,
            statusMessage,
            issues,
            metrics: {
                vertexCount: vIdx,
                triangleCount: validTriangles,
                validVertexPercentage,
                rejectedTriangleCount: rejectedTriangles,
                degenerateTriangleCount,
                giantTriangleCount,
                giantEdgeCount,
                componentCount,
                invalidHeightCount,
                maxRelativeElevation,
                minRelativeElevation,
                avgRelativeElevation
            }
        }
    };
}
