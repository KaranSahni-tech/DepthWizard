import * as THREE from 'three';

export function createPointCloud(depthData, imageCtx, options = { pointSize: 0.05 }) {
    const {
        depth_width,
        depth_height,
        depth_array,
        focal_length,
        principal_point,
        image_width,
        image_height
    } = depthData;

    // Use focal length from JSON or fallback to default
    const fx = focal_length || 1342.9;
    const fy = focal_length || 1342.9;
    
    const cx = (principal_point && principal_point[0]) || depth_width / 2.0;
    const cy = (principal_point && principal_point[1]) || depth_height / 2.0;

    // Calculate how many points are valid
    let validCount = 0;
    for (let i = 0; i < depth_array.length; i++) {
        if (Number.isFinite(depth_array[i]) && depth_array[i] > 0) {
            validCount++;
        }
    }

    const positions = new Float32Array(validCount * 3);
    const colors = new Float32Array(validCount * 3);

    // If image and depth resolutions differ, calculate scale
    const scaleX = image_width / depth_width;
    const scaleY = image_height / depth_height;

    // Extract image data once for fast access
    const imageData = imageCtx && imageCtx.ctx ? imageCtx.ctx.getImageData(0, 0, image_width, image_height).data : null;

    let pIdx = 0;
    
    // Bounds tracking to center model later
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (let y = 0; y < depth_height; y++) {
        for (let x = 0; x < depth_width; x++) {
            const idx = y * depth_width + x;
            const z = depth_array[idx];

            if (Number.isFinite(z) && z > 0) {
                // Perspective reconstruction formula
                const px = (x - cx) * z / fx;
                // Invert Y and Z so the point cloud renders upright in Three.js standard coordinates
                const py = -(y - cy) * z / fy;
                const pz = -z;

                positions[pIdx * 3] = px;
                positions[pIdx * 3 + 1] = py;
                positions[pIdx * 3 + 2] = pz;

                // Update bounds
                if (px < minX) minX = px; if (px > maxX) maxX = px;
                if (py < minY) minY = py; if (py > maxY) maxY = py;
                if (pz < minZ) minZ = pz; if (pz > maxZ) maxZ = pz;

                // Color extraction
                if (imageData) {
                    const imgX = Math.floor(x * scaleX);
                    const imgY = Math.floor(y * scaleY);
                    const imgIdx = (imgY * image_width + imgX) * 4;
                    
                    colors[pIdx * 3] = imageData[imgIdx] / 255.0;
                    colors[pIdx * 3 + 1] = imageData[imgIdx + 1] / 255.0;
                    colors[pIdx * 3 + 2] = imageData[imgIdx + 2] / 255.0;
                } else {
                    // Fallback color if no image
                    colors[pIdx * 3] = 1.0;
                    colors[pIdx * 3 + 1] = 1.0;
                    colors[pIdx * 3 + 2] = 1.0;
                }
                
                pIdx++;
            }
        }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    // Center geometry around origin automatically
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const centerZ = (minZ + maxZ) / 2;
    
    geometry.translate(-centerX, -centerY, -centerZ);

    // Auto-scale to a reasonable viewing size without breaking relative proportions
    // e.g. fit longest axis into a 10-unit box
    const sizeX = maxX - minX;
    const sizeY = maxY - minY;
    const sizeZ = maxZ - minZ;
    const maxDim = Math.max(sizeX, sizeY, sizeZ);
    const scaleTarget = 10.0;
    
    const scaleFactor = maxDim > 0 ? (scaleTarget / maxDim) : 1;
    geometry.scale(scaleFactor, scaleFactor, scaleFactor);

    // Create material and points object
    const material = new THREE.PointsMaterial({
        size: options.pointSize,
        vertexColors: true,
        sizeAttenuation: true
    });

    const points = new THREE.Points(geometry, material);
    
    return {
        points,
        stats: {
            validCount,
            minZ: minZ * scaleFactor,
            maxZ: maxZ * scaleFactor,
            rawMinZ: minZ,
            rawMaxZ: maxZ,
            scaleFactor: scaleFactor,
            centerX: centerX,
            centerY: centerY,
            centerZ: centerZ,
            scaledBounds: { x: sizeX * scaleFactor, y: sizeY * scaleFactor, z: sizeZ * scaleFactor }
        }
    };
}
