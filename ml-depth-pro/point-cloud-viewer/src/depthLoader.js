export async function loadDepthData(url) {
    const response = await fetch(url);
    const rawText = await response.text();
    if (!response.ok) {
        throw new Error(`Failed to load depth data from ${url}: HTTP ${response.status} ${response.statusText}`);
    }
    if (!rawText.trim()) {
        throw new Error(`Empty response received from ${url}`);
    }
    let data;
    try {
        data = JSON.parse(rawText);
    } catch (e) {
        throw new Error(`Invalid JSON received from ${url}: ${rawText.slice(0, 200)}`);
    }
    
    function flatten2D(arr, TargetType) {
        if (!arr || !Array.isArray(arr)) return null;
        if (!Array.isArray(arr[0])) return new TargetType(arr);
        const w = data.depth_width;
        const h = data.depth_height;
        const flat = new TargetType(w * h);
        let idx = 0;
        for (let i = 0; i < arr.length; i++) {
            const row = arr[i];
            for (let j = 0; j < row.length; j++) {
                flat[idx++] = row[j];
            }
        }
        return flat;
    }

    if (data.depth_array) data.depth_array = flatten2D(data.depth_array, Float32Array);
    if (data.segmentation_array) data.segmentation_array = flatten2D(data.segmentation_array, Uint8Array);
    if (data.boundary_array) data.boundary_array = flatten2D(data.boundary_array, Float32Array);
    if (data.ground_array) data.ground_array = flatten2D(data.ground_array, Float32Array);
    if (data.building_height_array) data.building_height_array = flatten2D(data.building_height_array, Float32Array);
    if (data.confidence_array) data.confidence_array = flatten2D(data.confidence_array, Float32Array);

    return data;
}

export async function loadImageToCanvas(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "Anonymous";
        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            ctx.drawImage(img, 0, 0);
            resolve({ canvas, ctx, image: img });
        };
        img.onerror = () => {
            reject(new Error(`Failed to load image from ${url}`));
        };
        img.src = url;
    });
}
