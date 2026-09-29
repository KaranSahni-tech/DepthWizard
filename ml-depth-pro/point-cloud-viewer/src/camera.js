import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export function setupCameraAndControls(canvas) {
    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.01, 2000);
    
    // Initial position — will be overridden once the model loads
    camera.position.set(0, 8, 14);
    camera.lookAt(0, 0, 0);

    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.enableZoom = true;
    controls.enablePan = true;
    controls.enableRotate = true;
    controls.zoomSpeed = 1.2;
    controls.rotateSpeed = 0.6;
    controls.panSpeed = 0.8;
    controls.minDistance = 0.5;
    controls.maxDistance = 500;
    // Allow looking from any elevation angle (no tilt lockout)
    controls.maxPolarAngle = Math.PI;
    
    return { camera, controls };
}

function getBoundingSphere(object) {
    if (!object) return null;
    if (object.geometry) {
        if (!object.geometry.boundingSphere) object.geometry.computeBoundingSphere();
        return object.geometry.boundingSphere;
    }
    const box = new THREE.Box3().setFromObject(object);
    if (box.isEmpty()) return null;
    const sphere = new THREE.Sphere();
    box.getBoundingSphere(sphere);
    return sphere;
}

// Fit the full model into view from current camera direction
export function fitModel(camera, controls, object) {
    const sphere = getBoundingSphere(object);
    if (!sphere) return;

    const radius = Math.max(sphere.radius, 0.5);
    const fov = camera.fov * (Math.PI / 180);
    const dist = Math.abs(radius / Math.sin(fov / 2)) * 1.4;

    const dir = camera.position.clone().sub(controls.target).normalize();
    const target = sphere.center.clone();
    camera.position.copy(target).addScaledVector(dir, dist);
    controls.target.copy(target);
    controls.update();
}

// Reset view = attractive aerial perspective showing the entire scene
export function resetCamera(camera, controls, object) {
    setAerialView(camera, controls, object);
}

// Attractive oblique aerial view — like a professional 3D map viewer
export function setAerialView(camera, controls, object) {
    const sphere = getBoundingSphere(object);
    if (!sphere) return;

    const radius = Math.max(sphere.radius, 0.5);
    const fov = camera.fov * (Math.PI / 180);
    const dist = Math.abs(radius / Math.sin(fov / 2)) * 1.35;

    const target = sphere.center.clone();
    // 45° oblique angle from slightly elevated northeast position
    camera.position.set(
        target.x + dist * 0.5,
        target.y + dist * 0.75,
        target.z + dist * 0.6
    );
    controls.target.copy(target);
    controls.update();
}

export function setTopView(camera, controls, object) {
    const sphere = getBoundingSphere(object);
    if (!sphere) return;

    const radius = Math.max(sphere.radius, 0.5);
    const fov = camera.fov * (Math.PI / 180);
    const dist = Math.abs(radius / Math.sin(fov / 2)) * 1.3;
    const target = sphere.center.clone();
    camera.position.set(target.x, target.y + dist, target.z + 0.001);
    controls.target.copy(target);
    controls.update();
}

export function setSideView(camera, controls, object) {
    const sphere = getBoundingSphere(object);
    if (!sphere) return;

    const radius = Math.max(sphere.radius, 0.5);
    const fov = camera.fov * (Math.PI / 180);
    const dist = Math.abs(radius / Math.sin(fov / 2)) * 1.4;
    const target = sphere.center.clone();
    camera.position.set(target.x + dist, target.y + dist * 0.15, target.z);
    controls.target.copy(target);
    controls.update();
}

export function setPerspectiveView(camera, controls, object) {
    const sphere = getBoundingSphere(object);
    if (!sphere) return;

    const radius = Math.max(sphere.radius, 0.5);
    const fov = camera.fov * (Math.PI / 180);
    const dist = Math.abs(radius / Math.sin(fov / 2)) * 1.2;
    const target = sphere.center.clone();
    camera.position.set(target.x + dist * 0.7, target.y + dist * 0.5, target.z + dist * 0.7);
    controls.target.copy(target);
    controls.update();
}
