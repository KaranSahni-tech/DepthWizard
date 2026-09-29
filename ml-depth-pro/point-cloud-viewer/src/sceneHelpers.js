import * as THREE from 'three';

export function createSceneHelpers() {
    // 1. Grid
    const gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
    gridHelper.visible = false;

    // 2. Axes
    const axesHelper = new THREE.AxesHelper(10);
    axesHelper.visible = false;

    // 3. Reference Plane (semitransparent)
    const planeGeo = new THREE.PlaneGeometry(20, 20);
    // Rotate to lay flat on XZ plane (since Y is up, but our mesh has Z-up ? Wait, our mesh Z is inverted depth.
    // In our reconstruction, Y is UP, Z is depth (towards camera is positive, but we set Z = -depth, so into the screen is negative).
    // The ground plane would be horizontal (XZ). Wait, an aerial photo is usually looking straight down (birds-eye view).
    // So X is horizontal, Y is vertical in the image, meaning depth is Z. So the reference plane should be parallel to the XY plane (screen plane).
    // Let's create a plane geometry that spans XY.
    const planeMat = new THREE.MeshBasicMaterial({
        color: 0x00ffaa,
        transparent: true,
        opacity: 0.2,
        side: THREE.DoubleSide,
        depthWrite: false
    });
    const referencePlane = new THREE.Mesh(planeGeo, planeMat);
    referencePlane.visible = false;

    // A group to hold them
    const group = new THREE.Group();
    group.add(gridHelper);
    group.add(axesHelper);
    group.add(referencePlane);

    // Grid normally lays on XZ, but we want it on XY if depth is Z
    gridHelper.rotation.x = Math.PI / 2;
    // Note: AxesHelper is (X=red, Y=green, Z=blue)

    return {
        group,
        gridHelper,
        axesHelper,
        referencePlane,
        setReferenceHeight: (zPos) => {
            referencePlane.position.z = zPos;
            gridHelper.position.z = zPos;
        }
    };
}
