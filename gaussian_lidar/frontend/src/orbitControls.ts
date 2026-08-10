import type { PerspectiveCamera, WebGLRenderer } from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export function createOrbitControls(camera: PerspectiveCamera, renderer: WebGLRenderer) {
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 1);
  controls.enableDamping = true;
  return controls;
}
