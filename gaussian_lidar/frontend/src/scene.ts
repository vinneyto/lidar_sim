import {
  Color,
  DirectionalLight,
  HemisphereLight,
  PerspectiveCamera,
  Scene,
  WebGLRenderer,
} from "three";

export function createScene() {
  const scene = new Scene();
  scene.background = new Color(0x101319);

  const camera = new PerspectiveCamera(60, innerWidth / innerHeight, 0.01, 1000);
  camera.position.set(4, -6, 3);
  camera.up.set(0, 0, 1);

  const renderer = new WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  document.body.append(renderer.domElement);

  scene.add(new HemisphereLight(0xffffff, 0x283044, 2.2));
  const sun = new DirectionalLight(0xffffff, 2.5);
  sun.position.set(3, -4, 8);
  scene.add(sun);

  return { scene, camera, renderer };
}
