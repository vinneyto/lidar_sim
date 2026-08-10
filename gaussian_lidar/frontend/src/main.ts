import "./style.css";
import { registerEvents } from "./events";
import { LidarClient } from "./lidarClient";
import { createLidarMarker } from "./lidarMarker";
import { createOrbitControls } from "./orbitControls";
import { PointCloud } from "./pointCloud";
import { createScene } from "./scene";
import { createSplatMesh } from "./splatMesh";

const { scene, camera, renderer } = createScene();
const controls = createOrbitControls(camera, renderer);
const splats = createSplatMesh("/scene.ply");
const marker = createLidarMarker();
const pointCloud = new PointCloud(scene);
const status = document.querySelector<HTMLDivElement>("#status")!;

scene.add(splats, marker);

const lidarClient = new LidarClient(status, (response) => {
  if (!response.ok) {
    status.textContent = response.error ?? "Scan failed";
    return;
  }
  pointCloud.update(response.points ?? []);
  status.textContent = `${response.points?.length ?? 0} LiDAR hits`;
});

registerEvents({
  camera,
  renderer,
  splats,
  marker,
  onLidarMove: (position) => lidarClient.scheduleScan(position),
});

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});
