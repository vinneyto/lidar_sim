import { Raycaster, Vector2, Vector3 } from "three";
import type { Group, PerspectiveCamera, WebGLRenderer } from "three";
import type { SplatMesh } from "@sparkjsdev/spark";

export function registerEvents({
  camera,
  renderer,
  splats,
  marker,
  onLidarMove,
}: {
  camera: PerspectiveCamera;
  renderer: WebGLRenderer;
  splats: SplatMesh;
  marker: Group;
  onLidarMove: (position: Vector3) => void;
}) {
  const raycaster = new Raycaster();
  const pointer = new Vector2();

  renderer.domElement.addEventListener("pointermove", (event) => {
    if (event.buttons !== 0) return;
    pointer.set((event.clientX / innerWidth) * 2 - 1, -(event.clientY / innerHeight) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObject(splats, true)[0];
    if (!hit) return;

    marker.visible = true;
    marker.position.copy(hit.point);
    onLidarMove(hit.point.clone().add(new Vector3(0, 0, 1)));
  });

  addEventListener("resize", () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
}
