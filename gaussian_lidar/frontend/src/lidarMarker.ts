import {
  BufferGeometry,
  CylinderGeometry,
  Group,
  Line,
  LineBasicMaterial,
  Mesh,
  MeshStandardMaterial,
  Vector3,
} from "three";

export function createLidarMarker() {
  const marker = new Group();
  marker.visible = false;
  marker.add(
    new Line(
      new BufferGeometry().setFromPoints([new Vector3(), new Vector3(0, 0, 1)]),
      new LineBasicMaterial({ color: 0xffc857 }),
    ),
  );

  const lidar = new Mesh(
    new CylinderGeometry(0.08, 0.08, 0.16, 24),
    new MeshStandardMaterial({ color: 0xff9f1c, roughness: 0.35 }),
  );
  lidar.rotation.x = Math.PI / 2;
  lidar.position.z = 1;
  marker.add(lidar);
  return marker;
}
