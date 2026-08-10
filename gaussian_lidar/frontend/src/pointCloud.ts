import { InstancedMesh, Matrix4, Scene, SphereGeometry } from "three";
import { createPointCloudMaterial } from "./pointCloudMaterial";

export type Point = [number, number, number];

export class PointCloud {
  private readonly geometry = new SphereGeometry(0.025, 8, 6);
  private readonly material = createPointCloudMaterial();
  private mesh?: InstancedMesh;

  constructor(private readonly scene: Scene) {}

  update(points: Point[]) {
    if (this.mesh) this.scene.remove(this.mesh);

    const mesh = new InstancedMesh(this.geometry, this.material, points.length);
    const matrix = new Matrix4();
    mesh.renderOrder = 10;
    points.forEach((point, index) => {
      matrix.makeTranslation(point[0], point[1], point[2]);
      mesh.setMatrixAt(index, matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
    this.scene.add(mesh);
    this.mesh = mesh;
  }
}
