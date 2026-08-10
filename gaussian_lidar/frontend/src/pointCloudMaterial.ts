import { MeshStandardMaterial } from "three";

export function createPointCloudMaterial() {
  return new MeshStandardMaterial({
    color: 0x35e6ff,
    emissive: 0x073c48,
    depthTest: false,
    roughness: 0.25,
  });
}
