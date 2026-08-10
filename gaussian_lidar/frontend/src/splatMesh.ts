import { SplatMesh } from "@sparkjsdev/spark";

export function createSplatMesh(url: string) {
  const splats = new SplatMesh({ url });
  splats.renderOrder = 0;
  return splats;
}
