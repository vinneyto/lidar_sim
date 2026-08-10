import { defineConfig } from "vite";

export default defineConfig({
  build: {
    outDir: "../src/gaussian_lidar/frontend",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/scene.ply": "http://127.0.0.1:8765",
      "/ws": { target: "ws://127.0.0.1:8765", ws: true },
    },
  },
});
