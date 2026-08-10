import type { Vector3 } from "three";
import type { Point } from "./pointCloud";

type ScanResponse = { ok: boolean; points?: Point[]; error?: string };
type ResponseHandler = (response: ScanResponse) => void;

export class LidarClient {
  private readonly socket: WebSocket;
  private requestedPosition?: Vector3;
  private debounceTimer?: number;
  private requestInFlight = false;

  constructor(status: HTMLDivElement, onResponse: ResponseHandler) {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    this.socket = new WebSocket(`${protocol}//${location.host}/ws`);
    this.socket.addEventListener("open", () => {
      status.textContent = "Move over the scene to place the LiDAR";
    });
    this.socket.addEventListener("close", () => {
      status.textContent = "WebSocket disconnected";
    });
    this.socket.addEventListener("message", (event) => {
      this.requestInFlight = false;
      const response = JSON.parse(event.data) as ScanResponse;
      onResponse(response);
      if (this.requestedPosition) this.scheduleScan(this.requestedPosition);
    });
  }

  scheduleScan(position: Vector3) {
    this.requestedPosition = position.clone();
    window.clearTimeout(this.debounceTimer);
    this.debounceTimer = window.setTimeout(() => this.sendScan(), 180);
  }

  private sendScan() {
    if (
      this.socket.readyState !== WebSocket.OPEN ||
      this.requestInFlight ||
      !this.requestedPosition
    ) return;

    const position = this.requestedPosition;
    this.requestedPosition = undefined;
    this.requestInFlight = true;
    this.socket.send(JSON.stringify({
      sensor: { position: position.toArray(), quaternion: [1, 0, 0, 0] },
      scan: {
        width: 512,
        height: 32,
        horizontal_fov_deg: 360,
        vertical_fov_deg: 30,
        max_distance: 200,
      },
    }));
  }
}
