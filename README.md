# Virtual LiDAR from 3D Gaussian Splatting

An educational, PyTorch-first virtual LiDAR for canonical 3DGS PLY scenes. The
computed surface is **the first point along each ray where front-to-back
accumulated Gaussian opacity reaches a configurable threshold**. A Gaussian's
`3σ` ellipsoid is only a finite-support/BVH approximation—not a physical surface
and not an automatic return.

The CPU implementation is the readable mathematical reference. On Apple
Silicon, the Metal backend keeps the scene, flat BVH, rays, and outputs in MPS
tensors and dispatches a custom shader compiled with
`torch.mps.compile_shader`. It never uses CUDA, MLX, or a native extension.

## Install and run

```bash
uv sync --extra test
uv run pytest
# Edit the constants at the top of experiments/01_basic_scan.py, then run:
uv run python experiments/01_basic_scan.py
# Or animate repeated scans while the LiDAR follows an XZ-plane circle:
uv run python experiments/02_circular_scan.py
# Or move a tangential camera and render its 3DGS view with Metal:
uv run python experiments/03_circular_scan_with_camera.py
```

The PLY loader deliberately accepts the canonical Inria 3DGS convention:
`scale_*` are log standard deviations, `opacity` is a logit, and `rot_0..3` is a
scalar-first `(w,x,y,z)` quaternion. Missing fields cause an explicit error
rather than guessed interpretation. Internally scales are linear, opacities are
in `[0,1]`, and normalized quaternions remain scalar-first.

Coordinates are right-handed: `+X` forward, `+Y` left, `+Z` up. Azimuth is
endpoint-exclusive, so the default `[-π, π)` scan does not duplicate its seam.
CLI yaw/pitch/roll are degrees composed as intrinsic Z-Y-X rotations.

## Учебный курс

Пошаговый русскоязычный курс из десяти практических задач находится в [`course/`](course/README.md). В ходе курса пользователь собирает второй эксперимент с круговым сканированием и знакомится с публичными модулями проекта.

## Library API

```python
import torch
from gs_lidar import *

scene = load_gaussian_ply("scene.ply")
lo, hi = gaussian_aabbs(scene.means, scene.scales, scene.rotations)
bvh = build_bvh(lo, hi)

# Upload static data once; subsequent scans only change pose/config.
scene, bvh = scene.to("mps"), bvh.to("mps")
simulator = LidarSimulator(scene, bvh)
pose = LidarPose(torch.zeros(3, device="mps"),
                 torch.tensor([1., 0, 0, 0], device="mps"))
scan = simulator.scan(pose, LidarConfig())
points = scan.valid_points()  # [M, 3], still on MPS
```

For every ray, candidates are gathered by BVH traversal, evaluated at their
minimum Mahalanobis distance, sorted by `t_peak`, then composited as
`A ← A + (1-A)α`. Transparent foreground Gaussians therefore do not stop a ray.
The bounded Metal candidate list (128) and traversal stack (64) expose overflow
counters in `LidarScan`; overflow is never silent. Misses use `inf` range, `nan`
point, `false` mask, and Gaussian ID `-1`.

Rerun is isolated in `rerun_viewer.py`; only visualization copies tensors to
CPU/NumPy. Scene Gaussians are rendered with Rerun's native Gaussian splat
archetype, including their anisotropic scales, rotations, colors, and opacities.
Set `RERUN_UP_AXIS` in the experiment to `+X`, `-X`, `+Y`, `-Y`, `+Z`, or `-Z`
to match a reconstructed model's coordinate convention and make Rerun's
turntable camera orbit around that axis (the example defaults to `+Y`). The
3D view is explicitly rooted at `world`, so Rerun applies this convention to
its camera controls instead of falling back to its default Z-up root view.
This setting changes the viewer convention only; it does not flip or otherwise
transform scene data.

The second experiment uses a fixed one-meter orbit in the XZ plane, centered at
the coordinate origin. Its angular velocity, update interval, and number of
scans are constants at the top of `experiments/02_circular_scan.py`. Rerun shows
the path as a thin ring and records every sensor position and return cloud on
the `scan` timeline. Because this reconstructed scene is Y-up, the experiment
also rotates the simulator's native Z-up scan pattern so its azimuth plane is
parallel to the XZ tabletop/orbit plane.

The third experiment contains no LiDAR simulation. A pinhole camera follows the
same circular trajectory while a wireframe frustum pyramid marks its current
pose in the 3D view. The synchronized SH-free Metal 3DGS render appears on the
right at `1280×960`. The camera always points toward the center of the circular
trajectory and 15° downward, keeping the well-reconstructed central object in
view. PLY
rotations are loaded as `(w,x,y,z)` and converted once to `(x,y,z,w)`; that same
converted camera scene is passed to both Rerun and the course-derived renderer.
The camera keeps a right-handed `+X`-right/`+Y`-up/`+Z`-forward basis and clips
geometry outside `0.1–10 m`.
Projection, covariance projection, tile binning, radix sorting, and tile
rasterization run through custom Metal kernels; PyTorch owns and dispatches the
MPS buffers.

For diagnosing LiDAR backend instability, the second experiment also writes
`circular_scan_debug.jsonl`. It contains the exact pose, hit and overflow
counts, output fingerprints, range/alpha summaries, per-elevation-row hit
counts, and differences from the preceding scan. Attach this file when
reporting alternating or otherwise inconsistent frames.

## Scope and future work

This ray-space opacity threshold differs from camera rasterization and is not a
continuous density integral. The first milestone intentionally omits noise,
beam divergence, reflectance, multiple returns, autograd, GPU BVH construction,
and physical laser modeling. Natural follow-ups include thresholds 0.3/0.5/0.8,
2σ/3σ/4σ cutoffs, continuous integration, scan patterns, intensity/noise,
multi-return scans, priority traversal, and sorting-free compositing.
