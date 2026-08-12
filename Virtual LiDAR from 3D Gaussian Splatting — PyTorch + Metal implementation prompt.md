# Task: Virtual LiDAR point cloud generation from 3D Gaussian Splatting

Create an educational but well-structured Python project that generates a synthetic LiDAR point cloud by ray tracing a 3D Gaussian Splatting scene.

The input is a standard 3DGS `.ply` file.

The project must use **PyTorch as its primary computational framework**.

The final LiDAR point cloud must be returned as a `torch.Tensor`.

The GPU implementation must target Apple Silicon / Metal through PyTorch MPS, preferably using custom Metal compute shaders compiled directly from Python through:

```python
torch.mps.compile_shader(...)
```

Do not use CUDA.

Do not use MLX.

Do not introduce a C++/Objective-C++ extension unless it becomes absolutely impossible to implement the required Metal kernel through the current PyTorch MPS tooling.

The first milestone should explicitly attempt the pure:

```text
Python
+
PyTorch
+
MPS tensors
+
custom Metal shader
```

approach.

---

# 1. Main goal

Implement a reusable module that computes a virtual LiDAR scan from:

- a Gaussian scene;
- a BVH built over the Gaussians;
- LiDAR position;
- LiDAR orientation;
- LiDAR angular configuration.

The LiDAR ray must **not stop at the first Gaussian it encounters**.

Instead, a ray may pass through many partially transparent Gaussians.

Their opacity contributions must be processed front-to-back using alpha compositing similar in spirit to a 3DGS renderer.

A LiDAR return is produced when accumulated opacity first exceeds a configurable threshold.

This behavior is central to the project.

---

# 2. Conceptual API

The public API should look approximately like:

```python
scan = simulate_lidar(
    scene=gaussian_cloud,
    bvh=bvh,
    pose=lidar_pose,
    config=lidar_config,
)
```

The result should contain PyTorch tensors.

For example:

```python
scan.points
scan.ranges
scan.hit_mask
scan.gaussian_ids
scan.accumulated_alpha
```

The main point-cloud API should allow:

```python
points = scan.valid_points()
```

with:

```text
points.shape == [M, 3]
```

and:

```python
isinstance(points, torch.Tensor)
```

If the simulation runs on MPS, the result should remain on MPS:

```python
points.device.type == "mps"
```

Do not automatically copy the final result back to CPU.

---

# 3. Technology

Use:

- Python
- `uv`
- PyTorch
- PyTorch MPS
- custom Metal shader through `torch.mps.compile_shader`
- `gsply` for loading 3DGS PLY when practical
- Rerun Python SDK
- pytest

Use NumPy only when required by an external library or file loader.

Convert external data into PyTorch tensors at the project boundary.

Do not make NumPy the project's internal computational representation.

---

# 4. Target platform

Primary target:

```text
macOS
Apple Silicon
PyTorch MPS
Metal
```

The CPU reference implementation must also work.

The main backends are conceptually:

```text
CPU:
    PyTorch tensors on device="cpu"

GPU:
    PyTorch tensors on device="mps"
    +
    custom Metal shader
```

---

# 5. Architecture

Use a structure approximately like:

```text
pyproject.toml
README.md

src/
    gs_lidar/
        __init__.py

        gaussian_cloud.py
        loader.py
        gaussian_geometry.py

        bvh.py
        bvh_builder.py

        lidar.py
        scan.py

        cpu_tracer.py
        metal_tracer.py
        metal/
            lidar_trace.metal

        rerun_viewer.py

experiments/
    01_basic_scan.py

tests/
    test_gaussian_geometry.py
    test_ray_gaussian.py
    test_lidar_rays.py
    test_alpha_accumulation.py
    test_bvh.py
    test_cpu_tracer.py
    test_metal_vs_cpu.py
```

The exact structure may differ if justified.

Do not create a monolithic implementation.

---

# 6. PyTorch-first design

All primary scene and simulation structures must use `torch.Tensor`.

For example:

```python
@dataclass
class GaussianCloud:
    means: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
```

Expected shapes:

```text
means:
    [N, 3]

scales:
    [N, 3]

rotations:
    [N, 4]

opacities:
    [N]
```

Optionally retain:

```python
colors: torch.Tensor | None
sh_coefficients: torch.Tensor | None
```

for visualization.

---

# 7. Device semantics

Objects should support explicit device placement.

For example:

```python
scene = scene.to("mps")
bvh = bvh.to("mps")
pose = pose.to("mps")
```

or an equivalent clean API.

Do not silently move tensors between devices inside performance-sensitive code.

Do not repeatedly transfer the static Gaussian scene or BVH between CPU and MPS for every scan.

Expected workflow:

```text
load PLY
    ↓
CPU torch tensors
    ↓
build BVH once
    ↓
transfer scene + BVH to MPS once
    ↓
perform many LiDAR scans
    ↓
results remain MPS tensors
```

This is important because the LiDAR pose may change while the Gaussian scene remains static.

---

# 8. PLY loading

Use `gsply` if practical.

If `gsply` produces NumPy arrays, immediately convert them:

```python
torch.from_numpy(...)
```

The rest of the project should operate on PyTorch tensors.

Be careful with standard 3DGS representation:

- scale may be stored in log-space;
- opacity may be stored as logits;
- quaternion component ordering must be verified.

Do not guess conventions.

Normalize loaded data into one explicit internal representation.

---

# 9. Internal Gaussian representation

Inside `GaussianCloud` always use natural values:

```text
scale:
    linear standard deviation

opacity:
    [0, 1]

rotation:
    normalized quaternion
```

Document quaternion order explicitly.

For example, if choosing:

```text
w, x, y, z
```

use that convention consistently everywhere.

Add assertions where useful.

---

# 10. Gaussian geometry

Gaussian `i` is described by:

```text
mean = μ

Σ = R diag(sx², sy², sz²) Rᵀ
```

where:

```text
R = rotation matrix from quaternion
s = linear Gaussian scales
```

Do not assume a Gaussian has a hard physical surface.

A mathematical Gaussian has infinite support.

---

# 11. Finite support approximation

For BVH acceleration truncate Gaussian influence using:

```text
(x - μ)ᵀ Σ⁻¹ (x - μ) <= k²
```

Use:

```python
gaussian_sigma_cutoff = 3.0
```

by default.

Therefore:

```text
k = 3
k² = 9
```

This defines the region in which a Gaussian is considered relevant to a ray.

---

# 12. Critical interpretation of the 3σ ellipsoid

The `3σ` ellipsoid is **not a surface**.

A ray entering a `3σ` Gaussian does NOT automatically produce a LiDAR return.

The cutoff is only:

- a broad-phase approximation;
- a negligible-contribution cutoff;
- a way to construct finite AABBs;
- a way to make BVH traversal possible.

Conceptually:

```text
q_min > 9
    -> ignore Gaussian

q_min <= 9
    -> evaluate Gaussian contribution
```

---

# 13. Gaussian AABB

For:

```text
Σ = R diag(s²) Rᵀ
```

the world-space half extents of the `kσ` ellipsoid are:

```text
extent_x = k * sqrt(Σxx)
extent_y = k * sqrt(Σyy)
extent_z = k * sqrt(Σzz)
```

Then:

```python
bbox_min = mean - extent
bbox_max = mean + extent
```

Implement this with PyTorch.

Prefer vectorized PyTorch operations.

Avoid unnecessarily constructing `[N, 3, 3]` covariance tensors if the diagonal elements can be calculated directly from rotation matrices and scale.

---

# 14. BVH

Build a binary BVH over Gaussian AABBs.

For the first milestone, CPU BVH construction is acceptable and preferred for simplicity.

Use a straightforward builder:

```text
calculate primitive centroids

choose axis with largest centroid extent

partition around median

recursively split

stop when primitive count <= leaf size
```

Use configurable leaf size, for example:

```text
8
```

Do not implement LBVH or Morton-code construction yet.

---

# 15. BVH representation

After construction convert the BVH into flat PyTorch tensors.

For example:

```python
@dataclass
class FlatBVH:
    bbox_min: torch.Tensor
    bbox_max: torch.Tensor

    left_child: torch.Tensor
    right_child: torch.Tensor

    first_primitive: torch.Tensor
    primitive_count: torch.Tensor

    primitive_indices: torch.Tensor
```

Expected device after GPU setup:

```python
bvh.bbox_min.device.type == "mps"
```

Use GPU-friendly dtypes.

For example:

```text
float32 for geometry
int32 where appropriate for indices
```

Verify what the Metal/PyTorch shader interface supports best.

---

# 16. LiDAR pose

Create:

```python
@dataclass
class LidarPose:
    position: torch.Tensor
    orientation: torch.Tensor
```

Position:

```text
[3]
```

Orientation may be:

```text
quaternion [4]
```

or:

```text
rotation matrix [3,3]
```

Internally choose whatever makes ray generation cleanest.

Document conventions.

---

# 17. LiDAR configuration

Create something similar to:

```python
@dataclass
class LidarConfig:
    azimuth_min: float
    azimuth_max: float

    elevation_min: float
    elevation_max: float

    azimuth_samples: int
    elevation_samples: int

    near: float
    far: float

    gaussian_sigma_cutoff: float = 3.0
    accumulated_alpha_threshold: float = 0.5
```

Use radians internally.

---

# 18. Default LiDAR FOV

Horizontal coordinate is **azimuth**.

Vertical coordinate is **elevation**.

Default:

```text
azimuth:
    [-180°, +180°)
    360° horizontal FOV

elevation:
    [-60°, +60°]
    120° vertical FOV
```

Do not duplicate the `-π` and `+π` ray.

Use endpoint-exclusive horizontal sampling.

---

# 19. Coordinate system

Use:

```text
+X = forward
+Y = left
+Z = up
```

For azimuth `a` and elevation `e`:

```python
direction_local = [
    cos(e) * cos(a),
    cos(e) * sin(a),
    sin(e),
]
```

Then:

```python
direction_world = R_lidar @ direction_local
```

and:

```python
origin_world = lidar_position
```

Write PyTorch tests for the coordinate convention.

---

# 20. PyTorch ray generation

Generate rays using PyTorch.

Prefer something conceptually similar to:

```python
azimuth = torch.linspace(...)
elevation = torch.linspace(...)

elevation_grid, azimuth_grid = torch.meshgrid(
    elevation,
    azimuth,
    indexing="ij",
)
```

Then calculate all directions vectorized.

Output should be something like:

```text
origins:
    [R, 3]

directions:
    [R, 3]
```

where:

```text
R = elevation_samples * azimuth_samples
```

For MPS scans these tensors should live directly on:

```python
device="mps"
```

Do not generate them on CPU and copy them unnecessarily if they can be generated directly on MPS.

---

# 21. Ray/Gaussian math

For a ray:

```text
p(t) = o + t d
```

and Gaussian:

```text
mean = μ
rotation = R
scale = S = diag(sx, sy, sz)
```

transform the ray into normalized Gaussian local space:

```text
o_local = S⁻¹ Rᵀ (o - μ)

d_local = S⁻¹ Rᵀ d
```

Then:

```text
q(t) = |o_local + t d_local|²
```

is squared Mahalanobis distance.

Do not compute a generic matrix inverse for every ray/Gaussian pair.

---

# 22. Gaussian peak along ray

The maximum Gaussian contribution occurs where `q(t)` is minimal.

Calculate:

```text
t_peak =
    -dot(o_local, d_local)
    /
    dot(d_local, d_local)
```

Reject if:

```text
t_peak < near
```

or:

```text
t_peak > far
```

Then:

```text
q_min = q(t_peak)
```

Reject if:

```text
q_min > k²
```

where default:

```text
k² = 9
```

---

# 23. Gaussian opacity contribution

For each valid candidate define:

```text
alpha_i =
    opacity_i * exp(-0.5 * q_min)
```

This represents the Gaussian as one discrete opacity contribution located at `t_peak`.

This approximation is intentional.

Do not implement continuous line integration of Gaussian density in the first milestone.

---

# 24. Multiple Gaussians per ray

A ray may encounter many Gaussian candidates.

The ray MUST NOT terminate at:

- the first BVH candidate;
- the first AABB;
- the first `3σ` ellipsoid;
- the first Gaussian with non-zero opacity.

Instead:

```text
ray
 ↓
BVH
 ↓
candidate Gaussians
 ↓
evaluate t_peak / q_min / alpha
 ↓
sort front-to-back
 ↓
accumulate alpha
 ↓
threshold crossing
 ↓
LiDAR hit
```

---

# 25. Candidate ordering

Candidates must be ordered by:

```text
t_peak ascending
```

before alpha compositing.

Gaussian storage order must have no effect on the result.

This is essential.

---

# 26. Alpha compositing

Initialize:

```python
A = 0.0
```

For every candidate front-to-back:

```python
A = A + (1.0 - A) * alpha_i
```

Equivalent transmittance formulation is acceptable:

```python
T *= 1.0 - alpha_i

A = 1.0 - T
```

Do NOT use:

```python
A += alpha_i
```

---

# 27. Hit threshold

Default:

```python
accumulated_alpha_threshold = 0.5
```

When:

```text
A >= threshold
```

the LiDAR ray returns a point.

Threshold must be configurable.

Make experiments with:

```text
0.3
0.5
0.8
```

easy.

---

# 28. Hit position

Suppose threshold crossing happens while processing Gaussian `i`.

Use:

```text
distance = t_peak_i
```

and:

```python
point = ray_origin + t_peak_i * ray_direction
```

Do not solve for an exact continuous threshold-crossing coordinate inside the Gaussian in the first version.

---

# 29. Example

Suppose:

```text
Gaussian A:
    t_peak = 4.90
    alpha = 0.20

Gaussian B:
    t_peak = 5.00
    alpha = 0.25

Gaussian C:
    t_peak = 5.08
    alpha = 0.30
```

Then:

```text
A0 = 0

A1 = 0.20

A2 =
    0.20 + (1 - 0.20) * 0.25
    = 0.40

A3 =
    0.40 + (1 - 0.40) * 0.30
    = 0.58
```

With threshold:

```text
0.5
```

the LiDAR return is:

```text
distance = 5.08
```

The point lies on the ray at `t_peak` of Gaussian C.

---

# 30. CPU reference implementation

Implement the reference tracer using PyTorch, not NumPy.

Create something like:

```python
class CpuLidarTracer:
    ...
```

It works with:

```python
torch.Tensor(device="cpu")
```

The purpose is correctness, not maximum speed.

---

# 31. Brute-force CPU tracer

Implement a simple brute-force implementation first.

For each ray:

1. evaluate all Gaussians;
2. calculate `t_peak`;
3. calculate `q_min`;
4. reject invalid candidates;
5. calculate alpha;
6. sort by `t_peak`;
7. accumulate alpha;
8. return threshold-crossing point.

This may be slow.

That is acceptable for tiny synthetic test scenes.

It is the mathematical reference.

---

# 32. CPU BVH tracer

After the brute-force implementation works, implement CPU BVH traversal.

Use the same PyTorch scene representation.

Compare:

```text
brute force
vs
BVH
```

before working on Metal.

---

# 33. Metal implementation

Implement:

```python
class MetalLidarTracer:
    ...
```

using PyTorch MPS tensors.

Prefer custom Metal shader compilation directly through PyTorch:

```python
torch.mps.compile_shader(...)
```

No MLX.

No CUDA.

No C++ extension for the first milestone.

---

# 34. Metal shader source

Keep substantial Metal source in a separate `.metal` file if practical:

```text
src/gs_lidar/metal/lidar_trace.metal
```

Python may load it as text:

```python
source = Path(...).read_text()

library = torch.mps.compile_shader(source)
```

or use the correct API required by the installed PyTorch version.

Do not embed hundreds of lines of Metal shader source into an unrelated Python module unless necessary.

---

# 35. PyTorch-to-Metal buffers

The Metal kernel should receive buffers backed directly by PyTorch MPS tensors.

Conceptually inputs include:

```text
Gaussian data:

means
scales
rotations
opacities
```

BVH:

```text
bbox_min
bbox_max
left_child
right_child
first_primitive
primitive_count
primitive_indices
```

Rays or LiDAR parameters:

```text
ray origins
ray directions
```

Outputs:

```text
ranges
points
hit_mask
gaussian_ids
accumulated_alpha
```

Avoid MPS → CPU → Metal intermediate copies.

---

# 36. One Metal thread per LiDAR ray

Conceptually:

```text
one Metal thread
    =
one LiDAR ray
```

Each thread performs:

```text
BVH traversal
    ↓
candidate collection
    ↓
ray/Gaussian evaluation
    ↓
candidate sorting
    ↓
alpha accumulation
    ↓
threshold crossing
    ↓
write result
```

This is the baseline design.

Optimize only after correctness is established.

---

# 37. Ray/AABB test

Implement an efficient slab-style ray/AABB intersection test in Metal.

BVH nodes that cannot intersect the ray between:

```text
near
far
```

must be skipped.

Near-first traversal is desirable but does not eliminate the requirement to correctly order final Gaussian contributions.

---

# 38. GPU candidate collection

For the first Metal implementation it is acceptable to collect a bounded number of candidates per ray.

Conceptually:

```text
candidate_t[MAX_CANDIDATES]

candidate_alpha[MAX_CANDIDATES]

candidate_gaussian_id[MAX_CANDIDATES]
```

Investigate Metal private/thread memory constraints before choosing `MAX_CANDIDATES`.

Do not allocate an absurdly large per-thread array blindly.

---

# 39. Candidate overflow

Candidate overflow must not silently corrupt results.

If candidate capacity is exceeded:

```text
mark ray as overflowed
```

and increment/report:

```text
candidate_overflow_count
```

Provide debug information about it.

Later implementations may remove the fixed candidate limit.

For milestone 1 explicit detection is sufficient.

---

# 40. BVH stack

A fixed-size Metal traversal stack is acceptable initially.

For example:

```text
stack[64]
```

or another justified capacity.

Again:

```text
stack overflow
```

must be detected.

Never silently discard BVH nodes.

---

# 41. Candidate sorting in Metal

Candidates must be processed front-to-back by `t_peak`.

For milestone 1 use a simple per-ray local sorting method appropriate for relatively small candidate lists.

Possible algorithms:

```text
insertion sort

small bitonic sort
```

Choose based on simplicity.

Do not build a global GPU sorting pipeline.

---

# 42. LidarScan

Create:

```python
@dataclass
class LidarScan:
    ranges: torch.Tensor
    points: torch.Tensor
    hit_mask: torch.Tensor
    gaussian_ids: torch.Tensor
    accumulated_alpha: torch.Tensor
```

Keep organized scan dimensions.

For example:

```text
ranges:
    [E, A]

points:
    [E, A, 3]

hit_mask:
    [E, A]

gaussian_ids:
    [E, A]

accumulated_alpha:
    [E, A]
```

where:

```text
E = elevation_samples
A = azimuth_samples
```

---

# 43. Miss representation

Use a consistent convention.

For example:

```text
range:
    inf

point:
    nan, nan, nan

gaussian_id:
    -1

hit_mask:
    false
```

---

# 44. Flattened point cloud

Provide:

```python
def valid_points(self) -> torch.Tensor:
    ...
```

returning:

```text
[M, 3]
```

using the same device as the scan.

For example:

```python
points = scan.valid_points()

assert points.device == scan.points.device
```

Do not call `.cpu()` inside this function.

---

# 45. Differentiability

Do not make differentiability a requirement for the Metal tracer.

BVH traversal, threshold crossing, candidate sorting and discrete hit selection are inherently non-trivial for autograd.

The initial project is a forward simulation tool.

However, keep PyTorch tensors as the API because this provides convenient downstream processing and interoperability.

---

# 46. Rerun integration

Rerun is strictly a visualization/debugging client.

The core modules:

```text
GaussianCloud
BVH
LiDAR
tracer
LidarScan
```

must not depend on Rerun.

Only `rerun_viewer.py` should know about Rerun.

---

# 47. Rerun and CPU transfer

If Rerun requires CPU/NumPy data, perform the conversion only inside the visualization boundary.

For example:

```python
points_cpu = (
    scan.valid_points()
    .detach()
    .cpu()
    .numpy()
)
```

This copy is acceptable because it is explicitly for visualization.

Do not let Rerun requirements affect the computational API.

---

# 48. Gaussian visualization

Use current Rerun Gaussian splatting support if available.

Prefer:

```python
rr.GaussianSplats3D(...)
```

using:

- means;
- linear scales;
- rotations;
- opacity/color;
- optionally SH coefficients.

If Gaussian splat visualization is unstable or unavailable in the installed Rerun version, isolate the fallback inside `rerun_viewer.py`.

Possible fallback:

```text
ellipsoids
or
points
```

The LiDAR simulation itself must remain unchanged.

---

# 49. LiDAR visualization

Draw the sensor at its world pose.

Use a small cylinder.

Optionally display its coordinate axes:

```text
+X forward
+Y left
+Z up
```

Its orientation must match actual ray generation.

---

# 50. Point cloud visualization

Visualize:

```python
scan.valid_points()
```

through:

```python
rr.Points3D
```

Use a distinct appearance from the Gaussian scene.

Do not draw misses.

---

# 51. Experiment 01

Create:

```text
experiments/01_basic_scan.py
```

Invocation:

```bash
uv run python experiments/01_basic_scan.py scene.ply
```

The experiment must:

1. load the PLY;
2. convert data to PyTorch tensors;
3. normalize Gaussian parameters;
4. print scene statistics;
5. build `3σ` Gaussian AABBs;
6. build CPU BVH;
7. place LiDAR at `(0, 0, 0)` by default;
8. use identity orientation;
9. transfer the static scene and BVH to MPS;
10. generate LiDAR rays;
11. run the selected backend;
12. obtain `LidarScan`;
13. visualize everything in Rerun.

---

# 52. Experiment CLI

Support:

```text
--backend cpu|metal

--azimuth-samples
--elevation-samples

--position X Y Z

--yaw
--pitch
--roll

--near
--far

--sigma-cutoff
--alpha-threshold
```

Use `argparse`.

Document Euler-angle convention and order clearly.

---

# 53. Default parameters

Use:

```text
LiDAR position:
    (0, 0, 0)

orientation:
    identity

azimuth:
    [-180°, +180°)

elevation:
    [-60°, +60°]

sigma cutoff:
    3.0

accumulated alpha threshold:
    0.5
```

Start development with low angular resolution.

---

# 54. Tests: ray directions

Verify:

```text
azimuth = 0
elevation = 0
```

produces:

```text
[1, 0, 0]
```

Verify:

```text
azimuth = +90°
elevation = 0
```

produces:

```text
[0, 1, 0]
```

Test positive/negative elevation.

Test orientation transforms.

Test the azimuth seam.

---

# 55. Tests: one Gaussian

Place one isotropic Gaussian directly in front of LiDAR.

Verify:

- expected `t_peak`;
- expected `q_min`;
- expected alpha;
- return only if accumulated alpha exceeds threshold.

Being inside `3σ` alone must not count as a hit.

---

# 56. Tests: transparent foreground

This test is mandatory.

Put:

```text
Gaussian A:
    close
    opacity contribution = very low

Gaussian B:
    behind A
    strong contribution
```

The ray must pass through A.

It must not terminate simply because A was encountered first.

---

# 57. Tests: cumulative opacity

Use three contributions such as:

```text
0.20
0.25
0.30
```

Expected:

```text
A1 = 0.20
A2 = 0.40
A3 = 0.58
```

For threshold:

```text
0.5
```

the third Gaussian produces the LiDAR point.

---

# 58. Tests: candidate ordering

Create Gaussians in scrambled memory order.

Example:

```text
t_peak:
    5
    2
    4
    3
```

Verify that output equals output for correctly sorted geometric order.

Storage order must not affect the scan.

---

# 59. Tests: anisotropic Gaussian

Verify behavior for different:

```text
sx
sy
sz
```

Test rays along different directions.

---

# 60. Tests: rotated anisotropic Gaussian

Rotate an elongated Gaussian.

Verify local-space ray transformation and quaternion handling.

---

# 61. Tests: sigma cutoff

Test candidates immediately inside and outside:

```text
q_min = 9
```

for:

```text
k = 3
```

Verify cutoff behavior.

Also verify again that `3σ` entry itself is not a LiDAR hit.

---

# 62. CPU brute force vs CPU BVH

For small deterministic/random scenes compare:

```text
PyTorch brute-force CPU
vs
PyTorch BVH CPU
```

Compare:

- hit mask;
- range;
- Gaussian ID;
- accumulated alpha.

---

# 63. CPU vs Metal

Compare:

```text
CpuLidarTracer
vs
MetalLidarTracer
```

For each synthetic test.

Use:

```python
torch.testing.assert_close(...)
```

where appropriate.

Metal outputs may need:

```python
metal_tensor.cpu()
```

only in the test comparison boundary.

---

# 64. Benchmarking

Measure:

```text
PLY load
Gaussian normalization
AABB generation
BVH build
CPU→MPS scene upload
ray generation
CPU trace
Metal trace
```

Keep scene-upload timing separate from per-scan timing.

This distinction matters because scene/BVH upload should normally happen only once.

---

# 65. Metal synchronization

Make sure benchmark timing waits for MPS work to finish.

Use the appropriate PyTorch MPS synchronization API.

Do not report asynchronous dispatch time as complete tracing time.

---

# 66. Benchmark statistics

Print:

```text
number of Gaussians
number of BVH nodes

azimuth samples
elevation samples
number of rays

number of hits
hit percentage

average candidate count
maximum candidate count

candidate overflow count
BVH stack overflow count

trace time
MRays/s
```

where practical.

---

# 67. Expected long-lived architecture

The intended usage is not necessarily:

```text
load scene
build BVH
scan once
exit
```

The module should support:

```python
scene = load_scene(...)
bvh = build_bvh(scene)

simulator = LidarSimulator(
    scene=scene.to("mps"),
    bvh=bvh.to("mps"),
)

scan1 = simulator.scan(pose1, config)
scan2 = simulator.scan(pose2, config)
scan3 = simulator.scan(pose3, config)
```

The Gaussian scene and BVH stay resident on MPS.

Only LiDAR pose/configuration changes.

This is an important architectural goal.

---

# 68. Public output must remain PyTorch

The central output must NOT be:

```text
NumPy array
Python list
Rerun object
```

It must be:

```python
torch.Tensor
```

For example:

```python
points = simulator.scan(
    pose,
    config,
).valid_points()

print(points.shape)
print(points.device)
```

Expected:

```text
torch.Size([M, 3])
mps:0
```

or the appropriate PyTorch representation of the MPS device.

---

# 69. Relationship to a 3DGS renderer

The project intentionally borrows the idea of front-to-back opacity compositing from a Gaussian splatting renderer.

A rasterizer conceptually has contributions:

```text
alpha_i
T_i
weight_i = T_i * alpha_i
```

This project asks a different question:

```text
Along this 3D world-space ray,
at what distance does accumulated Gaussian opacity
become large enough to represent a surface?
```

Therefore this is not literally camera rasterization.

It is a ray-space interpretation of Gaussian opacity.

Explain this distinction in README.

---

# 70. Conceptual definition of the resulting surface

The generated point cloud represents the surface:

> the first point along each LiDAR ray at which front-to-back accumulated contributions of nearby 3D Gaussians exceed the configured opacity threshold.

The `3σ` ellipsoid only limits which Gaussians participate.

This definition should be prominent in the README.

---

# 71. Implementation order

Implement in this order:

1. create `uv` project;
2. install PyTorch and dependencies;
3. implement PLY loader;
4. convert loader output to PyTorch;
5. implement `GaussianCloud`;
6. establish quaternion convention;
7. implement Gaussian geometry using PyTorch;
8. implement LiDAR ray generation using PyTorch;
9. implement ray → Gaussian local transformation;
10. implement `t_peak`;
11. implement `q_min`;
12. implement Gaussian alpha contribution;
13. implement brute-force CPU single-ray tracer;
14. test cumulative alpha;
15. implement brute-force CPU complete scan;
16. implement Gaussian AABBs;
17. implement CPU BVH builder;
18. flatten BVH into PyTorch tensors;
19. implement CPU BVH traversal;
20. compare CPU BVH with brute force;
21. implement `LidarScan`;
22. investigate the installed PyTorch version's `torch.mps.compile_shader` API;
23. create a minimal custom Metal test kernel using MPS tensors;
24. verify Python → MPS tensor → Metal → MPS tensor round trip;
25. implement ray/AABB Metal function;
26. implement Metal BVH traversal;
27. implement Metal ray/Gaussian evaluation;
28. implement Metal candidate collection;
29. implement candidate sorting;
30. implement alpha accumulation;
31. compare CPU and Metal;
32. implement Rerun visualization;
33. implement Experiment 01;
34. benchmark;
35. finish README.

Do not start the full Metal tracer before the tiny `compile_shader` smoke test works.

---

# 72. Very important Metal smoke test

Before implementing BVH traversal, create a tiny experiment proving that the installed PyTorch can:

1. create an MPS tensor;
2. pass it directly to a custom Metal shader compiled through PyTorch;
3. write into another MPS tensor;
4. read the result back from PyTorch.

For example conceptually:

```python
x = torch.arange(
    1024,
    device="mps",
    dtype=torch.float32,
)

y = torch.empty_like(x)

kernel(x, y)

torch.testing.assert_close(
    y.cpu(),
    x.cpu() * 2,
)
```

The exact invocation API must follow the installed PyTorch version.

This smoke test establishes the technical foundation before building the full tracer.

---

# 73. Do not prematurely implement

Do not add yet:

- CUDA;
- MLX;
- C++ extension;
- Objective-C++ extension;
- GPU BVH construction;
- LBVH;
- Morton sorting;
- mesh extraction;
- training;
- differentiability;
- physical laser reflectance;
- noise;
- beam divergence;
- multi-return LiDAR;
- continuous volume integration;
- global GPU sorting.

Keep milestone 1 focused.

---

# 74. Future experiments

Document possible future work:

```text
alpha threshold:
    0.3
    0.5
    0.8

sigma cutoff:
    2σ
    3σ
    4σ

continuous line integration

multi-return LiDAR

real LiDAR scan patterns

beam divergence

sensor noise

intensity estimation

GPU BVH construction

priority-queue BVH traversal

sorting-free front-to-back traversal
```

---

# 75. Definition of done

Milestone 1 is complete when:

```bash
uv run python experiments/01_basic_scan.py scene.ply --backend metal
```

opens Rerun and displays:

1. the original Gaussian scene;
2. a cylinder representing the LiDAR;
3. the generated LiDAR cloud.

The LiDAR point cloud must have been computed using:

```text
PyTorch MPS tensors
+
custom Metal shader compiled through PyTorch
+
BVH traversal
+
multiple Gaussian candidates per ray
+
front-to-back candidate ordering
+
alpha compositing
+
accumulated opacity threshold
```

The Metal kernel must not terminate at the first Gaussian.

A transparent foreground Gaussian must allow the ray to continue.

A CPU PyTorch reference implementation must exist.

Synthetic tests must show:

```text
CPU reference ≈ Metal result
```

The public scan result must remain PyTorch tensors.

The final valid point cloud should be obtainable as:

```python
points = scan.valid_points()
```

with:

```text
points.shape == [M, 3]
```

and for the Metal backend:

```text
points.device == MPS
```

Correctness, modularity, and educational clarity are more important than maximum performance for the first milestone.