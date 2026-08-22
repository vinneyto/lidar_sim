"""GPU SIFT detector backed by handwritten Metal kernels on PyTorch MPS."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.resources import files

import torch


@dataclass(frozen=True)
class SiftDebugStats:
    """Small CPU-side diagnostic summary for one detector invocation."""

    candidate_count: int
    spatial_survivor_count: int
    selected_count: int
    strongest_abs_response: float
    cutoff_abs_response: float
    first_rejected_abs_response: float | None
    boundary_gap: float | None
    near_cutoff_count: int

    @property
    def near_cutoff_fraction(self) -> float:
        if self.selected_count == 0:
            return 0.0
        return self.near_cutoff_count / self.selected_count


@dataclass(frozen=True)
class SiftFeatures:
    """Detected SIFT keypoints kept on the input image's MPS device."""

    keypoints_xy: torch.Tensor
    responses: torch.Tensor
    scales: torch.Tensor
    orientations: torch.Tensor
    candidate_overflow: int = 0
    debug: SiftDebugStats | None = None

    @property
    def count(self) -> int:
        return int(self.keypoints_xy.shape[0])


class MetalSiftDetector:
    """SIFT detector whose image-processing stages are handwritten Metal kernels.

    The custom kernels perform RGB-to-gray conversion, separable Gaussian
    filtering, octave downsampling, Difference-of-Gaussian construction,
    26-neighbour extrema detection, iterative 3D quadratic sub-pixel/scale
    localization, contrast rejection, Hessian edge rejection, dominant
    orientation estimation, and spatial non-maximum suppression. Only final
    top-k selection uses a native PyTorch MPS operation.

    The detector intentionally stops before the 128-dimensional SIFT descriptor:
    the camera experiment only needs stable feature locations to visualize.
    """

    def __init__(
        self,
        num_features: int = 512,
        *,
        scales_per_octave: int = 3,
        sigma0: float = 1.6,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10.0,
        max_octaves: int = 4,
        max_candidates: int = 32768,
        spatial_nms_radius: float = 7.0,
        debug: bool = False,
    ) -> None:
        if num_features < 1:
            raise ValueError("num_features must be positive")
        if scales_per_octave != 3:
            raise ValueError(
                "the current Metal refinement kernel is specialized for 3 scales per octave"
            )
        if sigma0 <= 0:
            raise ValueError("sigma0 must be positive")
        if contrast_threshold <= 0:
            raise ValueError("contrast_threshold must be positive")
        if edge_threshold <= 0:
            raise ValueError("edge_threshold must be positive")
        if max_octaves < 1:
            raise ValueError("max_octaves must be positive")
        if max_candidates < num_features:
            raise ValueError("max_candidates must be at least num_features")
        if spatial_nms_radius <= 0:
            raise ValueError("spatial_nms_radius must be positive")
        if not torch.backends.mps.is_available() or not hasattr(
            torch.mps, "compile_shader"
        ):
            raise RuntimeError("MetalSiftDetector requires MPS and torch.mps.compile_shader")

        self.num_features = num_features
        self.scales_per_octave = scales_per_octave
        self.sigma0 = sigma0
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.max_octaves = max_octaves
        self.max_candidates = max_candidates
        self.spatial_nms_radius = spatial_nms_radius
        self.debug = debug

        metal_files = files("metal_sift.metal")
        source = metal_files.joinpath("sift.metal").read_text()
        spatial_nms_source = metal_files.joinpath("spatial_nms.metal").read_text()
        self._kernels = torch.mps.compile_shader(source)
        self._spatial_kernels = torch.mps.compile_shader(spatial_nms_source)

    @staticmethod
    def _i32(value: int) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.int32, device="mps")

    @staticmethod
    def _f32(value: float) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float32, device="mps")

    def _dispatch(
        self,
        kernel,
        args: tuple[torch.Tensor, ...],
        *,
        threads: int,
        keepalive: list[torch.Tensor],
    ) -> None:
        keepalive.extend(args)
        kernel(*args, threads=threads)

    def _blur(
        self,
        image: torch.Tensor,
        width: int,
        height: int,
        sigma: float,
        keepalive: list[torch.Tensor],
    ) -> torch.Tensor:
        if sigma <= 1e-4:
            return image
        radius = max(1, min(16, math.ceil(3.0 * sigma)))
        temporary = torch.empty_like(image)
        output = torch.empty_like(image)
        width_t = self._i32(width)
        height_t = self._i32(height)
        sigma_t = self._f32(sigma)
        radius_t = self._i32(radius)
        count = width * height
        self._dispatch(
            self._kernels.gaussian_horizontal,
            (image, temporary, width_t, height_t, sigma_t, radius_t),
            threads=count,
            keepalive=keepalive,
        )
        self._dispatch(
            self._kernels.gaussian_vertical,
            (temporary, output, width_t, height_t, sigma_t, radius_t),
            threads=count,
            keepalive=keepalive,
        )
        return output

    def _dog(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
        count: int,
        keepalive: list[torch.Tensor],
    ) -> torch.Tensor:
        output = torch.empty_like(lower)
        self._dispatch(
            self._kernels.difference_of_gaussians,
            (lower, upper, output, self._i32(count)),
            threads=count,
            keepalive=keepalive,
        )
        return output

    def _downsample(
        self,
        image: torch.Tensor,
        width: int,
        height: int,
        keepalive: list[torch.Tensor],
    ) -> tuple[torch.Tensor, int, int]:
        next_width = max(1, width // 2)
        next_height = max(1, height // 2)
        output = torch.empty(next_width * next_height, device="mps", dtype=torch.float32)
        self._dispatch(
            self._kernels.downsample_half,
            (
                image,
                output,
                self._i32(width),
                self._i32(height),
                self._i32(next_width),
                self._i32(next_height),
            ),
            threads=next_width * next_height,
            keepalive=keepalive,
        )
        return output, next_width, next_height

    def _debug_stats(
        self,
        *,
        candidate_count: int,
        spatial_survivor_count: int,
        selected_values: torch.Tensor,
        first_rejected_value: torch.Tensor | None,
    ) -> SiftDebugStats:
        """Read only a few ranking scalars needed to diagnose top-k churn."""
        selected_count = int(selected_values.shape[0])
        if selected_count == 0:
            return SiftDebugStats(
                candidate_count=candidate_count,
                spatial_survivor_count=spatial_survivor_count,
                selected_count=0,
                strongest_abs_response=0.0,
                cutoff_abs_response=0.0,
                first_rejected_abs_response=None,
                boundary_gap=None,
                near_cutoff_count=0,
            )

        strongest = selected_values[0]
        cutoff = selected_values[-1]
        # A large population whose response is within 5% of the cutoff means
        # tiny frame-to-frame response changes can reshuffle many features around
        # rank num_features even when the underlying extrema remain stable.
        near_cutoff = (selected_values <= cutoff * 1.05).sum()
        torch.mps.synchronize()

        strongest_value = float(strongest.item())
        cutoff_value = float(cutoff.item())
        rejected_value = (
            float(first_rejected_value.item()) if first_rejected_value is not None else None
        )
        boundary_gap = cutoff_value - rejected_value if rejected_value is not None else None
        return SiftDebugStats(
            candidate_count=candidate_count,
            spatial_survivor_count=spatial_survivor_count,
            selected_count=selected_count,
            strongest_abs_response=strongest_value,
            cutoff_abs_response=cutoff_value,
            first_rejected_abs_response=rejected_value,
            boundary_gap=boundary_gap,
            near_cutoff_count=int(near_cutoff.item()),
        )

    def detect(self, image: torch.Tensor) -> SiftFeatures:
        """Detect up to ``num_features`` strongest SIFT keypoints in an RGB image.

        ``image`` must be a contiguous-compatible float32 MPS tensor shaped
        ``[H, W, C]`` with at least three channels. The returned tensors remain
        on MPS so callers only need to copy the small final keypoint set to CPU.
        """
        if image.device.type != "mps":
            raise ValueError("image must already reside on the MPS device")
        if image.dtype != torch.float32:
            raise ValueError("image must have dtype torch.float32")
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("image must have shape [H, W, C] with at least 3 channels")

        height, width, channels = map(int, image.shape)
        if min(width, height) < 16:
            raise ValueError("image must be at least 16x16 pixels")

        image = image.contiguous()
        keepalive: list[torch.Tensor] = [image]
        pixel_count = width * height
        gray = torch.empty(pixel_count, device="mps", dtype=torch.float32)
        self._dispatch(
            self._kernels.rgb_to_gray,
            (
                image,
                gray,
                self._i32(pixel_count),
                self._i32(channels),
            ),
            threads=pixel_count,
            keepalive=keepalive,
        )

        out_x = torch.empty(self.max_candidates, device="mps", dtype=torch.float32)
        out_y = torch.empty_like(out_x)
        out_scale = torch.empty_like(out_x)
        out_response = torch.empty_like(out_x)
        out_orientation = torch.empty_like(out_x)
        counter = torch.zeros(1, device="mps", dtype=torch.int32)
        overflow = torch.zeros(1, device="mps", dtype=torch.int32)
        keepalive.extend(
            [out_x, out_y, out_scale, out_response, out_orientation, counter, overflow]
        )

        k = 2.0 ** (1.0 / self.scales_per_octave)
        gaussian_level_count = self.scales_per_octave + 3
        octave_base = gray
        octave_width = width
        octave_height = height

        for octave in range(self.max_octaves):
            if min(octave_width, octave_height) < 16:
                break

            if octave == 0:
                assumed_input_sigma = 0.5
                initial_sigma = math.sqrt(
                    max(self.sigma0 * self.sigma0 - assumed_input_sigma**2, 1e-8)
                )
                gaussian_levels = [
                    self._blur(
                        octave_base,
                        octave_width,
                        octave_height,
                        initial_sigma,
                        keepalive,
                    )
                ]
            else:
                gaussian_levels = [octave_base]

            for level in range(1, gaussian_level_count):
                previous_sigma = self.sigma0 * (k ** (level - 1))
                target_sigma = self.sigma0 * (k**level)
                incremental_sigma = math.sqrt(
                    max(target_sigma * target_sigma - previous_sigma * previous_sigma, 1e-8)
                )
                gaussian_levels.append(
                    self._blur(
                        gaussian_levels[-1],
                        octave_width,
                        octave_height,
                        incremental_sigma,
                        keepalive,
                    )
                )

            octave_count = octave_width * octave_height
            dogs = [
                self._dog(
                    gaussian_levels[level],
                    gaussian_levels[level + 1],
                    octave_count,
                    keepalive,
                )
                for level in range(gaussian_level_count - 1)
            ]
            if len(dogs) != 5:
                raise RuntimeError("the current Metal localization kernel expects five DoG levels")

            coordinate_scale = float(1 << octave)
            # Every localization dispatch gets all five DoG levels. This lets a
            # candidate move into a neighbouring scale and repeat its quadratic
            # fit instead of being discarded when the fitted scale crosses a
            # half-sample boundary.
            for dog_level in range(1, self.scales_per_octave + 1):
                self._dispatch(
                    self._kernels.detect_extrema,
                    (
                        dogs[0],
                        dogs[1],
                        dogs[2],
                        dogs[3],
                        dogs[4],
                        gaussian_levels[1],
                        gaussian_levels[2],
                        gaussian_levels[3],
                        out_x,
                        out_y,
                        out_scale,
                        out_response,
                        out_orientation,
                        counter,
                        overflow,
                        self._i32(octave_width),
                        self._i32(octave_height),
                        self._f32(self.contrast_threshold / self.scales_per_octave),
                        self._f32(self.edge_threshold),
                        self._f32(self.sigma0),
                        self._f32(k),
                        self._f32(coordinate_scale),
                        self._i32(dog_level),
                        self._i32(self.max_candidates),
                    ),
                    threads=octave_count,
                    keepalive=keepalive,
                )

            if octave + 1 >= self.max_octaves:
                break
            octave_base, octave_width, octave_height = self._downsample(
                gaussian_levels[self.scales_per_octave],
                octave_width,
                octave_height,
                keepalive,
            )

        # All image processing above is asynchronous custom Metal work. One sync
        # here both protects temporary buffers from allocator reuse and obtains
        # the compact candidate count required for spatial suppression.
        torch.mps.synchronize()
        candidate_count = min(int(counter.item()), self.max_candidates)
        candidate_overflow = int(overflow.item())

        if candidate_count == 0:
            empty_points = torch.empty((0, 2), device="mps", dtype=torch.float32)
            empty = torch.empty(0, device="mps", dtype=torch.float32)
            debug_stats = (
                SiftDebugStats(
                    candidate_count=0,
                    spatial_survivor_count=0,
                    selected_count=0,
                    strongest_abs_response=0.0,
                    cutoff_abs_response=0.0,
                    first_rejected_abs_response=None,
                    boundary_gap=None,
                    near_cutoff_count=0,
                )
                if self.debug
                else None
            )
            return SiftFeatures(
                keypoints_xy=empty_points,
                responses=empty,
                scales=empty,
                orientations=empty,
                candidate_overflow=candidate_overflow,
                debug=debug_stats,
            )

        # Kornia's MultiResolutionDetector uses a 15x15 2D NMS window on each
        # pyramid level. Our detector already has scale-space NMS, but its final
        # candidate pool can still contain several nearby extrema from different
        # scales/octaves. Suppress those competitors in original-image coordinates
        # before global top-k; a 7 px radius approximates half of Kornia's window.
        spatial_scores = torch.empty(candidate_count, device="mps", dtype=torch.float32)
        self._dispatch(
            self._spatial_kernels.suppress_nearby_candidates,
            (
                out_x[:candidate_count],
                out_y[:candidate_count],
                out_response[:candidate_count],
                spatial_scores,
                self._i32(candidate_count),
                self._f32(self.spatial_nms_radius),
            ),
            threads=candidate_count,
            keepalive=keepalive,
        )
        survivor_count_tensor = (spatial_scores > 0).sum()
        torch.mps.synchronize()
        spatial_survivor_count = int(survivor_count_tensor.item())

        if spatial_survivor_count == 0:
            empty_points = torch.empty((0, 2), device="mps", dtype=torch.float32)
            empty = torch.empty(0, device="mps", dtype=torch.float32)
            debug_stats = (
                SiftDebugStats(
                    candidate_count=candidate_count,
                    spatial_survivor_count=0,
                    selected_count=0,
                    strongest_abs_response=0.0,
                    cutoff_abs_response=0.0,
                    first_rejected_abs_response=None,
                    boundary_gap=None,
                    near_cutoff_count=0,
                )
                if self.debug
                else None
            )
            return SiftFeatures(
                keypoints_xy=empty_points,
                responses=empty,
                scales=empty,
                orientations=empty,
                candidate_overflow=candidate_overflow,
                debug=debug_stats,
            )

        responses = out_response[:candidate_count]
        keep_count = min(self.num_features, spatial_survivor_count)
        request_extra = self.debug and spatial_survivor_count > keep_count
        ranked_count = keep_count + 1 if request_extra else keep_count
        ranked = torch.topk(spatial_scores, ranked_count, largest=True, sorted=True)
        indices = ranked.indices[:keep_count]
        selected_values = ranked.values[:keep_count]
        first_rejected_value = ranked.values[keep_count] if request_extra else None

        debug_stats = (
            self._debug_stats(
                candidate_count=candidate_count,
                spatial_survivor_count=spatial_survivor_count,
                selected_values=selected_values,
                first_rejected_value=first_rejected_value,
            )
            if self.debug
            else None
        )

        return SiftFeatures(
            keypoints_xy=torch.stack(
                (out_x[:candidate_count][indices], out_y[:candidate_count][indices]), dim=1
            ),
            responses=responses[indices],
            scales=out_scale[:candidate_count][indices],
            orientations=out_orientation[:candidate_count][indices],
            candidate_overflow=candidate_overflow,
            debug=debug_stats,
        )
