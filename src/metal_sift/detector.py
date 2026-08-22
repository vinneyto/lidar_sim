"""Kornia-style multi-resolution DoG detector backed by handwritten Metal kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.resources import files

import torch


@dataclass(frozen=True)
class SiftDebugStats:
    """Diagnostic summary for the Kornia-style multi-resolution detector."""

    candidate_count: int
    per_level_candidate_counts: tuple[int, ...]
    pyramid_preselected_count: int
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
    """Detected keypoints kept on the input image's MPS device."""

    keypoints_xy: torch.Tensor
    responses: torch.Tensor
    scales: torch.Tensor
    orientations: torch.Tensor
    candidate_overflow: int = 0
    debug: SiftDebugStats | None = None

    @property
    def count(self) -> int:
        return int(self.keypoints_xy.shape[0])


@dataclass(frozen=True)
class _PyramidLevel:
    scores: torch.Tensor
    width: int
    height: int
    factor_x: float
    factor_y: float
    quota: int


class MetalSiftDetector:
    """Port the detector stage used by the repository's Kornia SIFT experiment.

    This intentionally mirrors ``MultiResolutionDetector(BlobDoGSingle(1.0, 1.6))``
    rather than classical 3D SIFT scale-space extrema localization. The pipeline is:

    RGB -> gray -> six image-pyramid levels -> DoG(1.0, 1.6) -> positive 15x15
    2D NMS -> Kornia-style per-level quotas -> global top-k.

    Resize, pyramid smoothing, Gaussian filters, DoG and NMS are handwritten Metal.
    Only the small per-level and final top-k operations use native PyTorch MPS.
    The Kornia comparison used PassLAF for orientation and affine estimation, so
    orientations returned here are zeros and no 128D descriptor is computed.
    """

    _PYRAMID_LEVELS = 4
    _UPSCALE_LEVELS = 1
    _SCALE_FACTOR = math.sqrt(2.0)
    _MR_SIZE = 22.0
    _NMS_BORDER = 15

    def __init__(self, num_features: int = 512, *, debug: bool = False) -> None:
        if num_features < 1:
            raise ValueError("num_features must be positive")
        if not torch.backends.mps.is_available() or not hasattr(
            torch.mps, "compile_shader"
        ):
            raise RuntimeError("MetalSiftDetector requires MPS and torch.mps.compile_shader")

        self.num_features = num_features
        self.debug = debug
        source = files("metal_sift.metal").joinpath("kornia_detector.metal").read_text()
        self._kernels = torch.mps.compile_shader(source)

    @staticmethod
    def _i32(value: int) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.int32, device="mps")

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

    def _resize(
        self,
        image: torch.Tensor,
        src_width: int,
        src_height: int,
        dst_width: int,
        dst_height: int,
        keepalive: list[torch.Tensor],
    ) -> torch.Tensor:
        output = torch.empty(dst_width * dst_height, device="mps", dtype=torch.float32)
        self._dispatch(
            self._kernels.resize_bilinear,
            (
                image,
                output,
                self._i32(src_width),
                self._i32(src_height),
                self._i32(dst_width),
                self._i32(dst_height),
            ),
            threads=dst_width * dst_height,
            keepalive=keepalive,
        )
        return output

    def _fixed_gaussian(
        self,
        image: torch.Tensor,
        width: int,
        height: int,
        horizontal_kernel,
        vertical_kernel,
        keepalive: list[torch.Tensor],
    ) -> torch.Tensor:
        """Run one of the two precomputed separable BlobDoGSingle Gaussians."""
        temporary = torch.empty_like(image)
        output = torch.empty_like(image)
        width_t = self._i32(width)
        height_t = self._i32(height)
        count = width * height
        self._dispatch(
            horizontal_kernel,
            (image, temporary, width_t, height_t),
            threads=count,
            keepalive=keepalive,
        )
        self._dispatch(
            vertical_kernel,
            (temporary, output, width_t, height_t),
            threads=count,
            keepalive=keepalive,
        )
        return output

    def _pyrdown(
        self,
        image: torch.Tensor,
        width: int,
        height: int,
        keepalive: list[torch.Tensor],
    ) -> tuple[torch.Tensor, int, int]:
        """Match Kornia pyrdown: fixed 5x5 Gaussian blur, then bilinear resize."""
        count = width * height
        temporary = torch.empty_like(image)
        blurred = torch.empty_like(image)
        width_t = self._i32(width)
        height_t = self._i32(height)
        self._dispatch(
            self._kernels.pyramid_blur5_horizontal,
            (image, temporary, width_t, height_t),
            threads=count,
            keepalive=keepalive,
        )
        self._dispatch(
            self._kernels.pyramid_blur5_vertical,
            (temporary, blurred, width_t, height_t),
            threads=count,
            keepalive=keepalive,
        )

        next_width = max(1, int(width / self._SCALE_FACTOR))
        next_height = max(1, int(height / self._SCALE_FACTOR))
        return (
            self._resize(
                blurred,
                width,
                height,
                next_width,
                next_height,
                keepalive,
            ),
            next_width,
            next_height,
        )

    def _dog_nms(
        self,
        image: torch.Tensor,
        width: int,
        height: int,
        level_counts: torch.Tensor,
        level_index: int,
        keepalive: list[torch.Tensor],
    ) -> torch.Tensor:
        """Run BlobDoGSingle(1.0, 1.6) followed by positive 15x15 NMS."""
        count = width * height
        blur1 = self._fixed_gaussian(
            image,
            width,
            height,
            self._kernels.gaussian_sigma1_horizontal,
            self._kernels.gaussian_sigma1_vertical,
            keepalive,
        )
        blur2 = self._fixed_gaussian(
            image,
            width,
            height,
            self._kernels.gaussian_sigma2_horizontal,
            self._kernels.gaussian_sigma2_vertical,
            keepalive,
        )
        response = torch.empty_like(image)
        horizontal_max = torch.empty_like(image)
        nms_response = torch.empty_like(image)
        width_t = self._i32(width)
        height_t = self._i32(height)
        border_t = self._i32(self._NMS_BORDER)

        self._dispatch(
            self._kernels.difference_of_gaussians,
            (blur1, blur2, response, self._i32(count)),
            threads=count,
            keepalive=keepalive,
        )
        self._dispatch(
            self._kernels.max_filter15_horizontal,
            (response, horizontal_max, width_t, height_t, border_t),
            threads=count,
            keepalive=keepalive,
        )
        self._dispatch(
            self._kernels.nms15_positive_vertical,
            (
                response,
                horizontal_max,
                nms_response,
                level_counts,
                self._i32(level_index),
                width_t,
                height_t,
                border_t,
            ),
            threads=count,
            keepalive=keepalive,
        )
        return nms_response

    def _kornia_quotas(self) -> list[int]:
        """Reproduce MultiResolutionDetector's feature allocation across six levels."""
        levels = self._PYRAMID_LEVELS + self._UPSCALE_LEVELS + 1
        factor_points = self._SCALE_FACTOR**2
        weights = [
            factor_points ** (-(idx - self._UPSCALE_LEVELS))
            for idx in range(levels)
        ]
        weight_sum = sum(weights)
        base = [int(self.num_features * weight / weight_sum) for weight in weights]

        # One upscaled level receives base[0]. The original/downsample chain uses
        # the cumulative quotas from Kornia MultiResolutionDetector.detect().
        quotas = [base[0]]
        for idx in range(self._PYRAMID_LEVELS + 1):
            quotas.append(sum(base[: idx + 1 + self._UPSCALE_LEVELS]))
        return quotas

    def _debug_stats(
        self,
        *,
        counts: tuple[int, ...],
        pyramid_preselected_count: int,
        selected_values: torch.Tensor,
        first_rejected_value: torch.Tensor | None,
    ) -> SiftDebugStats:
        selected_count = int(selected_values.shape[0])
        if selected_count == 0:
            return SiftDebugStats(
                candidate_count=sum(counts),
                per_level_candidate_counts=counts,
                pyramid_preselected_count=pyramid_preselected_count,
                selected_count=0,
                strongest_abs_response=0.0,
                cutoff_abs_response=0.0,
                first_rejected_abs_response=None,
                boundary_gap=None,
                near_cutoff_count=0,
            )

        strongest = selected_values[0]
        cutoff = selected_values[-1]
        near_cutoff = (selected_values <= cutoff * 1.05).sum()
        torch.mps.synchronize()
        strongest_value = float(strongest.item())
        cutoff_value = float(cutoff.item())
        rejected_value = (
            float(first_rejected_value.item()) if first_rejected_value is not None else None
        )
        return SiftDebugStats(
            candidate_count=sum(counts),
            per_level_candidate_counts=counts,
            pyramid_preselected_count=pyramid_preselected_count,
            selected_count=selected_count,
            strongest_abs_response=strongest_value,
            cutoff_abs_response=cutoff_value,
            first_rejected_abs_response=rejected_value,
            boundary_gap=(
                cutoff_value - rejected_value if rejected_value is not None else None
            ),
            near_cutoff_count=int(near_cutoff.item()),
        )

    def detect(self, image: torch.Tensor) -> SiftFeatures:
        """Detect the strongest Kornia-style multi-resolution DoG keypoints."""
        if image.device.type != "mps":
            raise ValueError("image must already reside on the MPS device")
        if image.dtype != torch.float32:
            raise ValueError("image must have dtype torch.float32")
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("image must have shape [H, W, C] with at least 3 channels")

        height, width, channels = map(int, image.shape)
        if min(width, height) < 32:
            raise ValueError("image must be at least 32x32 pixels")

        image = image.contiguous()
        keepalive: list[torch.Tensor] = [image]
        gray = torch.empty(width * height, device="mps", dtype=torch.float32)
        self._dispatch(
            self._kernels.rgb_to_gray,
            (image, gray, self._i32(width * height), self._i32(channels)),
            threads=width * height,
            keepalive=keepalive,
        )

        quotas = self._kornia_quotas()
        level_count = len(quotas)
        level_counts = torch.zeros(level_count, device="mps", dtype=torch.int32)
        keepalive.append(level_counts)
        levels: list[_PyramidLevel] = []

        # Kornia config has one upscaled level at sqrt(2).
        up_width = int(width * self._SCALE_FACTOR)
        up_height = int(height * self._SCALE_FACTOR)
        up_image = self._resize(
            gray,
            width,
            height,
            up_width,
            up_height,
            keepalive,
        )
        levels.append(
            _PyramidLevel(
                scores=self._dog_nms(
                    up_image,
                    up_width,
                    up_height,
                    level_counts,
                    0,
                    keepalive,
                ),
                width=up_width,
                height=up_height,
                factor_x=width / up_width,
                factor_y=height / up_height,
                quota=quotas[0],
            )
        )

        # Original resolution plus four successive Kornia pyrdown(sqrt(2)) levels.
        current = gray
        current_width = width
        current_height = height
        for pyramid_index in range(self._PYRAMID_LEVELS + 1):
            if pyramid_index > 0:
                current, current_width, current_height = self._pyrdown(
                    current,
                    current_width,
                    current_height,
                    keepalive,
                )
            level_index = pyramid_index + 1
            levels.append(
                _PyramidLevel(
                    scores=self._dog_nms(
                        current,
                        current_width,
                        current_height,
                        level_counts,
                        level_index,
                        keepalive,
                    ),
                    width=current_width,
                    height=current_height,
                    factor_x=width / current_width,
                    factor_y=height / current_height,
                    quota=quotas[level_index],
                )
            )

        # One synchronization after all handwritten Metal image processing.
        torch.mps.synchronize()
        counts = tuple(int(value) for value in level_counts.cpu().tolist())

        level_points: list[torch.Tensor] = []
        level_responses: list[torch.Tensor] = []
        level_scales: list[torch.Tensor] = []
        for level, candidate_count in zip(levels, counts):
            keep_count = min(level.quota, candidate_count)
            if keep_count <= 0:
                continue
            ranked = torch.topk(level.scores, keep_count, largest=True, sorted=True)
            flat = ranked.indices
            x = (flat % level.width).to(torch.float32) * level.factor_x
            y = torch.div(flat, level.width, rounding_mode="floor").to(torch.float32) * level.factor_y
            level_points.append(torch.stack((x, y), dim=1))
            level_responses.append(ranked.values)
            scale = 0.5 * (level.factor_x + level.factor_y) * self._MR_SIZE
            level_scales.append(torch.full_like(ranked.values, scale))

        if not level_responses:
            empty_points = torch.empty((0, 2), device="mps", dtype=torch.float32)
            empty = torch.empty(0, device="mps", dtype=torch.float32)
            debug_stats = (
                self._debug_stats(
                    counts=counts,
                    pyramid_preselected_count=0,
                    selected_values=empty,
                    first_rejected_value=None,
                )
                if self.debug
                else None
            )
            return SiftFeatures(
                keypoints_xy=empty_points,
                responses=empty,
                scales=empty,
                orientations=empty,
                debug=debug_stats,
            )

        points = torch.cat(level_points, dim=0)
        responses = torch.cat(level_responses, dim=0)
        scales = torch.cat(level_scales, dim=0)
        pyramid_preselected_count = int(responses.shape[0])
        keep_count = min(self.num_features, pyramid_preselected_count)

        ranked_count = (
            keep_count + 1
            if self.debug and pyramid_preselected_count > keep_count
            else keep_count
        )
        ranked = torch.topk(responses, ranked_count, largest=True, sorted=True)
        indices = ranked.indices[:keep_count]
        selected_values = ranked.values[:keep_count]
        first_rejected = (
            ranked.values[keep_count]
            if self.debug and pyramid_preselected_count > keep_count
            else None
        )

        debug_stats = (
            self._debug_stats(
                counts=counts,
                pyramid_preselected_count=pyramid_preselected_count,
                selected_values=selected_values,
                first_rejected_value=first_rejected,
            )
            if self.debug
            else None
        )
        return SiftFeatures(
            keypoints_xy=points[indices],
            responses=responses[indices],
            scales=scales[indices],
            orientations=torch.zeros_like(selected_values),
            debug=debug_stats,
        )
