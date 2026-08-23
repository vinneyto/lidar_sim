from dataclasses import dataclass

import kornia as K
import torch


@dataclass(frozen=True)
class SiftFeatures:
    """SIFT detector keypoints extracted from one rendered RGB image."""

    keypoints_xy: torch.Tensor
    responses: torch.Tensor
    scales: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.keypoints_xy.shape[0])


class SiftFeatureDetector:
    """Run only the DoG detection stage used by Kornia SIFTFeature."""

    def __init__(self, num_features: int = 512, *, compile_detector: bool = False) -> None:
        if num_features < 1:
            raise ValueError("num_features must be positive")

        self.num_features = num_features
        self.compile_detector = compile_detector
        self._detectors: dict[torch.device, torch.nn.Module] = {}

    def _detector(self, device: torch.device) -> torch.nn.Module:
        detector = self._detectors.get(device)
        if detector is None:
            detector = K.feature.MultiResolutionDetector(
                K.feature.BlobDoGSingle(1.0, 1.6),
                num_features=self.num_features,
                ori_module=K.feature.PassLAF(),
                aff_module=K.feature.PassLAF(),
            ).to(device)
            detector.eval()

            if self.compile_detector:
                # Let torch.compile choose when to introduce dynamic shapes.
                # The detector reuses gaussian_blur2d across multiple pyramid
                # resolutions, so forcing dynamic=False causes repeated
                # shape-specialized recompilations and can hit the recompile limit.
                detector = torch.compile(
                    detector,
                    backend="inductor",
                    fullgraph=False,
                )

            self._detectors[device] = detector
        return detector

    @torch.inference_mode()
    def detect(self, image: torch.Tensor) -> SiftFeatures:
        """Detect DoG keypoints in an ``H x W x 3`` float RGB image."""
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("image must have shape [H, W, 3]")

        rgb = image.clamp(0, 1).permute(2, 0, 1).unsqueeze(0)
        gray = K.color.rgb_to_grayscale(rgb)
        lafs, responses = self._detector(image.device)(gray)

        keypoints_xy = K.feature.get_laf_center(lafs)[0]
        scales = K.feature.get_laf_scale(lafs)[0, :, 0, 0]
        responses = responses[0].reshape(-1)

        feature_count = keypoints_xy.shape[0]
        if not (responses.shape[0] == scales.shape[0] == feature_count):
            raise RuntimeError(
                "Kornia detector outputs have inconsistent feature counts: "
                f"lafs={feature_count}, responses={responses.shape[0]}, "
                f"scales={scales.shape[0]}"
            )

        order = torch.argsort(responses, descending=True)
        return SiftFeatures(
            keypoints_xy=keypoints_xy[order],
            responses=responses[order],
            scales=scales[order],
        )
