import pytest
import torch

from gs_lidar import SiftFeatureDetector


class _FakeDetector:
    def __init__(self, responses: torch.Tensor) -> None:
        self.responses = responses

    def __call__(self, image: torch.Tensor):
        device = image.device
        dtype = image.dtype
        lafs = torch.tensor(
            [
                [
                    [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]],
                    [[2.0, 0.0, 30.0], [0.0, 2.0, 40.0]],
                    [[3.0, 0.0, 50.0], [0.0, 3.0, 60.0]],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        return lafs, self.responses.to(device=device, dtype=dtype)


class _TestDetector(SiftFeatureDetector):
    def __init__(self, responses: torch.Tensor) -> None:
        super().__init__(num_features=3)
        self.fake_detector = _FakeDetector(responses)

    def _detector(self, device: torch.device):
        return self.fake_detector


@pytest.mark.parametrize(
    "responses",
    [
        torch.tensor([[0.2, 0.9, 0.5]]),
        torch.tensor([[[0.2], [0.9], [0.5]]]),
    ],
)
def test_sift_detector_accepts_response_tensor_variants(responses: torch.Tensor) -> None:
    detector = _TestDetector(responses)
    image = torch.zeros(80, 100, 3)

    features = detector.detect(image)

    assert features.count == 3
    assert features.responses.shape == (3,)
    assert features.scales.shape == (3,)
    assert torch.allclose(features.responses, torch.tensor([0.9, 0.5, 0.2]))
    assert torch.allclose(
        features.keypoints_xy,
        torch.tensor([[30.0, 40.0], [50.0, 60.0], [10.0, 20.0]]),
    )
