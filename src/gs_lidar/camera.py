from dataclasses import dataclass


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera with +X right, +Y up and +Z forward."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    near: float = 0.1
    far: float = 10.0

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("camera width and height must be positive")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera focal lengths must be positive")
        if not 0 < self.near < self.far:
            raise ValueError("camera planes must satisfy 0 < near < far")
