from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LidarScan:
    ranges: torch.Tensor
    points: torch.Tensor
    hit_mask: torch.Tensor
    gaussian_ids: torch.Tensor
    accumulated_alpha: torch.Tensor
    candidate_overflow_count: torch.Tensor | None = None
    bvh_stack_overflow_count: torch.Tensor | None = None

    def valid_points(self) -> torch.Tensor:
        return self.points[self.hit_mask]
