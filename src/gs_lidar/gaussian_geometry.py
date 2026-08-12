import torch


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    ), -1).reshape(q.shape[:-1] + (3, 3))


def gaussian_aabbs(means: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor,
                   sigma_cutoff: float = 3.0) -> tuple[torch.Tensor, torch.Tensor]:
    rotation = quaternion_to_matrix(rotations)
    extent = sigma_cutoff * torch.sqrt(torch.sum(rotation.square() * scales[:, None, :].square(), dim=-1))
    return means - extent, means + extent


def ray_gaussian_peaks(origins: torch.Tensor, directions: torch.Tensor, means: torch.Tensor,
                       scales: torch.Tensor, rotations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return t_peak and minimum squared Mahalanobis distance for all ray/Gaussian pairs."""
    rotation = quaternion_to_matrix(rotations)
    delta = origins[:, None, :] - means[None, :, :]
    local_o = torch.einsum("njg,rnj->rng", rotation, delta) / scales[None]
    local_d = torch.einsum("njg,rj->rng", rotation, directions) / scales[None]
    denominator = local_d.square().sum(-1).clamp_min(torch.finfo(origins.dtype).tiny)
    t = -(local_o * local_d).sum(-1) / denominator
    q = (local_o + t[..., None] * local_d).square().sum(-1)
    return t, q
