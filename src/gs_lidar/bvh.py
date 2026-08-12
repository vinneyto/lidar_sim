from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class FlatBVH:
    bbox_min: torch.Tensor
    bbox_max: torch.Tensor
    left_child: torch.Tensor
    right_child: torch.Tensor
    first_primitive: torch.Tensor
    primitive_count: torch.Tensor
    primitive_indices: torch.Tensor

    def to(self, device: torch.device | str) -> FlatBVH:
        return replace(self, **{k: v.to(device) for k, v in self.__dict__.items()})


def build_bvh(
    box_min: torch.Tensor, box_max: torch.Tensor, leaf_size: int = 8
) -> FlatBVH:
    """Median-split CPU builder producing GPU-friendly flat tensors."""
    if box_min.device.type != "cpu" or leaf_size < 1:
        raise ValueError("BVH construction requires CPU tensors and positive leaf_size")
    nodes: list[dict] = []
    ordered: list[int] = []
    centroids = (box_min + box_max) * 0.5

    def create(ids: torch.Tensor) -> int:
        index = len(nodes)
        node = {
            "min": box_min[ids].amin(0),
            "max": box_max[ids].amax(0),
            "left": -1,
            "right": -1,
            "first": -1,
            "count": 0,
        }
        nodes.append(node)
        if ids.numel() <= leaf_size:
            node["first"], node["count"] = len(ordered), ids.numel()
            ordered.extend(ids.tolist())
        else:
            spread = centroids[ids].amax(dim=0) - centroids[ids].amin(dim=0)
            axis = int(torch.argmax(spread))
            ids = ids[torch.argsort(centroids[ids, axis])]
            middle = ids.numel() // 2
            node["left"], node["right"] = create(ids[:middle]), create(ids[middle:])
        return index

    if box_min.shape[0] == 0:
        raise ValueError("cannot build an empty BVH")
    create(torch.arange(box_min.shape[0]))

    def node_tensor(key: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([node[key] for node in nodes], dtype=dtype)

    return FlatBVH(
        torch.stack([n["min"] for n in nodes]).float(),
        torch.stack([n["max"] for n in nodes]).float(),
        node_tensor("left", torch.int32),
        node_tensor("right", torch.int32),
        node_tensor("first", torch.int32),
        node_tensor("count", torch.int32),
        torch.tensor(ordered, dtype=torch.int32),
    )


def ray_aabb(
    origin: torch.Tensor,
    direction: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    near: float,
    far: float,
) -> bool:
    inv = torch.where(
        direction.abs() > 1e-12,
        direction.reciprocal(),
        torch.full_like(direction, float("inf")),
    )
    t0, t1 = (lo - origin) * inv, (hi - origin) * inv
    return bool(
        torch.maximum(t0.minimum(t1).amax(), torch.tensor(near))
        <= torch.minimum(t0.maximum(t1).amin(), torch.tensor(far))
    )


def bvh_candidates(
    bvh: FlatBVH, origin: torch.Tensor, direction: torch.Tensor, near: float, far: float
) -> torch.Tensor:
    stack, result = [0], []
    while stack:
        node = stack.pop()
        if not ray_aabb(
            origin, direction, bvh.bbox_min[node], bvh.bbox_max[node], near, far
        ):
            continue
        count = int(bvh.primitive_count[node])
        if count:
            first = int(bvh.first_primitive[node])
            result.extend(bvh.primitive_indices[first : first + count].tolist())
        else:
            stack.extend((int(bvh.left_child[node]), int(bvh.right_child[node])))
    return torch.tensor(result, dtype=torch.long)
