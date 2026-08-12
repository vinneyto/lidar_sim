import sys
from types import SimpleNamespace

import pytest
import torch

from gs_lidar import load_gaussian_ply, load_gaussian_scene


def test_load_gaussian_scene_uses_gsply_for_sog(monkeypatch, tmp_path):
    path = tmp_path / "scene.sog"
    path.write_bytes(b"compact scene")
    loaded = []
    model = SimpleNamespace(
        positions=[[1.0, 2.0, 3.0]],
        scales=[[0.1, 0.2, 0.3]],
        quaternions=[[2.0, 0.0, 0.0, 0.0]],
        opacity=[[0.75]],
        colors=[[0.2, 0.4, 0.6]],
    )
    monkeypatch.setitem(
        sys.modules,
        "gsply",
        SimpleNamespace(
            load=lambda source, *, device: loaded.append((source, device)) or model
        ),
    )

    scene = load_gaussian_scene(path)

    assert loaded == [(path, "cpu")]
    assert torch.equal(scene.means, torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.equal(scene.scales, torch.tensor([[0.1, 0.2, 0.3]]))
    assert torch.equal(scene.rotations, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.equal(scene.opacities, torch.tensor([0.75]))
    assert torch.allclose(scene.colors, torch.tensor([[0.2, 0.4, 0.6]]))


def test_load_gaussian_scene_accepts_mapping_and_checks_counts(monkeypatch):
    model = {
        "means": [[0.0, 0.0, 0.0]],
        "scales": [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        "rotations": [[1.0, 0.0, 0.0, 0.0]],
        "opacities": [1.0],
    }
    monkeypatch.setitem(
        sys.modules, "gsply", SimpleNamespace(load=lambda _, *, device: model)
    )

    with pytest.raises(ValueError, match="inconsistent Gaussian counts"):
        load_gaussian_scene("broken.splat")


def test_old_ply_loader_is_a_compatible_alias(monkeypatch):
    model = {
        "positions": [[0.0, 0.0, 0.0]],
        "scales": [[1.0, 1.0, 1.0]],
        "quaternions": [[1.0, 0.0, 0.0, 0.0]],
        "opacities": [0.5],
    }
    monkeypatch.setitem(
        sys.modules, "gsply", SimpleNamespace(load=lambda _, *, device: model)
    )

    assert torch.equal(load_gaussian_ply("legacy.ply").means, torch.zeros(1, 3))
