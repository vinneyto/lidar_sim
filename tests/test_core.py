import math
import pytest
import torch
from gs_lidar import *
from gs_lidar.gaussian_geometry import ray_gaussian_peaks


def cloud(x, opacity):
    n=len(x); return GaussianCloud(torch.tensor([[v,0.,0.] for v in x]),torch.ones(n,3),
        torch.tensor([[1.,0,0,0]]*n),torch.tensor(opacity))


def pose(): return LidarPose(torch.zeros(3),torch.tensor([1.,0,0,0]))


def config(threshold=.5):
    return LidarConfig(azimuth_min=0,azimuth_max=2*math.pi,elevation_min=0,elevation_max=0,
                       azimuth_samples=1,elevation_samples=1,near=.1,far=20,accumulated_alpha_threshold=threshold)


def test_coordinate_convention_and_seam():
    c=LidarConfig(azimuth_min=0,azimuth_max=2*math.pi,elevation_min=0,elevation_max=0,azimuth_samples=4,elevation_samples=1)
    _,d=generate_rays(pose(),c)
    torch.testing.assert_close(d,torch.tensor([[1.,0,0],[0,1.,0],[-1,0,0],[0,-1.,0]]),atol=1e-6,rtol=0)


def test_cumulative_opacity_and_storage_order():
    scene=cloud([4.90,5,5.08],[.2,.25,.3]); lo,hi=gaussian_aabbs(scene.means,scene.scales,scene.rotations)
    scan=CpuLidarTracer().trace(scene,pose(),config(),build_bvh(lo,hi,1))
    assert scan.hit_mask.item() and scan.gaussian_ids.item()==2
    torch.testing.assert_close(scan.accumulated_alpha,torch.tensor([[.58]]))
    perm=torch.tensor([2,0,1]); shuffled=GaussianCloud(scene.means[perm],scene.scales[perm],scene.rotations[perm],scene.opacities[perm])
    lo,hi=gaussian_aabbs(shuffled.means,shuffled.scales,shuffled.rotations)
    other=CpuLidarTracer().trace(shuffled,pose(),config(),build_bvh(lo,hi,1))
    torch.testing.assert_close(other.ranges,scan.ranges)


def test_transparent_foreground_does_not_stop_ray():
    scene=cloud([2,5],[.05,.9]); scan=CpuLidarTracer(False).trace(scene,pose(),config(),None)
    assert scan.gaussian_ids.item()==1 and scan.ranges.item()==pytest.approx(5)


def test_cutoff_is_not_a_surface_hit():
    scene=GaussianCloud(torch.tensor([[5.,3.,0.]]),torch.ones(1,3),torch.tensor([[1.,0,0,0]]),torch.tensor([1.]))
    scan=CpuLidarTracer(False).trace(scene,pose(),config(.5),None)
    assert not scan.hit_mask.item()  # q=9 participates, but alpha=exp(-4.5) is below threshold


def test_bvh_matches_bruteforce():
    torch.manual_seed(2); scene=cloud([2,3,4,6],[.1,.2,.3,.8]); lo,hi=gaussian_aabbs(scene.means,scene.scales,scene.rotations)
    a=CpuLidarTracer(False).trace(scene,pose(),config(),None); b=CpuLidarTracer().trace(scene,pose(),config(),build_bvh(lo,hi,2))
    assert torch.equal(a.hit_mask,b.hit_mask); assert torch.equal(a.gaussian_ids,b.gaussian_ids); torch.testing.assert_close(a.ranges,b.ranges)


def test_rotated_anisotropic_peak():
    q=torch.tensor([[math.sqrt(.5),0,0,math.sqrt(.5)]]); means=torch.tensor([[5.,1.,0.]])
    t,d=ray_gaussian_peaks(torch.zeros(1,3),torch.tensor([[1.,0,0]]),means,torch.tensor([[3.,.5,1.]]),q)
    assert t.item()==pytest.approx(5); assert d.item()==pytest.approx(4)


@pytest.mark.skipif(not torch.backends.mps.is_available() or not hasattr(torch.mps,"compile_shader"),reason="requires Apple MPS compile_shader")
def test_metal_smoke_and_cpu_parity():
    scene=cloud([2,3,4],[.2,.25,.3]); lo,hi=gaussian_aabbs(scene.means,scene.scales,scene.rotations); bvh=build_bvh(lo,hi,1)
    cpu=CpuLidarTracer().trace(scene,pose(),config(),bvh)
    gpu=MetalLidarTracer().trace(scene.to("mps"),pose().to("mps"),config(),bvh.to("mps"))
    torch.testing.assert_close(gpu.ranges.cpu(),cpu.ranges); assert torch.equal(gpu.hit_mask.cpu(),cpu.hit_mask)
