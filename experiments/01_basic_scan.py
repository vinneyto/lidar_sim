#!/usr/bin/env python3
import argparse
import math
import time
import torch
from gs_lidar import (LidarConfig, LidarPose, LidarSimulator, build_bvh, gaussian_aabbs, load_gaussian_ply)
from gs_lidar.rerun_viewer import visualize


def orientation(yaw: float, pitch: float, roll: float) -> torch.Tensor:
    """Intrinsic Z(up)-Y(left)-X(forward) rotations, supplied in degrees."""
    y, p, r = (math.radians(v)/2 for v in (yaw, pitch, roll))
    qz=torch.tensor([math.cos(y),0,0,math.sin(y)]); qy=torch.tensor([math.cos(p),0,math.sin(p),0]); qx=torch.tensor([math.cos(r),math.sin(r),0,0])
    def mul(a,b):
        aw,ax,ay,az=a; bw,bx,by,bz=b
        return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                            aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw))
    return mul(mul(qz,qy),qx).float()


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("scene"); p.add_argument("--backend",choices=("cpu","metal"),default="metal")
    p.add_argument("--azimuth-samples",type=int,default=360); p.add_argument("--elevation-samples",type=int,default=64)
    p.add_argument("--position",type=float,nargs=3,default=(0,0,0)); p.add_argument("--yaw",type=float,default=0)
    p.add_argument("--pitch",type=float,default=0); p.add_argument("--roll",type=float,default=0)
    p.add_argument("--near",type=float,default=.1); p.add_argument("--far",type=float,default=100)
    p.add_argument("--sigma-cutoff",type=float,default=3); p.add_argument("--alpha-threshold",type=float,default=.5)
    args=p.parse_args(); start=time.perf_counter(); scene=load_gaussian_ply(args.scene); loaded=time.perf_counter()
    lo,hi=gaussian_aabbs(scene.means,scene.scales,scene.rotations,args.sigma_cutoff); bvh=build_bvh(lo,hi); built=time.perf_counter()
    device="mps" if args.backend=="metal" else "cpu"; scene,bvh=scene.to(device),bvh.to(device)
    pose=LidarPose(torch.tensor(args.position,dtype=torch.float32,device=device),orientation(args.yaw,args.pitch,args.roll).to(device))
    config=LidarConfig(azimuth_samples=args.azimuth_samples,elevation_samples=args.elevation_samples,near=args.near,far=args.far,
                       gaussian_sigma_cutoff=args.sigma_cutoff,accumulated_alpha_threshold=args.alpha_threshold)
    uploaded=time.perf_counter(); scan=LidarSimulator(scene,bvh,args.backend).scan(pose,config)
    if device=="mps": torch.mps.synchronize()
    traced=time.perf_counter(); hits=int(scan.hit_mask.sum().cpu()); rays=scan.hit_mask.numel()
    print(f"Gaussians: {scene.means.shape[0]} | BVH nodes: {bvh.bbox_min.shape[0]} | rays: {rays}")
    print(f"hits: {hits} ({100*hits/rays:.1f}%) | load: {loaded-start:.3f}s | BVH: {built-loaded:.3f}s | upload: {uploaded-built:.3f}s")
    print(f"trace: {traced-uploaded:.3f}s | {rays/max(traced-uploaded,1e-9)/1e6:.3f} MRays/s")
    if scan.candidate_overflow_count is not None: print("candidate/stack overflow:",int(scan.candidate_overflow_count.cpu()),int(scan.bvh_stack_overflow_count.cpu()))
    visualize(scene,pose,scan)


if __name__ == "__main__": main()
