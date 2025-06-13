from dataclasses import dataclass
from os.path import join
import numpy as np
import open3d as o3d
from grid_opt.datasets.sdf_rgbd import PosedSdfRgbd
from grid_opt.utils.utils_data import CameraParameters
import logging
logger = logging.getLogger(__name__)

@dataclass
class SceneMetadata:
    bound: list
    name: str
    path: str
    intrinsics_file: str
    gt_mesh: str
    num_kfs: int
    anchor_kfs: list

def scannet_scenes():
    # Define meta information for all ScanNet scenes
    scene0000_00 = SceneMetadata(
        name='0000_00',
        path="./data/ScanNet/scene0000_00_mipsfusion",
        intrinsics_file='./data/ScanNet/scene0000_00_mipsfusion/scene0000_00.txt', 
        gt_mesh="/home/hanwen/data/ScanNet/scans/scene0000_00/scene0000_00_vh_clean.ply",
        bound=[[-0.02,  10.38], [-0.01, 8.74], [-0.01,  3.03]],
        num_kfs=372,
        anchor_kfs=[0, 124, 255],
    )
    scene0011_00 = SceneMetadata(
        name='0011_00',
        path="./data/ScanNet/scene0011_00_mipsfusion",
        intrinsics_file='./data/ScanNet/scene0011_00_mipsfusion/scene0011_00.txt', 
        gt_mesh="/home/hanwen/data/ScanNet/scans/scene0011_00/scene0011_00_vh_clean.ply",
        bound=[[1.50, 7.50], [-0.05, 8.25], [-0.05, 2.70]],
        num_kfs=159,
        anchor_kfs=[0, 73, 86, 121]
    )
    scene0024_00 = SceneMetadata(
        name='0024_00',
        path="./data/ScanNet/scene0024_00_mipsfusion",
        intrinsics_file='./data/ScanNet/scene0024_00_mipsfusion/scene0024_00.txt', 
        gt_mesh="/home/hanwen/data/ScanNet/scans/scene0024_00/scene0024_00_vh_clean.ply",
        bound=[[0.00, 7.20], [-0.05, 8.05], [-0.05, 2.50]],
        num_kfs=227,
        anchor_kfs=[0, 30, 84, 101, 131]
    )
    scene0207_00 = SceneMetadata(
        name='0207_00',
        path="./data/ScanNet/scene0207_00_mipsfusion",
        intrinsics_file='./data/ScanNet/scene0207_00_mipsfusion/scene0207_00.txt', 
        gt_mesh="/home/hanwen/data/ScanNet/scans/scene0207_00/scene0207_00_vh_clean.ply",
        bound=[[1.00, 9.00], [0.00, 7.10], [-0.10, 2.90]],
        num_kfs=133,
        anchor_kfs=[0, 35]
    )
    return {
        '0000_00': scene0000_00,
        '0011_00': scene0011_00,
        '0024_00': scene0024_00,
        '0207_00': scene0207_00
    }

def get_scannet_metadata(file):
    info = {}
    with open(file, 'r') as f:
        for line in f.read().splitlines():
            split = line.split(' = ')
            info[split[0]] = split[1]
    return info

def get_scannet_cam_intrinsics(file) -> CameraParameters:
    info = get_scannet_metadata(file)
    return CameraParameters(
        depth_scale=1000.0, 
        fx=float(info['fx_depth']),
        fy=float(info['fy_depth']),
        cx=float(info['mx_depth']),
        cy=float(info['my_depth']),
        H=int(info['depthHeight']),
        W=int(info['depthWidth']),
    )

def create_scannet_dataset(
        scannet_root: str,
        scene_id: str,
        trunc_dist: float = 0.15,
        frame_downsample: int = 15,
        n_rays: int = 200,
        n_surf_samples: int = 8,
        n_strat_samples: int = 19,
        voxel_size: float = None
    ):
    scene_name = f"scene{scene_id}"
    scene_file = join(scannet_root, scene_name, f"{scene_name}.txt")
    info = get_scannet_metadata(scene_file)
    cam_params = get_scannet_cam_intrinsics(scene_file)
    num_input_frames = int(info['numColorFrames'])
    dataset = PosedSdfRgbd(
        dataset_root=join(scannet_root, scene_name),
        num_input_frames=num_input_frames,
        cam_params=cam_params,
        frame_downsample=frame_downsample,
        n_rays=n_rays,
        min_depth=0.07,
        max_depth=12.0,
        n_surf_samples=n_surf_samples,
        n_strat_samples=n_strat_samples,
        trunc_dist=trunc_dist,
        voxel_size=voxel_size, 
    )
    return dataset

def align_mesh_to_ref(est_mesh, ref_mesh, 
                      constraint_type='point_to_plane', 
                      voxel_size=0.02, 
                      threshold_factor_coarse=15, 
                      threshold_factor_fine=1.5, 
                      num_iters=100):
    num_points = 1000000
    pcd_src = est_mesh.sample_points_uniformly(number_of_points=num_points)
    pcd_dst = ref_mesh.sample_points_uniformly(number_of_points=num_points)
    if constraint_type == 'point_to_plane':
        logger.debug("Apply point-to-plane ICP")
        pcd_src.estimate_normals()
        pcd_dst.estimate_normals()
        TransformationEstimation = o3d.pipelines.registration.TransformationEstimationPointToPlane
    elif constraint_type == 'point_to_point':
        logger.debug("Apply point-to-point ICP")
        TransformationEstimation = o3d.pipelines.registration.TransformationEstimationPointToPoint
    else:
        raise ValueError(f"Unknown constraint type {constraint_type}")
    
    loss = o3d.pipelines.registration.TukeyLoss(k=1e-2)
    T_dst_src_init = np.eye(4)
    icp_coarse = o3d.pipelines.registration.registration_icp(
        pcd_src, pcd_dst, voxel_size * threshold_factor_coarse, T_dst_src_init,
        TransformationEstimation(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=num_iters)
    )
    icp_fine = o3d.pipelines.registration.registration_icp(
        pcd_src, pcd_dst, voxel_size * threshold_factor_fine, icp_coarse.transformation,
        TransformationEstimation(loss),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=num_iters)
    )
    print(f"Finetuned ICP result:")
    print(icp_fine)
    T_dst_src = icp_fine.transformation.copy()
    print(f"Finetuned ICP alignment is:\n {T_dst_src}")    
    est_mesh.transform(T_dst_src)
    # Visualize
    est_pcd = est_mesh.sample_points_uniformly(number_of_points=100000)
    est_pcd.estimate_normals()
    o3d.visualization.draw_geometries([ref_mesh, est_pcd])
    return est_mesh