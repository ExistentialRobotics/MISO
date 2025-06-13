from os.path import join
import numpy as np
from grid_opt.configs import *
from grid_opt.utils.utils_sdf import *
import grid_opt.utils.utils_eval as utils_eval
from evo.core import metrics as evo_metrics
import open3d as o3d
import logging
logger = logging.getLogger(__name__)


def create_ncd_dataset(
        cfg, 
        voxel_size=0.03, 
        near_surf_std=0.1, 
        n_near=4, 
        n_free=2, 
        n_behind=1,
        frame_samples=2**12,
        frame_batchsize=2**10, 
        num_frames=None
    ) -> PosedSdf3DLidar:
    dataset = PosedSdf3DLidar(
        lidar_folder=cfg['dataset']['path'],
        pose_file_gt=cfg['dataset']['pose_gt'],
        pose_file_init=cfg['dataset']['pose_init'],
        trunc_dist=cfg['dataset']['trunc_dist'],
        num_frames=num_frames,
        frame_samples=frame_samples,
        frame_batchsize=frame_batchsize,
        voxel_size=voxel_size,
        near_surface_std=near_surf_std,
        near_surface_n=n_near,
        free_space_n=n_free,
        behind_surface_n=n_behind,
        min_dist_ratio=0.50,
        min_z=-10.0,
        max_z=60.0,
        min_range=1.5, 
        max_range=60.0,
        adaptive_range=False
    )
    return dataset


def align_mesh_to_ref(est_mesh, ref_mesh, 
                      constraint_type='point_to_plane', 
                      voxel_size=0.02, 
                      threshold_factor_coarse=15, 
                      threshold_factor_fine=1.5, 
                      num_iters=100, 
                      ref_format='mesh'):
    num_points = 1000000
    pcd_src = est_mesh.sample_points_uniformly(number_of_points=num_points)
    if ref_format == 'mesh':
        pcd_dst = ref_mesh.sample_points_uniformly(number_of_points=num_points)
    elif ref_format == 'pointcloud':
        pcd_dst = ref_mesh.voxel_down_sample(0.01)  # 1cm voxel size
    else:
        raise ValueError(f"Unknown reference format {ref_format}!")
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
    return est_mesh


def evaluate_mapping(file_pred, file_trgt, file_pred_aligned, R_align=None, t_align=None, ref_format='mesh'):
    """Evaluate mesh reconstruction for NCD dataset.

    Args:
        file_pred (_type_): _description_
        file_trgt (_type_): _description_
        R_align (_type_, optional): Optional rotation to align to ref. Defaults to None.
        t_align (_type_, optional): Optional translation to align to ref. Defaults to None.
        ref_format (str, optional): Format of the reference map file. Defaults to 'mesh'. Other option is 'pointcloud'.

    Returns:
        Dictionary of map metrics.
    """
    mesh_est = o3d.io.read_triangle_mesh(file_pred)
    if R_align is not None:
        T_align = np.eye(4)
        T_align[:3, :3] = R_align
        T_align[:3, 3] = t_align.flatten()  
        mesh_est.transform(T_align)
    if ref_format == 'mesh':
        mesh_ref = o3d.io.read_triangle_mesh(file_trgt)
    elif ref_format == 'pointcloud':
        mesh_ref = o3d.io.read_point_cloud(file_trgt)
    else:
        raise ValueError(f"Unknown reference format {ref_format}!")
    mesh_est = align_mesh_to_ref(mesh_est, mesh_ref, ref_format=ref_format)
    o3d.io.write_triangle_mesh(file_pred_aligned, mesh_est)
    mesh_sample_point = 1000000
    voxel_down_sample_res = 0.02
    verts_pred = utils_eval.sample_points_from_mesh(file_pred_aligned, mesh_sample_point=mesh_sample_point, voxel_down_sample_res=voxel_down_sample_res)
    verts_trgt = utils_eval.sample_points_from_mesh(file_trgt, mesh_sample_point=mesh_sample_point, voxel_down_sample_res=voxel_down_sample_res, input_format=ref_format)
    # Filter points using GT oriented bounding box
    obb = mesh_ref.get_minimal_oriented_bounding_box()
    verts_pred = utils_eval.filter_points_by_oriented_bound(verts_pred, obb)
    verts_trgt = utils_eval.filter_points_by_oriented_bound(verts_trgt, obb)
    # Compute and return metric
    metrics_map = utils_eval.compute_chamfer_metrics(
        verts_pred, verts_trgt, threshold=0.20, truncation_acc=0.50, truncation_com=None
    )
    return metrics_map