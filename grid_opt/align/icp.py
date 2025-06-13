import copy
import open3d as o3d
import torch
from torch.utils.data import Dataset
from grid_opt.configs import *
import grid_opt.utils.utils as utils
import grid_opt.utils.utils_geometry as utils_geometry
from grid_opt.models.grid_atlas import GridAtlas
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_points_for_submap(
    grid_atlas: GridAtlas,
    dataset: Dataset,
    submap_id: int,
    num_batches=1,
    trunc_dist=1e-3,
):
    coords_list = []
    with torch.no_grad():
        for batch_id in range(num_batches):
            model_input, gt = dataset[0]
            submap_idxs = model_input['submap_idxs'][:,0]
            sample_indices = torch.nonzero(submap_idxs == submap_id, as_tuple=False).squeeze(1)
            if sample_indices.numel() == 0:
                logger.warning(f"Submap {submap_id} is not in batch {batch_id}.")
                continue
            gt_sdf = gt['sdf'][sample_indices, :]                          # (num_samples, 1)
            coords_kf = model_input['coords_kf'][sample_indices, :]        # (num_samples, 3)
            kf_idxs = model_input['keyframe_idxs'][sample_indices, 0]      # (num_samples, )
            coords_submap = coords_kf.clone()  
            for kf_id in range(grid_atlas.num_keyframes):
                idxs_select = torch.nonzero(kf_idxs == kf_id, as_tuple=False).squeeze(1)
                if idxs_select.numel() == 0:
                    continue
                R_submap_kf, t_submap_kf = grid_atlas.updated_kf_pose_in_submap(kf_id, submap_id)
                coords_submap[idxs_select, :] = utils_geometry.transform_points_to(
                    coords_kf[idxs_select, :],
                    R_submap_kf,
                    t_submap_kf
                )
            mask_valid = torch.abs(gt_sdf) < trunc_dist
            valid_indices = torch.nonzero(mask_valid, as_tuple=False)[:, 0]
            coords_list.append(coords_submap[valid_indices, :])
        coords = torch.cat(coords_list, dim=0)
    return coords.detach().cpu().numpy()


def align_submap_pair(
    grid_atlas: GridAtlas,
    dataset: Dataset,
    src_id: int,
    dst_id: int,
    constraint_type='point_to_plane',
    voxel_size=0.02,
    threshold_factor_coarse=15,
    threshold_factor_fine=1.5,
    num_iters=30,
    num_batches=10,
    update_grid_atlas=True, 
):
    timer = utils.PerfTimer(activate=True)
    timer.reset()
    # points_src = dataset.get_points_for_submap(src_id, num_batches=num_batches)
    # points_dst = dataset.get_points_for_submap(dst_id, num_batches=num_batches)
    points_src = get_points_for_submap(grid_atlas, dataset, src_id, num_batches=num_batches)
    points_dst = get_points_for_submap(grid_atlas, dataset, dst_id, num_batches=num_batches)
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(points_src)
    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(points_dst)
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
    pcd_src = pcd_src.voxel_down_sample(voxel_size=voxel_size)
    pcd_dst = pcd_dst.voxel_down_sample(voxel_size=voxel_size)
    R_world_src, t_world_src = grid_atlas.initial_submap_pose(src_id)
    R_world_dst, t_world_dst = grid_atlas.initial_submap_pose(dst_id)
    T_world_src = utils_geometry.pose_matrix(R_world_src, t_world_src)
    T_world_dst = utils_geometry.pose_matrix(R_world_dst, t_world_dst)
    T_dst_src = torch.linalg.solve(T_world_dst, T_world_src)
    T_dst_src = T_dst_src.detach().cpu().numpy()
    # evaluation = o3d.pipelines.registration.evaluate_registration(pcd_src, pcd_dst, threshold, T_dst_src)
    # print(evaluation)
    logger.debug(f"Alignment starts. Rotation correction:\n {grid_atlas.rotation_corrections[dst_id].flatten()} \n Translation correction:\n {grid_atlas.translation_corrections[dst_id].flatten()}.")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        pcd_src, pcd_dst, voxel_size * threshold_factor_coarse, T_dst_src,
        TransformationEstimation(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=num_iters)
    )
    icp_fine = o3d.pipelines.registration.registration_icp(
        pcd_src, pcd_dst, voxel_size * threshold_factor_fine, icp_coarse.transformation,
        TransformationEstimation(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=num_iters)
    )
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        pcd_src, pcd_dst, voxel_size * threshold_factor_fine,
        icp_fine.transformation)
    if update_grid_atlas:
        T_dst_src_opt = torch.from_numpy(icp_fine.transformation.copy()).float().to(T_world_src)
        T_world_dst_opt = T_world_src @ torch.linalg.inv(T_dst_src_opt)
        R_world_dst_opt = T_world_dst_opt[:3, :3]
        t_world_dst_opt = T_world_dst_opt[:3, [3]]
        R_delta, t_delta = utils_geometry.get_pose_correction(R_world_dst, t_world_dst, R_world_dst_opt, t_world_dst_opt)
        logger.debug(f"Alignment ends. Rotation correction:\n {R_delta.flatten()} \n Translation correction:\n {t_delta.flatten()}.")
        grid_atlas.set_submap_pose_correction(dst_id, R_delta, t_delta)
    cpu_time, gpu_time = timer.check()
    logger.info(f"ICP {src_id}-{dst_id} ends. cpu_time={cpu_time:.2f} sec, gpu_time={gpu_time:.2f} sec.")
    return icp_fine, information_icp, {'cpu_time_sec': cpu_time, 'gpu_time_sec': gpu_time}


def align_multiple_submaps(
    grid_atlas: GridAtlas,
    dataset: Dataset,
    submap_pairs=None,
    check_intersection=True,
    constraint_type='point_to_plane',
    voxel_size=0.02,
    threshold_factor_coarse=15,
    threshold_factor_fine=1.5,
    num_icp_iters=30,
    num_batches=1,
    set_odometry_certain=False,
):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    timer = utils.PerfTimer(activate=True)
    timer.reset()
    # Create pose graph nodes
    for submap_id in range(grid_atlas.num_submaps):
        R_world_submap, t_world_submap = grid_atlas.initial_submap_pose(submap_id)
        T_world_submap = utils_geometry.pose_matrix(R_world_submap, t_world_submap).detach().cpu().numpy()
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(T_world_submap))
    # Create pose graph edges
    if submap_pairs is None:
        submap_pairs = []
        for src_id in range(dataset.num_submaps):
            for dst_id in range(src_id+1, dataset.num_submaps):
                submap_pairs.append((src_id, dst_id))
    for src_id, dst_id in submap_pairs:
        # Check submap intersection
        if check_intersection and grid_atlas.check_submap_intersection(src_id, dst_id) == False:
            logger.debug(f"Skip pair {src_id}, {dst_id}: no intersection.")
            continue
        icp_result, information_icp, icp_info = align_submap_pair(
            grid_atlas,
            dataset,
            src_id,
            dst_id,
            constraint_type=constraint_type,
            voxel_size=voxel_size,
            threshold_factor_coarse=threshold_factor_coarse,
            threshold_factor_fine=threshold_factor_fine,
            num_iters=num_icp_iters,
            num_batches=num_batches,
            update_grid_atlas=False,
        )
        uncertain = True
        if set_odometry_certain and dst_id == src_id + 1:
            uncertain = False
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                src_id,
                dst_id,
                icp_result.transformation,
                information_icp,
                uncertain=uncertain,
            )
        )
    # Optimize pose graph
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=voxel_size * threshold_factor_fine,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )
    # Update grid atlas
    for submap_id in range(grid_atlas.num_submaps):
        R_world_submap, t_world_submap = grid_atlas.initial_submap_pose(submap_id)
        T_world_submap_opt = torch.from_numpy(pose_graph.nodes[submap_id].pose.copy()).float().to(R_world_submap)
        R_world_submap_opt = T_world_submap_opt[:3, :3]
        t_world_submap_opt = T_world_submap_opt[:3, [3]]
        R_delta, t_delta = utils_geometry.get_pose_correction(R_world_submap, t_world_submap, R_world_submap_opt, t_world_submap_opt)
        grid_atlas.set_submap_pose_correction(submap_id, R_delta, t_delta)
    cpu_time, gpu_time = timer.check()
    info = {'cpu_time_sec': cpu_time, 'gpu_time_sec': gpu_time}
    return info
