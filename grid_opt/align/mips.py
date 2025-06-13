import numpy as np
import torch
from grid_opt.align.base import *
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

##########################
#      MIPS-Fusion       #
##########################

def pairwise_loss_mips_scannet(
    grid_atlas: GridAtlas,
    data_loader: DataLoader,
    src_id: int,
    dst_id: int,
    residual_weight=3000,
    use_bound=True,
    constraint_type='point_to_plane',
    device="cuda:0"
):
    """Implement pairwise submap alignment using the method from MIPS-Fusion, eq. 19-22. 

    Args:
        grid_atlas (GridAtlas): GridAtlas containing the pair of submaps to be aligned.
        dataset (SubmapSdf3D): Dataset object containing observations from all submaps.
        src_id (int): ID of source submap. During pairwise alignment, this submap pose will be held fixed.
        dst_id (int): ID of dest submap. During pairwise alignment, this submap pose will be optimized.
    """
    assert src_id < grid_atlas.num_submaps
    assert dst_id < grid_atlas.num_submaps
    grid_src = grid_atlas.get_submap(src_id)
    grid_dst = grid_atlas.get_submap(dst_id) 
    loss_dict = {}
    model_input, gt = utils.get_batch(data_loader, device)
    # Obtain corrected submap poses
    R_world_src, t_world_src = grid_atlas.updated_submap_pose(src_id, device)
    R_world_dst, t_world_dst = grid_atlas.updated_submap_pose(dst_id, device)
    # Select coords that are on surface and observed in both src and dst submaps
    with torch.no_grad():
        # Get coordinates that belong to src submap
        submap_idxs = model_input['submap_idxs'][0,:,0]
        sample_indices = torch.nonzero(submap_idxs == src_id, as_tuple=False).squeeze(1)
        if sample_indices.numel() == 0:
            raise ValueError(f"No samples found for submap {src_id}!")
        coords_kf = model_input['coords_kf'][0, sample_indices, :]        # (num_samples, 3)
        kf_idxs = model_input['keyframe_idxs'][0, sample_indices, 0]      # (num_samples, )
        # Can we get rid of this for loop?
        coords_submap = coords_kf.clone()                                 # (num_samples, 3)
        for kf_id in range(grid_atlas.num_keyframes):
            idxs_select = torch.nonzero(kf_idxs == kf_id, as_tuple=False).squeeze(1)
            if idxs_select.numel() == 0:
                continue
            R_submap_kf, t_submap_kf = grid_atlas.updated_kf_pose_in_submap(kf_id, src_id)
            coords_submap[idxs_select, :] = utils_geometry.transform_points_to(
                coords_kf[idxs_select, :],
                R_submap_kf,
                t_submap_kf
            )
        coords_src = coords_submap
        coords_world = utils_geometry.transform_points_to(coords_src, R_world_src, t_world_src)
        coords_dst = utils_geometry.transfrom_points_from(coords_world, R_world_dst, t_world_dst)
        gt_sdf = gt['sdf'][0][sample_indices, :]
        mask_valid = torch.abs(gt_sdf) < 1e-3
        logger.debug(f"Valid measurements after src surface pruning: {torch.count_nonzero(mask_valid)}.")
        if use_bound:
            mask_bnd = utils_geometry.coords_in_bound(coords_dst, grid_dst.bound)
            assert mask_bnd.shape == mask_valid.shape
            mask_valid = torch.logical_and(mask_bnd, mask_valid)
            logger.debug(f"Valid measurements after dst bound pruning: {torch.count_nonzero(mask_valid)}.")
        valid_indices = torch.nonzero(mask_valid, as_tuple=False)[:, 0]
    points_src = coords_src[valid_indices, :].requires_grad_(True)
    n_surf = points_src.shape[0]
    sdfs_src = grid_src(points_src)
    with torch.no_grad():
        grad_src = torch.autograd.grad(sdfs_src, points_src, grad_outputs=torch.ones_like(sdfs_src), create_graph=True)[0]
    # Find correspondence in dst submap
    points_world = utils_geometry.transform_points_to(points_src, R_world_src, t_world_src)
    points_dst = utils_geometry.transfrom_points_from(points_world, R_world_dst, t_world_dst)
    sdfs_dst = grid_dst(points_dst)
    with torch.no_grad():
        grad_dst = torch.autograd.grad(sdfs_dst, points_dst, grad_outputs=torch.ones_like(sdfs_dst), create_graph=True)[0]
    match_dst = points_dst - sdfs_dst * grad_dst    # eq (19)
    match_world = utils_geometry.transform_points_to(match_dst, R_world_dst, t_world_dst)
    match_src = utils_geometry.transfrom_points_from(match_world, R_world_src, t_world_src)
    if constraint_type == 'point_to_plane':
        cons = torch.sum((points_src - match_src) * grad_src, dim=1)    # eq (20)
    elif constraint_type == 'point_to_point':
        cons = points_src - match_src
    else:
        raise ValueError(f"Invalid constraint type: {constraint_type}!")
    assert cons.shape[0] == n_surf
    loss = torch.mean(cons**2) * residual_weight
    loss_key = f'mips_{src_id}_{dst_id}'
    loss_dict[loss_key] = loss
    return loss_dict

def pairwise_loss_mips(
    grid_atlas: GridAtlas,
    data_loader: DataLoader,
    src_id: int,
    dst_id: int,
    residual_weight=3000,
    use_bound=True,
    constraint_type='point_to_plane',
    device="cuda:0"
):
    """Implement pairwise submap alignment using the method from MIPS-Fusion, eq. 19-22. 

    Args:
        grid_atlas (GridAtlas): GridAtlas containing the pair of submaps to be aligned.
        dataset (SubmapSdf3D): Dataset object containing observations from all submaps.
        src_id (int): ID of source submap. During pairwise alignment, this submap pose will be held fixed.
        dst_id (int): ID of dest submap. During pairwise alignment, this submap pose will be optimized.
    """
    assert src_id < grid_atlas.num_submaps
    assert dst_id < grid_atlas.num_submaps
    data_loader.dataset.getitem_from_single_submap(submap_id=src_id)
    grid_src = grid_atlas.get_submap(src_id)
    grid_dst = grid_atlas.get_submap(dst_id) 
    loss_dict = {}
    model_input, gt = utils.get_batch(data_loader, device)
    # Obtain corrected submap poses
    R_world_src, t_world_src = grid_atlas.updated_submap_pose(src_id, device)
    R_world_dst, t_world_dst = grid_atlas.updated_submap_pose(dst_id, device)
    # Select coords that are on surface and observed in both src and dst submaps
    with torch.no_grad():
        coords_src = model_input['coords_submap'][0]
        coords_world = utils_geometry.transform_points_to(coords_src, R_world_src, t_world_src)
        coords_dst = utils_geometry.transfrom_points_from(coords_world, R_world_dst, t_world_dst)
        gt_sdf = gt['sdf'][0]
        gt_sdf_valid = gt['sdf_valid'][0]
        mask_surf = torch.abs(gt_sdf) < 1e-5
        assert mask_surf.shape == gt_sdf_valid.shape
        mask_valid = torch.logical_and(gt_sdf_valid, mask_surf)
        if use_bound:
            mask_bnd = utils_geometry.coords_in_bound(coords_dst, grid_dst.bound)
            assert mask_bnd.shape == gt_sdf_valid.shape
            mask_valid = torch.logical_and(mask_bnd, mask_valid)
        num_valid = torch.sum(mask_valid)
        logger.debug(f"Select {num_valid} out of {mask_valid.shape[0]} valid surface points.")
        valid_indices = torch.nonzero(mask_valid, as_tuple=False)[:, 0]
    points_src = coords_src[valid_indices, :].requires_grad_(True)
    n_surf = points_src.shape[0]
    sdfs_src = grid_src(points_src)
    with torch.no_grad():
        grad_src = torch.autograd.grad(sdfs_src, points_src, grad_outputs=torch.ones_like(sdfs_src), create_graph=True)[0]
    # Find correspondence in dst submap
    points_world = utils_geometry.transform_points_to(points_src, R_world_src, t_world_src)
    points_dst = utils_geometry.transfrom_points_from(points_world, R_world_dst, t_world_dst)
    sdfs_dst = grid_dst(points_dst)
    with torch.no_grad():
        grad_dst = torch.autograd.grad(sdfs_dst, points_dst, grad_outputs=torch.ones_like(sdfs_dst), create_graph=True)[0]
    match_dst = points_dst - sdfs_dst * grad_dst    # eq (19)
    match_world = utils_geometry.transform_points_to(match_dst, R_world_dst, t_world_dst)
    match_src = utils_geometry.transfrom_points_from(match_world, R_world_src, t_world_src)
    if constraint_type == 'point_to_plane':
        cons = torch.sum((points_src - match_src) * grad_src, dim=1)    # eq (20)
    elif constraint_type == 'point_to_point':
        cons = points_src - match_src
    else:
        raise ValueError(f"Invalid constraint type: {constraint_type}!")
    assert cons.shape[0] == n_surf
    loss = torch.mean(cons**2) * residual_weight
    loss_key = f'mips_{src_id}_{dst_id}'
    loss_dict[loss_key] = loss
    return loss_dict

################################
#      Alignment Methods       #
################################

def align_submap_pair_mips(grid_atlas: GridAtlas, dataset: Dataset, src_id: int, dst_id: int,
                           dataset_name='SubmapSdf3D', num_iters=100, lr=1e-2, residual_weight=3000, 
                           use_bound=True, constraint_type='point_to_plane', device='cuda:0', verbose=True):
    if dataset_name == 'ScanNet':
        pairwise_loss = pairwise_loss_mips_scannet
    elif dataset_name == 'SubmapSdf3D':
        pairwise_loss = pairwise_loss_mips
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}!")
    def loss_func(grid_atlas, loader, src_id, dst_id):
        return pairwise_loss(grid_atlas, loader, src_id, dst_id, residual_weight, use_bound, constraint_type, device)
    loss_tuple = ('mips', loss_func)
    return generic_align_submap_pair(
        grid_atlas, 
        dataset, 
        src_id, 
        dst_id, 
        loss_tuple, 
        num_iters=num_iters,
        lr=lr,
        verbose=verbose
    )


def align_multiple_submaps_mips(
    grid_atlas: GridAtlas,
    dataset: Dataset,
    dataset_name="SubmapSdf3D",
    num_iters=100,
    lr=1e-2,
    residual_weight=3000,
    use_bound=True,
    constraint_type="point_to_plane",
    pose_reg_weight=0,
    pose_thresh_m=1.0,
    pose_thresh_rad=1.0,
    device="cuda:0",
    verbose=True,
):
    if dataset_name == 'ScanNet':
        pairwise_loss = pairwise_loss_mips_scannet
    elif dataset_name == 'SubmapSdf3D':
        pairwise_loss = pairwise_loss_mips
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}!")
    def loss_func(grid_atlas, loader, src_id, dst_id):
        return pairwise_loss(grid_atlas, loader, src_id, dst_id, residual_weight, use_bound, constraint_type, device)
    loss_tuple = ('mips', loss_func)
    return generic_align_multiple_submaps(
        grid_atlas,
        dataset,
        pairwise_loss_tuple=loss_tuple,
        num_iters=num_iters,
        lr=lr,
        verbose=verbose,
        pose_reg_weight=pose_reg_weight,
        pose_thresh_m=pose_thresh_m,
        pose_thresh_rad=pose_thresh_rad,
    )
