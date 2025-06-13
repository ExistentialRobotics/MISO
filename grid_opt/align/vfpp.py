import numpy as np
import torch
from grid_opt.align.base import *
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

##########################
#      Vox-Fusion++      #
##########################

def pairwise_loss_vfpp_scannet(
    grid_atlas: GridAtlas,
    data_loader: DataLoader,
    src_id: int,
    dst_id: int,
    sdf_weight=3000,
    use_bound=True,
    stability_thresh=0,
    subsample_points=None,
    trunc_dist=0.15,
    device="cuda:0"
):
    """Implement the inter-map optimization method as in VoxFusion++, see eq (9)-(10).

    Args:
        grid_atlas (GridAtlas): GridAtlas containing the pair of submaps to be aligned.
        data_loader (DataLoader): torch data loader containing the dataset object with observations from all submaps.
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
    # Get coordinates that belong to src submap
    submap_idxs = model_input['submap_idxs'][0,:,0]
    sample_indices = torch.nonzero(submap_idxs == src_id, as_tuple=False).squeeze(1)
    if sample_indices.numel() == 0:
        logger.warning(f"No samples found for submap {src_id}!")
        return {}
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
    mask_valid = torch.abs(gt_sdf) < trunc_dist
    logger.debug(f"Valid measurements after SDF pruning: {torch.count_nonzero(mask_valid)}.")
    if subsample_points is not None:
        down_points = min(subsample_points, coords_src.shape[0])
        down_indices = np.random.choice(coords_src.shape[0], down_points, replace=False)
        coords_src = coords_src[down_indices, :]
        coords_world = coords_world[down_indices, :]
        coords_dst = coords_dst[down_indices, :]
        gt_sdf = gt_sdf[down_indices, :]
        mask_valid = mask_valid[down_indices, :]
        logger.debug(f"Valid measurements after downsample: {torch.count_nonzero(mask_valid)}.")
    if use_bound:
        mask_bnd = utils_geometry.coords_in_bound(coords_dst, grid_dst.bound)
        assert mask_bnd.shape == mask_valid.shape
        mask_valid = torch.logical_and(mask_bnd, mask_valid)
        logger.debug(f"Valid measurements after bound pruning: {torch.count_nonzero(mask_valid)}.")
    if stability_thresh > 0:
        mu, _ = torch.min(grid_dst.query_stability(coords_dst), dim=1, keepdim=True)
        mask_mu = mu > stability_thresh
        assert mask_mu.shape == mask_valid.shape
        mask_valid = torch.logical_and(mask_mu, mask_valid)
        logger.debug(f"Valid measurements after stability pruning: {torch.count_nonzero(mask_valid)}.")
    pred_sdf = grid_dst(coords_dst)
    sdf_constraint = torch.where(
        mask_valid == 1,
        pred_sdf - gt_sdf,
        torch.zeros_like(pred_sdf)
    )
    loss = torch.mean(sdf_constraint**2) * sdf_weight
    loss_key = f'vfpp_{src_id}_{dst_id}'
    loss_dict[loss_key] = loss
    return loss_dict

def pairwise_loss_vfpp(
    grid_atlas: GridAtlas,
    data_loader: DataLoader,
    src_id: int,
    dst_id: int,
    sdf_weight=3000,
    use_bound=True,
    stability_thresh=0,
    subsample_points=None,
    device="cuda:0"
):
    """Implement the inter-map optimization method as in VoxFusion++, see eq (9)-(10).

    Args:
        grid_atlas (GridAtlas): GridAtlas containing the pair of submaps to be aligned.
        data_loader (DataLoader): torch data loader containing the dataset object with observations from all submaps.
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
    # Subsample observations
    coords_src = model_input['coords_submap'][0]
    sample_indices = range(coords_src.shape[0])
    if subsample_points is not None:
        subsample_points = min(subsample_points, coords_src.shape[0])
        sample_indices = np.random.choice(coords_src.shape[0], subsample_points, replace=False)
    # Transform observed coordinates in src submap to dst submap
    coords_src = coords_src[sample_indices]
    coords_world = utils_geometry.transform_points_to(coords_src, R_world_src, t_world_src)
    coords_dst = utils_geometry.transfrom_points_from(coords_world, R_world_dst, t_world_dst)
    # Compute SDF loss based on dst feature grid
    gt_sdf = gt['sdf'][0][sample_indices, :]
    gt_sdf_valid = gt['sdf_valid'][0][sample_indices, :]
    mask_valid = gt_sdf_valid
    if use_bound:
        mask_bnd = utils_geometry.coords_in_bound(coords_dst, grid_dst.bound)
        assert mask_bnd.shape == gt_sdf_valid.shape
        mask_valid = torch.logical_and(mask_bnd, mask_valid)
    if stability_thresh > 0:
        num_valid_before = torch.count_nonzero(mask_valid)
        mu, _ = torch.min(grid_dst.query_stability(coords_dst), dim=1, keepdim=True)
        mask_mu = mu > stability_thresh
        assert mask_mu.shape == mask_valid.shape
        mask_valid = torch.logical_and(mask_mu, mask_valid)
        num_valid_after = torch.count_nonzero(mask_valid)
        logger.debug(f"Stability pruning: {num_valid_after}/{num_valid_before} measurements remain.")
    pred_sdf = grid_dst(coords_dst)
    sdf_constraint = torch.where(
        mask_valid == 1,
        pred_sdf - gt_sdf,
        torch.zeros_like(pred_sdf)
    )
    loss = torch.mean(sdf_constraint**2) * sdf_weight
    loss_key = f'vfpp_{src_id}_{dst_id}'
    loss_dict[loss_key] = loss
    return loss_dict

################################
#      Alignment Methods       #
################################

def align_submap_pair_vfpp(
    grid_atlas: GridAtlas,
    dataset: Dataset,
    src_id: int,
    dst_id: int,
    dataset_name="SubmapSdf3D",
    num_iters=10,
    lr=1e-2,
    sdf_weight=3000,
    use_bound=True,
    stability_thresh=0,
    subsample_points=None,
    device="cuda:0",
    verbose=True,
):
    if dataset_name == 'ScanNet':
        pairwise_loss = pairwise_loss_vfpp_scannet
    elif dataset_name == 'SubmapSdf3D':
        pairwise_loss = pairwise_loss_vfpp
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}!")
    def loss_func(grid_atlas, loader, src_id, dst_id):
        return pairwise_loss(
            grid_atlas,
            loader,
            src_id,
            dst_id,
            sdf_weight=sdf_weight,
            use_bound=use_bound,
            stability_thresh=stability_thresh,
            subsample_points=subsample_points,
            device=device,
        )
    loss_tuple = ('vfpp', loss_func)
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


def align_multiple_submaps_vfpp(
    grid_atlas: GridAtlas,
    dataset: Dataset,
    dataset_name='SubmapSdf3D',
    num_iters=10,
    lr=1e-2,
    sdf_weight=3000,
    use_bound=True,
    stability_thresh=0,
    subsample_points=None,
    pose_reg_weight=0,
    pose_thresh_m=1.0,
    pose_thresh_rad=1.0,
    device="cuda:0",
    verbose=True
):
    if dataset_name == 'ScanNet':
        pairwise_loss = pairwise_loss_vfpp_scannet
    elif dataset_name == 'SubmapSdf3D':
        pairwise_loss = pairwise_loss_vfpp
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}!")
    def loss_func(grid_atlas, loader, src_id, dst_id):
        return pairwise_loss(
            grid_atlas,
            loader,
            src_id,
            dst_id,
            sdf_weight=sdf_weight,
            use_bound=use_bound,
            stability_thresh=stability_thresh,
            subsample_points=subsample_points,
            device=device,
        )
    loss_tuple = ('vfpp', loss_func)
    return generic_align_multiple_submaps(
        grid_atlas,
        dataset, 
        loss_tuple,
        num_iters=num_iters,
        lr=lr,
        verbose=verbose,
        pose_reg_weight=pose_reg_weight,
        pose_thresh_m=pose_thresh_m,
        pose_thresh_rad=pose_thresh_rad,
    )
