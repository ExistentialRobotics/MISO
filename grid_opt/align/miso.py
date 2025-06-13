import numpy as np
import torch
from grid_opt.align.base import *
from grid_opt.utils.utils_geometry import transform_points_to
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

################################
#      Proposed Alignment      #
################################


def pairwise_loss_sdf(
    grid_atlas: GridAtlas,
    data_loader,
    src_id: int,
    dst_id: int,
    align_weight=3000,
    align_loss='L2',
    use_bound=True,
    stability_thresh=0,
    covariance_thresh=None, 
    subsample_points=None,
    gm_scale_sdf=0.1,
    device="cuda:0"
):
    assert src_id < grid_atlas.num_submaps
    assert dst_id < grid_atlas.num_submaps
    model_input, gt = utils.get_batch(data_loader, device)
    index_from, index_to = src_id, dst_id
    submap_from = grid_atlas.get_submap(index_from)
    loss_dict = {}    
    # Retrieve corresponding submap IDs from grid atlas
    submap_idxs = grid_atlas.submap_id_for_kf_batch(kf_ids=model_input['sample_frame_ids'][0, :, 0])             
    sample_indices = torch.nonzero(submap_idxs == src_id, as_tuple=False).squeeze(1)
    if sample_indices.numel() == 0:
        logger.warning(f"No samples found for submap {src_id}!")
        return {}
    coords_kf = model_input['coords_frame'][0, sample_indices, :]        # (num_samples, 3)
    kf_idxs = model_input['sample_frame_ids'][0, sample_indices, 0]      # (num_samples, )
    # Can we get rid of this for loop?
    coords_submap = coords_kf.clone()                                 # (num_samples, 3)
    for kf_id in range(grid_atlas.num_keyframes):
        if grid_atlas.submap_id_for_kf(kf_id) != src_id: continue
        idxs_select = torch.nonzero(kf_idxs == kf_id, as_tuple=False).squeeze(1)
        if idxs_select.numel() == 0: continue
        R_submap_kf, t_submap_kf = grid_atlas.updated_kf_pose_in_submap(kf_id, src_id)
        coords_submap[idxs_select, :] = transform_points_to(
            coords_kf[idxs_select, :],
            R_submap_kf,
            t_submap_kf
        )
    mask_valid = gt['sdf_valid'][0][sample_indices, :]
    submap_to = grid_atlas.get_submap(index_to)
    R_world_from, t_world_from = grid_atlas.updated_submap_pose(index_from, device)
    R_world_to, t_world_to = grid_atlas.updated_submap_pose(index_to, device)
    coords_from = coords_submap
    coords_world = utils_geometry.transform_points_to(coords_from, R_world_from, t_world_from)
    coords_to = utils_geometry.transfrom_points_from(coords_world, R_world_to, t_world_to)
    logger.debug(f"Valid measurements after SDF pruning: {torch.count_nonzero(mask_valid)}.")
    if subsample_points is not None:
        down_points = min(subsample_points, coords_from.shape[0])
        down_indices = np.random.choice(coords_from.shape[0], down_points, replace=False)
        coords_from = coords_from[down_indices, :]
        coords_world = coords_world[down_indices, :]
        coords_to = coords_to[down_indices, :]
        mask_valid = mask_valid[down_indices, :]
        logger.debug(f"Valid measurements after downsample: {torch.count_nonzero(mask_valid)}.")
    if use_bound:
        mask_bnd = utils_geometry.coords_in_bound(coords_to, submap_to.bound)
        assert mask_bnd.shape == mask_valid.shape
        mask_valid = torch.logical_and(mask_bnd, mask_valid)
        logger.debug(f"Valid measurements after bound pruning: {torch.count_nonzero(mask_valid)}.")
    if stability_thresh > 0:
        # mu, _ = torch.min(submap_to.query_stability(coords_to), dim=1, keepdim=True)
        mu_to = submap_to.query_stability(coords_to)[:, [0]]
        mu_from = submap_from.query_stability(coords_from)[:, [0]]
        mask_to = mu_to > stability_thresh
        mask_from = mu_from > stability_thresh
        mask_mu = torch.logical_and(mask_to, mask_from)
        assert mask_mu.shape == mask_valid.shape
        mask_valid = torch.logical_and(mask_mu, mask_valid)
        logger.info(f"Valid measurements after stability pruning: {torch.count_nonzero(mask_valid)}.")
    if covariance_thresh is not None:
        raise NotImplementedError
    valid_indices = torch.nonzero(mask_valid, as_tuple=False)[:, 0]
    p_from = coords_from[valid_indices, :]
    p_to = coords_to[valid_indices, :]
    output_from = submap_from(p_from)
    output_to = submap_to(p_to)
    align_constraint = output_from - output_to
    # Debugging: print historgram over residual distribution
    # e_abs = torch.abs(align_constraint)
    # e_abs_np = e_abs.detach().cpu().numpy()
    # hist, bin_edges = np.histogram(e_abs_np, bins=10)
    # bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # bin_centers = [f"{x:.1e}" for x in bin_centers]
    # hist = 100 * hist / np.sum(hist)
    # hist = [f"{x:.1f}" for x in hist]
    # print(f"{align_loss}BIN: {bin_centers}\n{align_loss}PER: {hist}.")
    loss_key = f'align_sdf_{src_id}_{dst_id}'
    if align_loss == 'L2':
        loss_dict[loss_key] = torch.mean(align_constraint**2) * align_weight
    elif align_loss == 'L1':
        loss_dict[loss_key] = torch.mean(torch.linalg.vector_norm(align_constraint, dim=1)) * align_weight
    elif align_loss == 'GM':
        e = align_constraint.clone().detach()
        w = gm_scale_sdf / (gm_scale_sdf + e**2)**2
        loss_dict[loss_key] = torch.mean(w * align_constraint**2) * align_weight
    else:
        raise ValueError(f"Invalid align loss: {align_loss}!")
    return loss_dict


def pairwise_loss_latent(
    grid_atlas: GridAtlas,
    data_loader,
    src_id: int,
    dst_id: int,
    level: int,
    fdim=4,
    align_weight=3000,
    align_loss='L2',
    use_bound=True,
    stability_thresh=0,
    covariance_thresh=None,
    subsample_points=None,
    trunc_factor=None,
    device="cuda:0"
):
    loss_key = f'align_latent_level{level}_{src_id}_{dst_id}'
    assert src_id < grid_atlas.num_submaps
    assert dst_id < grid_atlas.num_submaps
    start_ch = 0
    end_ch = fdim * (level+1)
    # Get a single batch of coordinate
    # data_loader.dataset.getitem_from_subset_of_submaps([src_id, dst_id])
    # model_input, gt = utils.get_batch(data_loader, device)
    index_from, index_to = src_id, dst_id
    submap_from = grid_atlas.get_submap(index_from)
    submap_to = grid_atlas.get_submap(index_to)
    R_world_from, t_world_from = grid_atlas.updated_submap_pose(index_from, device)
    R_world_to, t_world_to = grid_atlas.updated_submap_pose(index_to, device)
    loss_dict = {}
    coords_from = grid_atlas.coordinates_for_alignment(submap_id=src_id, level=level)
    if subsample_points is not None:
        subsample_points = min(subsample_points, coords_from.shape[0])
        sample_indices = np.random.choice(coords_from.shape[0], subsample_points, replace=False)
        coords_from = coords_from[sample_indices, :]
    coords_world = utils_geometry.transform_points_to(coords_from, R_world_from, t_world_from)
    coords_to = utils_geometry.transfrom_points_from(coords_world, R_world_to, t_world_to)
    num_coords = coords_from.shape[0]
    mask_valid = torch.ones((num_coords, 1)).to(device)
    if use_bound:
        mask_bnd = utils_geometry.coords_in_bound(coords_to, submap_to.bound)
        assert mask_bnd.shape == mask_valid.shape
        mask_valid = torch.logical_and(mask_bnd, mask_valid)
    if stability_thresh > 0:
        num_valid_before = torch.count_nonzero(mask_valid)
        # mu, _ = torch.min(submap_to.query_stability(coords_to), dim=1, keepdim=True)
        mu_to = submap_to.query_stability(coords_to)[:, [0]]
        mu_from = submap_from.query_stability(coords_from)[:, [0]]
        mask_to = mu_to > stability_thresh
        mask_from = mu_from > stability_thresh
        mask_mu = torch.logical_and(mask_to, mask_from)
        assert mask_mu.shape == mask_valid.shape
        mask_valid = torch.logical_and(mask_mu, mask_valid)
        num_valid_after = torch.count_nonzero(mask_valid)
        logger.info(f"Stability pruning: {num_valid_after}/{num_valid_before} measurements remain.")
    if covariance_thresh is not None:
        raise NotImplementedError
    if trunc_factor is not None:
        num_valid_before = torch.count_nonzero(mask_valid)
        sdf_from = submap_from(coords_from)
        mask_dist = torch.abs(sdf_from) < trunc_factor * submap_from.cell_sizes[level]
        assert mask_dist.shape == mask_valid.shape
        mask_valid = torch.logical_and(mask_dist, mask_valid)
        num_valid_after = torch.count_nonzero(mask_valid)
        logger.info(f"Truncation pruning: {num_valid_after}/{num_valid_before} measurements remain.")
    if torch.count_nonzero(mask_valid) == 0:
        logger.warning(f"No valid alignment between submaps {src_id} and {dst_id}!")
        return {loss_key: torch.tensor(0)}
    valid_indices = torch.nonzero(mask_valid, as_tuple=False)[:, 0]
    p_from = coords_from[valid_indices, :]
    p_to = coords_to[valid_indices, :]
    output_from = submap_from.query_feature(p_from)[:, start_ch:end_ch]
    output_to = submap_to.query_feature(p_to)[:, start_ch:end_ch]
    logger.debug(f"Output shape {output_to.shape}")
    align_constraint = output_from - output_to
    # Debugging: print historgram over residual distribution
    # e_abs = torch.linalg.norm(align_constraint, dim=1)
    # e_abs_np = e_abs.detach().cpu().numpy()
    # hist, bin_edges = np.histogram(e_abs_np, bins=10)
    # bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # bin_centers = [f"{x:.1e}" for x in bin_centers]
    # hist = 100 * hist / np.sum(hist)
    # hist = [f"{x:.1f}" for x in hist]
    # print(f"{align_loss}BIN: {bin_centers}\n{align_loss}PER: {hist}.")
    if align_loss == 'L2':
        loss_dict[loss_key] = torch.mean(align_constraint**2) * align_weight
    elif align_loss == 'L1':
        loss_dict[loss_key] = torch.mean(torch.linalg.vector_norm(align_constraint, dim=1)) * align_weight
    elif align_loss == 'cos':
        loss_dict[loss_key] = torch.mean(1.0 - F.cosine_similarity(output_from, output_to, dim=1)) * align_weight
    elif align_loss == 'InfoNCE':
        nce_loss = utils.InfoNCE()
        loss_dict[loss_key] = nce_loss(output_from, output_to) * align_weight
    else:
        raise ValueError(f"Invalid align loss: {align_loss}!")
    return loss_dict

################################
#      Alignment Methods       #
################################

def align_multiple_submaps_hierarchical(
    grid_atlas: GridAtlas,
    dataset: Dataset,
    level_iters=10,
    finetune_iters=10,
    level_thresh=0.0,
    lr=1e-2,
    align_weight=3000,
    align_loss="L2",
    use_bound=True,
    stability_thresh=0,
    subsample_points=None,
    latent_levels=None,
    skip_finetune=False,
    submap_pairs=None,
    pose_reg_weight=0,
    pose_thresh_m=1.0,
    pose_thresh_rad=1.0,
    gm_scale_sdf=0.1,
    device="cuda:0",
    verbose=True,
    save_iterations=False
):
    cpu_time_total, gpu_time_total = 0, 0
    grid_atlas.precompute_coordinates_for_alignment()
    grid_atlas.print_submap_pose_info()
    info = dict()
    # Hierarchical optimzation in latent space
    if latent_levels is None:
        latent_levels = range(grid_atlas.num_levels)
    for curr_level in latent_levels:
        def loss_func(grid_atlas, loader, src_id, dst_id):
            return pairwise_loss_latent(
                grid_atlas, 
                loader, 
                src_id, 
                dst_id, 
                level=curr_level,
                align_weight=align_weight, 
                align_loss=align_loss, 
                use_bound=use_bound, 
                stability_thresh=stability_thresh,
                subsample_points=subsample_points,
                device=device
            )
        loss_name = f'hier_latent_level{curr_level}_{align_loss}'
        loss_tuple = (loss_name, loss_func)
        logger.info(f"Start optimization at level {curr_level}.")
        level_dict = generic_align_multiple_submaps(
            grid_atlas,
            dataset, 
            loss_tuple,
            num_iters=level_iters,
            rel_change_thresh=level_thresh,
            lr=lr,
            submap_pairs=submap_pairs,
            pose_reg_weight=pose_reg_weight,
            pose_thresh_m=pose_thresh_m,
            pose_thresh_rad=pose_thresh_rad,
            verbose=verbose,
            save_iterations=save_iterations
        )
        logger.info(f"End optimization at level {curr_level}. cpu_time_sec={level_dict['cpu_time_sec']:.1f}, gpu_time_sec={level_dict['gpu_time_sec']:.1f}.")
        cpu_time_total += level_dict['cpu_time_sec']
        gpu_time_total += level_dict['gpu_time_sec']
        info[loss_name] = level_dict
    if not skip_finetune:
        # Final optimization in SDF space
        if align_loss=='cos':
            align_loss='L2'
        logger.info(f"Start final optimization in SDF space.")
        def loss_func(grid_atlas, loader, src_id, dst_id):
            return pairwise_loss_sdf(
                grid_atlas, 
                loader, 
                src_id, 
                dst_id,
                align_weight=align_weight,
                align_loss=align_loss,
                use_bound=use_bound, 
                stability_thresh=stability_thresh,
                subsample_points=subsample_points,
                gm_scale_sdf=gm_scale_sdf,
                device=device
            )
        loss_name = f'hier_sdf_{align_loss}'
        loss_tuple = (loss_name, loss_func)
        final_dict = generic_align_multiple_submaps(
            grid_atlas,
            dataset, 
            loss_tuple,
            lr=lr,
            num_iters=finetune_iters,
            submap_pairs=submap_pairs,
            pose_reg_weight=pose_reg_weight,
            pose_thresh_m=pose_thresh_m,
            pose_thresh_rad=pose_thresh_rad,
            verbose=verbose,
            save_iterations=save_iterations
        )
        cpu_time_total += final_dict['cpu_time_sec']
        gpu_time_total += final_dict['gpu_time_sec']
        info[loss_name] = final_dict
    info['cpu_time_sec'] = cpu_time_total
    info['gpu_time_sec'] = gpu_time_total
    return info

def bundle_adjust_multiple_submaps(
    grid_atlas: GridAtlas,
    dataset: Dataset,
    cfg,
    num_epochs=10,
    pose_lr=1e-3,
    map_lr=1e-4,
    verbose=True
):
    # FIXME: this implementation is outdated and should be removed?
    loss_func = cfg_loss(cfg)
    def pose_params():
        params = []
        for submap_id in range(1, grid_atlas.num_submaps):   # Fixing submap 0 at origin
            submap = grid_atlas.get_submap(submap_id)
            params += list(submap.params_for_poses())
        return params
    def map_params():
        params = []
        for submap_id in range(0, grid_atlas.num_submaps):  
            submap = grid_atlas.get_submap(submap_id)
            params += list(submap.params_for_features())
        return params
    param_groups = []
    param_groups.append({'params': pose_params(), 'lr': pose_lr})
    param_groups.append({'params': map_params(), 'lr': map_lr})
    optimizer = optim.Adam(param_groups, lr=1e-3)
    loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)
    epoch = 0
    params_prev = None
    timer = utils.PerfTimer(activate=True)
    timer.reset()
    while epoch <= num_epochs:
        for step, (model_input, gt) in enumerate(loader):
            model_input, gt = utils.prepare_batch(model_input, gt)
            optimizer.zero_grad()
            # Loss 
            total_loss = 0.
            loss_dict = loss_func.compute(grid_atlas, model_input, gt)
            for loss_name, loss in loss_dict.items():
                single_loss = loss.mean()
                total_loss += single_loss
            # Backward step
            if not torch.isnan(total_loss):
                total_loss.backward(retain_graph=False)
                optimizer.step()
            else:
                logger.warning(f"Loss at epoch {epoch} is nan! Skip backward step.")
            if verbose and step % 10 == 0:
                logger.info(f"BA epoch {epoch} step {step} | train_loss={total_loss.item():.2e}.")
        params_curr = [p.clone().detach() for p in pose_params()]
        relchange = utils.relative_param_change(params_curr, params_prev)
        params_prev = params_curr
        epoch += 1
    cpu_time, gpu_time = timer.check()
    logger.info(f"BA_Multi_{loss_name} ends. cpu_time={cpu_time:.2f} sec, gpu_time={gpu_time:.2f} sec.")
    info = {'cpu_time_sec': cpu_time, 'gpu_time_sec': gpu_time}
    return info

