import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from grid_opt.configs import *
import grid_opt.utils.utils as utils
from grid_opt.models.grid_atlas import GridAtlas
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def grid_atlas_pose_l2_loss(model: GridAtlas, weight=1e3):
    loss_dict = {}
    for submap_id in range(model.num_submaps):
        rot_norm = torch.sum(model.rotation_corrections[submap_id]**2)
        loss_dict[f'submap{submap_id}_l2_reg_R'] = weight * rot_norm
        tran_norm = torch.sum(model.translation_corrections[submap_id]**2)
        loss_dict[f'submap{submap_id}_l2_reg_t'] = weight * tran_norm
    return loss_dict

def grid_atlas_pose_trust_region_loss(model: GridAtlas, thresh_rad, thresh_m, weight=1e3):
    loss_dict = {}
    for submap_id in range(model.num_submaps):
        rot_norm = torch.linalg.norm(model.rotation_corrections[submap_id])
        loss_dict[f'submap{submap_id}_trust_region_R'] = weight * torch.nn.functional.relu(rot_norm - thresh_rad)
        tran_norm = torch.linalg.norm(model.translation_corrections[submap_id])
        loss_dict[f'submap{submap_id}_trust_region_t'] = weight * torch.nn.functional.relu(tran_norm - thresh_m)
    return loss_dict

def iteration_results_helper(model: GridAtlas):
    """A function handle to be provided to the training loop.
    This function is called at each training iteration to retrieve the solution at that iteration.
    """
    assert isinstance(model, GridAtlas), "Model must be an instance of GridAtlas."
    submap_poses = torch.zeros((model.num_submaps, 4, 4), dtype=torch.float32, device=model.device)
    for submap_id in range(model.num_submaps):
        R, t = model.updated_submap_pose(submap_id)
        T = utils_geometry.pose_matrix(R, t)
        submap_poses[submap_id] = T
    return submap_poses.detach()

def generic_align_submap_pair(
    grid_atlas: GridAtlas,
    dataset: Dataset,
    src_id: int,
    dst_id: int,
    pairwise_loss_tuple,
    num_iters=10,
    lr=1e-2,
    rel_change_thresh=0,
    verbose=True
):
    assert src_id < grid_atlas.num_submaps
    assert dst_id < grid_atlas.num_submaps
    grid_src = grid_atlas.get_submap(src_id)
    grid_dst = grid_atlas.get_submap(dst_id) 
    loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)
    param_groups = []
    param_groups.append({'params': grid_atlas.params_for_submap_pose(dst_id), 'lr': lr})
    optimizer = optim.Adam(param_groups, lr=lr)
    logger.info(f"Alignment starts. Target map rotation:\n {grid_dst.rotation_corrections} \n Translation:\n {grid_dst.translation_corrections}.")
    iter = 0
    loss_name, loss_func = pairwise_loss_tuple
    pose_params_prev = None
    timer = utils.PerfTimer(activate=True)
    timer.reset()
    while iter <= num_iters:
        optimizer.zero_grad()
        loss_dict = loss_func(grid_atlas, loader, src_id, dst_id)
        total_loss = sum([single_loss for loss_key, single_loss in loss_dict.items()])
        if not torch.isnan(total_loss):
            total_loss.backward(retain_graph=False)
            optimizer.step()
        else:
            logger.warning(f"Loss at iter {iter} is nan! Skip backward step.")
        pose_params_curr = [p.clone().detach() for p in grid_dst.params_for_poses()]
        pose_relchange = utils.relative_param_change(pose_params_curr, pose_params_prev)
        pose_params_prev = pose_params_curr
        if verbose:
            logger.info(f"AlignPair_{loss_name} iteration {iter}: loss = {total_loss.item():.2e}, pose_relchange={pose_relchange:.2e}")
        if pose_relchange < rel_change_thresh:
            break
        iter += 1
    cpu_time, gpu_time = timer.check()
    logger.debug(f"Alignment ends. Target map rotation:\n {grid_dst.rotation_corrections} \n Translation:\n {grid_dst.translation_corrections}.")
    logger.info(f"AlignPair_{loss_name} ends. cpu_time={cpu_time:.2f} sec, gpu_time={gpu_time:.2f} sec.")
    info = {'cpu_time_sec': cpu_time, 'gpu_time_sec': gpu_time}
    return info

def generic_align_multiple_submaps(
    grid_atlas: GridAtlas,
    dataset: Dataset,
    pairwise_loss_tuple,
    num_iters=10,
    lr=1e-2,
    rel_change_thresh=0,
    submap_pairs=None,
    check_intersection=True,
    pose_reg_weight=0,
    pose_thresh_rad=1.0,
    pose_thresh_m=1.0,
    verbose=True,
    save_iterations=False
):
    def pose_params():
        params = []
        for submap_id in range(1, grid_atlas.num_submaps):   # Fixing submap 0 at origin
            params += list(grid_atlas.params_for_submap_pose(submap_id))
        return params
    param_groups = []
    param_groups.append({'params': pose_params(), 'lr': lr})
    optimizer = optim.Adam(param_groups, lr=lr)
    loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)
    iter = 0
    loss_name, loss_func = pairwise_loss_tuple
    params_prev = None
    # By default, try to align all pairs of submaps
    if submap_pairs is None:
        submap_pairs = []
        for src_id in range(grid_atlas.num_submaps):
            for dst_id in range(src_id+1, grid_atlas.num_submaps):
                submap_pairs.append((src_id, dst_id))
    timer = utils.PerfTimer(activate=True)
    timer.reset()
    iteration_results = dict()
    if save_iterations:
        logger.info("Saving intermediate results at each iteration. This may slow down optimization.")
    while iter <= num_iters:
        if save_iterations:
            iteration_results[iter] = iteration_results_helper(grid_atlas)
        optimizer.zero_grad()
        loss_dict = {}
        for src_id, dst_id in submap_pairs:
            # Check submap intersection
            if check_intersection and grid_atlas.check_submap_intersection(src_id, dst_id) == False:
                logger.info(f"Skip pair {src_id}, {dst_id}: no intersection.")
                continue
            pair_loss_dict = loss_func(grid_atlas, loader, src_id, dst_id)
            for key, val in pair_loss_dict.items():
                if torch.isnan(val):
                    logger.warning(f"pairwise loss {key} is nan!")
                    pair_loss_dict[key] = torch.nan_to_num(val)
            loss_dict.update(pair_loss_dict)
        if pose_reg_weight > 0:
            pose_reg_losses = grid_atlas_pose_trust_region_loss(grid_atlas, thresh_rad=pose_thresh_rad, thresh_m=pose_thresh_m, weight=pose_reg_weight)
            loss_dict.update(pose_reg_losses)
        total_loss = sum([single_loss for loss_key, single_loss in loss_dict.items()])
        if not torch.isnan(total_loss):
            total_loss.backward(retain_graph=False)
            optimizer.step()
        else:
            logger.warning(f"Loss at iter {iter} is nan! Skip backward step.")
        params_curr = [p.clone().detach() for p in pose_params()]
        relchange = utils.relative_param_change(params_curr, params_prev)
        params_prev = params_curr
        if verbose:
            logger.info(f"AlignMulti_{loss_name} iteration {iter}: loss = {total_loss.item():.2e}, pose_relchange={relchange:.2e}, lr={lr:.2e}")
        if relchange < rel_change_thresh:
            break
        iter += 1
    cpu_time, gpu_time = timer.check()
    logger.info(f"AlignMulti_{loss_name} ends. cpu_time={cpu_time:.2f} sec, gpu_time={gpu_time:.2f} sec.")
    info = {'cpu_time_sec': cpu_time, 'gpu_time_sec': gpu_time, 'iteration_results': iteration_results}
    return info
