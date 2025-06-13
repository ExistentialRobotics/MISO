import torch
from .models.base_net import BaseNet
from .models.isdf.isdf import iSDF
from .models.grid_net import GridNet
from .models.grid_atlas import GridAtlas
from .loss import BaseLoss, compute_feature_stability_loss, compute_feature_regularization_loss, compute_pose_regularization_loss, compute_pose_trust_region_loss
from .utils.utils_geometry import transform_points_to
from torch.autograd import grad
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class iSDFLoss(BaseLoss):
    def __init__(self, model_name, trunc_weight, trunc_distance, noise_std=0, orien_loss=0, 
                 eik_apply_dist=0.1, eik_weight=0, grad_weight=0, smooth_weight=0.0, smooth_std=0.05,
                 loss_type="L1", slam_mode=True, pose_reg_weight=0, pose_thresh_rad=1.0, pose_thresh_m=1.0, residual_p75=0.05):
        super().__init__()
        self.model_name = model_name
        self.trunc_weight = trunc_weight
        self.trunc_distance = trunc_distance
        self.noise_std = noise_std
        self.orien_loss = orien_loss
        self.eik_apply_dist = eik_apply_dist
        self.eik_weight = eik_weight
        self.grad_weight = grad_weight

        self.smooth_weight = smooth_weight
        self.smooth_std = smooth_std

        self.loss_type = loss_type
        self.cosSim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.slam_mode = slam_mode
        self.pose_reg_weight = pose_reg_weight
        self.pose_thresh_rad = pose_thresh_rad
        self.pose_thresh_m = pose_thresh_m
        self.residual_p75 = residual_p75
        logger.info(f"Configured iSDF loss ({self.loss_type}).")


    def compute(self, model: BaseNet, model_input: dict, gt: dict):
        if self.slam_mode:
            return self.compute_slam(model, model_input, gt)
        else:
            return self.compute_default(model, model_input, gt)
        
    def compute_slam(self, model: iSDF, model_input: dict, gt: dict):
        pc_kf = model_input['coords_frame']
        kf_idxs = model_input['sample_frame_ids'][0, :, 0]
        # Convert from KF frame to world frame
        pc_kf = pc_kf[0]
        pc_world = pc_kf.clone()
        for kf_id in range(model.num_poses):
            idxs_select = torch.nonzero(kf_idxs == kf_id, as_tuple=False).squeeze(1)
            if idxs_select.numel() == 0:
                continue
            R_world_kf, t_world_kf = model.updated_kf_pose_in_world(kf_id)
            pc_world[idxs_select, :] = transform_points_to(
                pc_kf[idxs_select, :],
                R_world_kf,
                t_world_kf
            )
        pc = pc_world.unsqueeze(0)
        bounds = gt['sdf']
        sdf = model(pc[0], noise_std=self.noise_std)
        sdf_loss_mat, free_space_ixs = sdf_loss(
            sdf, bounds, self.trunc_distance, loss_type=self.loss_type, p75=self.residual_p75)
        eik_loss_mat = None
        grad_loss_mat = None
        total_loss, total_loss_mat, losses = tot_loss(
            sdf_loss_mat, grad_loss_mat, eik_loss_mat,
            free_space_ixs, bounds, self.eik_apply_dist,
            self.trunc_weight, self.grad_weight, self.eik_weight,
        )
        loss_dict = {"sdf": total_loss}
        if self.smooth_weight > 0:
            pc_perturb = pc + torch.randn_like(pc) * self.smooth_std
            pc_perturb.requires_grad_()
            sdf_perturb = model(pc_perturb[0], noise_std=self.noise_std)
            sdf_grad_perturb = gradient(pc_perturb, sdf_perturb)
            sdf_grad2 = gradient(pc, sdf)
            smooth_loss = torch.mean((sdf_grad_perturb[:, :, 0] - sdf_grad2[:, :, 0]).norm(2, dim=-1))
            loss_dict['smooth'] = self.smooth_weight * smooth_loss
        # Pose norm regularization for iSDF
        if self.pose_reg_weight > 0:
            rot_norm = torch.linalg.norm(model.rotation_corrections, dim=1)
            loss_dict[f'trust_region_R'] = self.pose_reg_weight * torch.sum(torch.nn.functional.relu(rot_norm - self.pose_thresh_rad))
            tran_norm = torch.linalg.norm(model.translation_corrections.squeeze(2), dim=1)
            loss_dict[f'trust_region_t'] = self.pose_reg_weight * torch.sum(torch.nn.functional.relu(tran_norm - self.pose_thresh_m))
        
        return loss_dict


    def compute_default(self, model: BaseNet, model_input: dict, gt: dict):
        
        pc, norm_sample = model_input['coords'], model_input['normals']
        
        bounds, grad_vec = gt['sdf'], gt['grad_vec']

        do_sdf_grad = self.eik_weight != 0 or self.grad_weight != 0 or self.smooth_weight != 0
        # do_sdf_grad = False
        if do_sdf_grad:
            pc.requires_grad_()
        
        sdf = model(pc[0], noise_std=self.noise_std)

        sdf_grad = None
        if do_sdf_grad:
            sdf_grad = gradient(pc, sdf)
        
        # compute loss
        sdf_loss_mat, free_space_ixs = sdf_loss(
            sdf, bounds, self.trunc_distance, loss_type=self.loss_type, p75=self.residual_p75)
        # print('sdf_loss_mat:', sdf_loss_mat.shape)
        eik_loss_mat = None
        if self.eik_weight != 0:
            eik_loss_mat = torch.abs(sdf_grad.norm(2, dim=-1) - 1)
        
        grad_loss_mat = None
        if self.grad_weight != 0:
            sdf_grad = sdf_grad.reshape(sdf_grad.shape[0], grad_vec.shape[1], -1, 3)

            pred_norms = sdf_grad[:, :, 0]
            surf_loss_mat = 1 - self.cosSim(pred_norms, norm_sample)
            
            grad_vec[torch.where(grad_vec[..., 0].isnan())] = \
                norm_sample[torch.where(grad_vec[..., 0].isnan())[:2]]
            
            grad_loss_mat = 1 - self.cosSim(grad_vec, sdf_grad[:, :, 1:])
            grad_loss_mat = torch.cat(
                (surf_loss_mat[:, :, None], grad_loss_mat), dim=2)
            grad_loss_mat = grad_loss_mat.reshape(grad_loss_mat.shape[0], -1, 1)
            if self.orien_loss:
                grad_loss_mat = (grad_loss_mat > 1).float()
        
        total_loss, total_loss_mat, losses = tot_loss(
            sdf_loss_mat, grad_loss_mat, eik_loss_mat,
            free_space_ixs, bounds, self.eik_apply_dist,
            self.trunc_weight, self.grad_weight, self.eik_weight,
        )
        loss_dict = {"sdf": total_loss}

        # compute the gradient smoothing loss
        if self.smooth_weight > 0:
            pc_perturb = pc + torch.randn_like(pc) * self.smooth_std
            pc_perturb.requires_grad_()
            sdf_perturb = model(pc_perturb[0], noise_std=self.noise_std)
            sdf_grad_perturb = gradient(pc_perturb, sdf_perturb)
            sdf_grad2 = gradient(pc, sdf)
            smooth_loss = torch.mean((sdf_grad_perturb[:, :, 0] - sdf_grad2[:, :, 0]).norm(2, dim=-1))
            loss_dict['smooth'] = self.smooth_weight * smooth_loss

        return loss_dict


class iSDFLossSubmap(BaseLoss):
    def __init__(self, model_name, trunc_weight, trunc_distance, noise_std, orien_loss, 
                 eik_apply_dist, eik_weight=0, grad_weight=0, smooth_weight=0.1, smooth_std=0.05,
                 loss_type="L1", feat_reg_weight=0, slam_mode=False, 
                 pose_reg_weight=1e3, pose_thresh_rad=1.0, pose_thresh_m=1.0):
        super().__init__()
        self.model_name = model_name
        self.trunc_weight = trunc_weight
        self.trunc_distance = trunc_distance
        self.noise_std = noise_std
        self.orien_loss = orien_loss
        self.eik_apply_dist = eik_apply_dist
        self.eik_weight = eik_weight
        self.grad_weight = grad_weight

        self.smooth_weight = smooth_weight
        self.smooth_std = smooth_std

        self.loss_type = loss_type
        self.cosSim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.feat_reg_weight = feat_reg_weight
        self.pose_reg_weight = pose_reg_weight
        self.pose_thresh_rad = pose_thresh_rad
        self.pose_thresh_m = pose_thresh_m
        self.slam_mode = slam_mode
    
    def compute_submap_loss(self, model, model_input: dict, gt: dict):
        pc = model_input['coords']
        bounds = gt['sdf']

        do_sdf_grad = self.eik_weight != 0 or self.grad_weight != 0
        # do_sdf_grad = False
        if do_sdf_grad:
            pc.requires_grad_()
        
        sdf = model(pc[0], noise_std=self.noise_std)

        sdf_grad = None
        if do_sdf_grad:
            sdf_grad = gradient(pc, sdf)
        
        # compute loss
        sdf_loss_mat, free_space_ixs = sdf_loss(
            sdf, bounds, self.trunc_distance, loss_type=self.loss_type)
        # print('sdf_loss_mat:', sdf_loss_mat.shape)
        eik_loss_mat = None
        if self.eik_weight != 0:
            eik_loss_mat = torch.abs(sdf_grad.norm(2, dim=-1) - 1)
        
        grad_loss_mat = None
        if self.grad_weight != 0:
            raise NotImplementedError
        
        total_loss, total_loss_mat, losses = tot_loss(
            sdf_loss_mat, grad_loss_mat, eik_loss_mat,
            free_space_ixs, bounds, self.eik_apply_dist,
            self.trunc_weight, self.grad_weight, self.eik_weight,
        )
        loss_dict = {"sdf": total_loss}

        # compute the gradient smoothing loss
        if self.smooth_weight > 0:
            raise NotImplementedError

        # Stability loss
        if isinstance(model, GridNet):
            mu_loss_dict = compute_feature_stability_loss(model, pc[0])
            loss_dict.update(mu_loss_dict)

        # Feature norm reguarlization
        if self.feat_reg_weight > 0 and isinstance(model, GridNet):
            reg_loss_dict = compute_feature_regularization_loss(model, weight=self.feat_reg_weight)
            loss_dict.update(reg_loss_dict)

        # Pose norm regularization
        if self.pose_reg_weight > 0 and isinstance(model, GridNet):
            # pose_loss_dict = compute_pose_regularization_loss(model, weight=self.pose_reg_weight)
            pose_loss_dict = compute_pose_trust_region_loss(
                model, 
                thresh_rad=self.pose_thresh_rad,
                thresh_m=self.pose_thresh_m,
                weight=self.pose_reg_weight
            )  # Trust region based regularization
            loss_dict.update(pose_loss_dict)

        return loss_dict
    
    def compute(self, model: GridAtlas, model_input: dict, gt: dict):
        losses = {}
        # Separately compute iSDF loss, one for each submap
        indices_m = model_input['submap_idxs'][0, :, 0]
        for submap_id in range(model.num_submaps):
            indices = torch.nonzero(indices_m == submap_id, as_tuple=False).squeeze(1)
            if indices.numel() == 0:
                continue
            # TODO: include normals and grad_vec
            submap_input, submap_gt = {}, {}
            submap_gt['sdf'] = gt['sdf'][:, indices, :]
            submap_gt['sdf_valid'] = gt['sdf_valid'][:, indices, :]
            if self.slam_mode:
                # Recover the coords in submap frame using the optimized keyframe poses in submap
                coords_kf = model_input['coords_kf'][0, indices, :]        # (num_samples, 3)
                kf_idxs = model_input['keyframe_idxs'][0, indices, 0]      # (num_samples, )
                coords_submap = coords_kf.clone()                          # (num_samples, 3)
                # Can we get rid of this for loop to speed up?
                for kf_id in range(model.num_keyframes):
                    idxs_select = torch.nonzero(kf_idxs == kf_id, as_tuple=False).squeeze(1)
                    if idxs_select.numel() == 0:
                        continue
                    R_submap_kf, t_submap_kf = model.updated_kf_pose_in_submap(kf_id, submap_id)
                    coords_submap[idxs_select, :] = transform_points_to(
                        coords_kf[idxs_select, :],
                        R_submap_kf,
                        t_submap_kf
                    )
                submap_input['coords'] = coords_submap.unsqueeze(0)      # (1, num_samples, 3)
            else:
                # Recover the coords in submap frame provided by input
                submap_input['coords'] = model_input['coords_submap'][:, indices, :]
            submap_losses = self.compute_submap_loss(model.get_submap(submap_id), submap_input, submap_gt)
            for key, val in submap_losses.items():
                losses[f'submap{submap_id}_{key}'] = val
        return losses
    

def full_sdf_loss(sdf, target_sdf, free_space_factor=5.0):
    """
    For samples that lie in free space before truncation region:
        loss(sdf_pred, sdf_gt) =  { max(0, sdf_pred - sdf_gt), if sdf_pred >= 0
                                  { exp(-sdf_pred) - 1, if sdf_pred < 0

    For samples that lie in truncation region:
        loss(sdf_pred, sdf_gt) = sdf_pred - sdf_gt
    """

    free_space_loss_mat = torch.max(
        torch.nn.functional.relu(sdf - target_sdf),
        torch.exp(-free_space_factor * sdf) - 1.
    )
    trunc_loss_mat = sdf - target_sdf

    return free_space_loss_mat, trunc_loss_mat


def sdf_loss(sdf, bounds, t, loss_type="L1", p75=0.05):
    """
        params:
        sdf: predicted sdf values.
        bounds: upper bound on abs(sdf)
        t: truncation distance up to which the sdf value is directly supevised.
        loss_type: L1 or L2 loss.
    """
    free_space_loss_mat, trunc_loss_mat = full_sdf_loss(sdf, bounds)

    free_space_ixs = bounds > t
    free_space_loss_mat[~free_space_ixs] = 0.
    trunc_loss_mat[free_space_ixs] = 0.

    if loss_type == 'GM':
        sdf_loss_mat = trunc_loss_mat
    else:
        sdf_loss_mat = free_space_loss_mat + trunc_loss_mat

    if loss_type == "L1":
        sdf_loss_mat = torch.abs(sdf_loss_mat)
    elif loss_type == "L2":
        sdf_loss_mat = torch.square(sdf_loss_mat)
    elif loss_type == "GM":
        # GM for truncation region, L1 for free space
        raise ValueError("GM loss is deprecated.")
        gm_sigma = 1.5 * p75
        rsq = torch.square(trunc_loss_mat)
        trunc_loss_mat = rsq / (rsq + gm_sigma**2)
        free_space_loss_mat = torch.abs(free_space_loss_mat)
        sdf_loss_mat = free_space_loss_mat + trunc_loss_mat
    else:
        raise ValueError("Must be L1 or L2")

    return sdf_loss_mat, free_space_ixs

def tot_loss(
    sdf_loss_mat, grad_loss_mat, eik_loss_mat,
    free_space_ixs, bounds, eik_apply_dist,
    trunc_weight, grad_weight, eik_weight,
):
    sdf_loss_mat[~free_space_ixs] *= trunc_weight
    # print("zero losses",
    #       sdf_loss_mat.numel() - sdf_loss_mat.nonzero().shape[0])

    losses = {"sdf_loss": sdf_loss_mat.mean().item()}
    tot_loss_mat = sdf_loss_mat

    # surface normal loss
    if grad_loss_mat is not None:
        tot_loss_mat = tot_loss_mat + grad_weight * grad_loss_mat
        losses["grad_loss"] = grad_loss_mat.mean().item()

    # eikonal loss
    if eik_loss_mat is not None:
        # print("eik_loss_mat", eik_loss_mat.shape)
        # print("bounds", bounds.shape)
        # print("eik_apply_dist", eik_apply_dist)
        eik_loss_mat[bounds.squeeze(-1) < eik_apply_dist] = 0.
        eik_loss_mat = eik_loss_mat * eik_weight
        tot_loss_mat = tot_loss_mat + eik_loss_mat
        losses["eikonal_loss"] = eik_loss_mat.mean().item()

    tot_loss = tot_loss_mat.mean()
    losses["total_loss"] = tot_loss

    return tot_loss, tot_loss_mat, losses

def gradient(inputs, outputs):
    d_points = torch.ones_like(
        outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad
