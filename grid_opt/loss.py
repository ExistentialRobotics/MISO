import numpy as np
import torch
import torch.nn.functional as F
from .diff import gradient3d
from .models.base_net import BaseNet
from .models.grid_net import GridNet
from .models.grid_atlas import GridAtlas
import grid_opt.utils.utils_geometry as utils_geometry

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseLoss:
    def __init__(self):
        super().__init__()

    def compute(self, model, model_input: dict, gt: dict) -> dict:
        """Should be implemented in base class.

        Args:
            model: network
            model_input (dict): input to model
            gt (dict): supervisoin to model

        Returns:
            dict: a dict of different loss functions
        """
        raise NotImplementedError


class SdfLoss2D(BaseLoss):
    def __init__(self, sdf_weight=3e3):
        super().__init__()
        self.sdf_weight = sdf_weight

    def compute(self, model: BaseNet, model_input: dict, gt: dict) -> dict:
        coords = model_input['coords']
        gt_sdf = gt['sdf']
        assert coords.ndim == 3 and gt_sdf.ndim == 3
        assert coords.shape[0] == 1 and gt_sdf.shape[0] == 1
        pred_sdf = model(coords[0])
        pred_sdf = pred_sdf.unsqueeze(0)
        sdf_constraint = pred_sdf - gt_sdf
        return {'sdf': torch.mean(sdf_constraint**2) * self.sdf_weight}


class SdfLoss3D(BaseLoss):
    def __init__(self, sdf_weight=3e3):
        super().__init__()
        self.sdf_weight = sdf_weight

    def compute(self, model: BaseNet, model_input: dict, gt: dict) -> dict:
        coords = model_input['coords']
        gt_sdf = gt['sdf']
        gt_sdf_valid = gt['sdf_valid']
        assert coords.ndim == 3 and gt_sdf.ndim == 3
        assert coords.shape[0] == 1 and gt_sdf.shape[0] == 1
        pred_sdf = model(coords[0])
        pred_sdf = pred_sdf.unsqueeze(0)

        sdf_constraint = torch.where(
            gt_sdf_valid == 1,
            pred_sdf - gt_sdf,
            torch.zeros_like(pred_sdf)
        )
        
        loss_dict = {'sdf': torch.mean(sdf_constraint**2) * self.sdf_weight}
        return loss_dict
    

class TsdfLoss3D(BaseLoss):
    def __init__(self, sdf_weight=3e3, sign_weight=1e2, eik_weight=5e1, trunc_dist=0.15, grad_method='autograd', finite_diff_eps=1e-2):
        super().__init__()
        self.sdf_weight = sdf_weight
        self.sign_weight = sign_weight
        self.eik_weight = eik_weight
        self.trunc_dist = trunc_dist
        self.grad_method = grad_method
        self.finite_diff_eps = finite_diff_eps

    def compute(self, model, model_input, gt):
        coords = model_input['coords'][0]
        gt_sdf = gt['sdf'][0]
        gt_sdf_valid = gt['sdf_valid'][0]
        gt_sdf_sign = gt['sdf_sign'][0]
        assert coords.ndim == 2 and gt_sdf.ndim == 2
        pred_sdf = model(coords)
        pred_sdf = pred_sdf
        # SDF prediction loss
        sdf_constraint = torch.where(
            gt_sdf_valid == 1,
            pred_sdf - gt_sdf,
            torch.zeros_like(pred_sdf)
        )
        loss_dict = {'sdf': torch.mean(sdf_constraint**2) * self.sdf_weight}
        # Sign prediction loss
        if self.sign_weight > 0:
            assert self.trunc_dist is not None
            # Positive free space
            # pos_exp_constraint = torch.where(
            #     gt_sdf_sign == 1,
            #     torch.exp(-1e2 * pred_sdf) - 1.0, 
            #     torch.zeros_like(pred_sdf)
            # )  # Exponential punishment if predicting negative SDF: from GO-SURF
            pos_trunc_constraint = torch.where(
                gt_sdf_sign == 1,
                self.trunc_dist - pred_sdf,
                torch.zeros_like(pred_sdf)
            )  # In free space, should predict value greater than truncation distance
            pos_constraint = torch.maximum(torch.zeros_like(pred_sdf), pos_trunc_constraint)
            loss_dict['pos_space'] = torch.mean(pos_constraint) * self.sign_weight
            
            # Negative free space
            # neg_exp_constraint = torch.where(
            #     gt_sdf_sign == -1,
            #     torch.exp(1e2 * pred_sdf) - 1.0, 
            #     torch.zeros_like(pred_sdf)
            # )  # Exponential punishment if predicting positive SDF: from GO-SURF
            neg_trunc_constraint = torch.where(
                gt_sdf_sign == -1,
                pred_sdf + self.trunc_dist,
                torch.zeros_like(pred_sdf)
            )  # Should predict value smaller than negative truncation distance
            neg_constraint = torch.maximum(torch.zeros_like(pred_sdf), neg_trunc_constraint)
            loss_dict['neg_space'] = torch.mean(neg_constraint) * self.sign_weight


        # Eikonal regularization
        # Enfoce by randomly sampling within model bound
        if self.eik_weight > 0:
            N = gt_sdf.shape[0] 
            bound = model.bound.detach().cpu().numpy()
            xs = np.reshape(np.random.uniform(bound[0,0], bound[0,1], N), (N,1))
            ys = np.reshape(np.random.uniform(bound[1,0], bound[1,1], N), (N,1))
            zs = np.reshape(np.random.uniform(bound[2,0], bound[2,1], N), (N,1))
            x_np = np.concatenate([xs, ys, zs], axis=1)
            x = torch.from_numpy(x_np).to(gt_sdf)
            x.requires_grad_(True)
            # x = coords
            gradient = gradient3d(x, model, method=self.grad_method, finite_diff_eps=self.finite_diff_eps, create_graph=True)
            grad_constraint = gradient.norm(dim=-1) - 1
            loss_dict['eik'] = torch.mean(grad_constraint**2) * self.eik_weight

        return loss_dict

def compute_feature_regularization_loss(model: GridNet, weight=1.0):
    loss_dict = {}
    for level in range(model.num_levels):
        loss_dict[f'feat_reg_level{level}'] = torch.mean(model.features[level].feature**2) * weight
    return loss_dict

def compute_pose_regularization_loss(model: GridNet, weight=1.0):
    """L2  regularization on pose corrections."""
    loss_dict = {}
    loss_dict[f'pose_l2_reg_R'] = torch.mean(model.rotation_corrections**2) * weight
    loss_dict[f'pose_l2_reg_t'] = torch.mean(model.translation_corrections**2) * weight
    return loss_dict

def compute_pose_trust_region_loss(model: GridNet, thresh_rad, thresh_m, weight=1e3):
    """Trust-region regularization on pose corrections."""
    loss_dict = {}
    rot_norm = torch.linalg.norm(model.rotation_corrections, dim=1)
    loss_dict[f'trust_region_R'] = weight * torch.sum(torch.nn.functional.relu(rot_norm - thresh_rad))
    tran_norm = torch.linalg.norm(model.translation_corrections.squeeze(2), dim=1)
    loss_dict[f'trust_region_t'] = weight * torch.sum(torch.nn.functional.relu(tran_norm - thresh_m))
    return loss_dict

def compute_feature_stability_loss(model: GridNet, coords, mask_valid=None):
    if mask_valid is None:
        mask_valid = torch.ones((coords.shape[0], 1)).to(coords)
    loss_dict = {}
    pred_stab = model.query_stability(coords)
    assert pred_stab.shape[0] == mask_valid.shape[0]
    stab_constraint = torch.where(
        mask_valid == 1,
        pred_stab - torch.ones_like(pred_stab),
        torch.zeros_like(pred_stab)
    )
    loss_dict['stability'] = torch.mean(stab_constraint**2) 
    for level in range(model.num_levels):
        loss_dict[f'stability_reg_level{level}'] = 1e-2 * torch.mean(model.feature_stability[level].feature**2)
    return loss_dict

class PosedSdfLoss3D(BaseLoss):
    def __init__(self, sdf_weight=3e3, sign_weight=1e2, eik_weight=0, smooth_weight=0, trunc_dist=0.15, smooth_std=0.1, grad_method='autograd', finite_diff_eps=1e-2, loss_type='L2'):
        super().__init__()
        self.sdf_weight = sdf_weight
        self.sign_weight = sign_weight
        self.eik_weight = eik_weight
        self.smooth_weight = smooth_weight
        self.trunc_dist = trunc_dist
        self.smooth_std = smooth_std
        self.grad_method = grad_method
        self.finite_diff_eps = finite_diff_eps
        self.loss_type = loss_type
        self.beta = 5.0  # same as iSDF eq 6

    def compute(self, model, model_input, gt):
        coords_frame = model_input['coords_frame'][0]
        sample_frame_ids = model_input['sample_frame_ids'][0, :, 0]
        gt_sdf = gt['sdf'][0]
        gt_sdf_valid = gt['sdf_valid'][0]
        gt_sdf_sign = gt['sdf_signs'][0]
        assert coords_frame.ndim == 2 and gt_sdf.ndim == 2
        # Transform coords from keyframe to world frame
        coords_world = coords_frame.clone()
        for kf_id in range(model.num_poses):
            idxs_select = torch.nonzero(sample_frame_ids == kf_id, as_tuple=False).squeeze(1)
            if idxs_select.numel() == 0: continue
            R_world_frame, t_world_frame = model.updated_kf_pose_in_world(kf_id)
            coords_world[idxs_select, :] = utils_geometry.transform_points_to(
                coords_frame[idxs_select, :],
                R_world_frame,
                t_world_frame
            )
        pred_sdf = model(coords_world)

        # SDF prediction loss
        sdf_constraint = torch.where(
            gt_sdf_valid == 1,
            pred_sdf - gt_sdf,
            torch.zeros_like(pred_sdf)
        )
        if self.loss_type == 'L2':
            sdf_loss = torch.mean(sdf_constraint**2)
        elif self.loss_type == 'L1':
            sdf_loss = torch.mean(torch.abs(sdf_constraint))
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        loss_dict = {'sdf': sdf_loss * self.sdf_weight}
        # Free space loss: implement eq (6) in iSDF paper
        # For now, using the "ray" bounds approach
        if self.sign_weight > 0:
            fs_upper_constraint = torch.where(
                gt_sdf_sign == 1,
                F.relu(pred_sdf - gt_sdf),
                torch.zeros_like(pred_sdf)
            )  # linear cost if exceeding bound
            # fs_lower_constraint = F.relu(torch.where(
            #     gt_sdf_sign == 1,
            #     torch.exp(-self.beta * pred_sdf) - 1.0, 
            #     torch.zeros_like(pred_sdf)
            # ))  # Option 1: exponential cost if predicting negative, same as iSDF
            fs_lower_constraint = torch.where(
                gt_sdf_sign == 1,
                F.relu(self.trunc_dist - pred_sdf),
                torch.zeros_like(pred_sdf)
            )  # Option 2: linear loss if predicting smaller than truncation distance
            fs_constraint = torch.maximum(
                fs_upper_constraint,
                fs_lower_constraint
            )
            loss_dict['free_space'] = torch.mean(fs_constraint) * self.sign_weight
        # Eikonal loss:
        if self.eik_weight > 0:
            N = gt_sdf.shape[0] 
            bound = model.bound.detach().cpu().numpy()
            xs = np.reshape(np.random.uniform(bound[0,0], bound[0,1], N), (N,1))
            ys = np.reshape(np.random.uniform(bound[1,0], bound[1,1], N), (N,1))
            zs = np.reshape(np.random.uniform(bound[2,0], bound[2,1], N), (N,1))
            x_np = np.concatenate([xs, ys, zs], axis=1)
            x = torch.from_numpy(x_np).to(gt_sdf)
            x.requires_grad_(True)
            # x = coords
            gradient = gradient3d(x, model, method=self.grad_method, finite_diff_eps=self.finite_diff_eps, create_graph=True)
            grad_constraint = gradient.norm(dim=-1) - 1
            loss_dict['eik'] = torch.mean(grad_constraint**2) * self.eik_weight
        # Smoothness loss from GO-SURF, eq (10)
        if self.smooth_weight > 0:
            # N = gt_sdf.shape[0] // 10 
            # permuted_indices = np.random.permutation(coords_world.shape[0])
            # selected_indices = permuted_indices[:N]
            coords_1 = coords_world
            coords_2 = coords_1 + torch.normal(0, self.smooth_std, size=coords_1.shape).to(coords_1)
            coords_1.requires_grad_(True)
            coords_2.requires_grad_(True)
            grad1 = gradient3d(coords_1, model, method=self.grad_method, finite_diff_eps=self.finite_diff_eps)
            grad2 = gradient3d(coords_2, model, method=self.grad_method, finite_diff_eps=self.finite_diff_eps)
            # Only impose smoothness on near-surface points, which can be identified by those with valid GT sdf labels
            smooth_constraint = torch.where(
                gt_sdf_valid == 1,
                grad1 - grad2,
                torch.zeros_like(grad1)
            ) 
            loss_dict['smooth'] = torch.mean(smooth_constraint**2) * self.smooth_weight
        # Stability loss
        if isinstance(model, GridNet):
            mu_loss_dict = compute_feature_stability_loss(model, coords_world, gt_sdf_valid)
            loss_dict.update(mu_loss_dict)
        # Regularization
        # if isinstance(model, GridNet):
        #     reg_loss_dict = compute_feature_regularization_loss(model, weight=100.0)
        #     loss_dict.update(reg_loss_dict)
        return loss_dict


class PosedSdfLoss3DSubmap(BaseLoss):
    def __init__(self, cfg: dict):
        super().__init__()
        self.sdf_weight=cfg['loss']['sdf_weight']
        self.sign_weight=cfg['loss']['sign_weight']
        self.eik_weight=cfg['loss']['eik_weight']
        self.smooth_weight=cfg['loss']['smooth_weight']
        self.smooth_std=cfg['loss']['smooth_std']
        self.trunc_dist=cfg['loss']['trunc_dist']
        self.grad_method=cfg['loss']['grad_method']
        self.finite_diff_eps=cfg['loss']['finite_diff_eps']
        self.loss_type = cfg['loss']['type']
        self.pose_reg_weight = cfg['loss']['pose_reg_weight']
        self.pose_thresh_rad = cfg['loss']['pose_thresh_rad']
        self.pose_thresh_m = cfg['loss']['pose_thresh_m']
        logger.info(f"Constructed PosedSdfLoss3DSubmap ({self.loss_type}).")
        self.mode = 'submap'

    def train_submap(self):
        self.mode = 'submap'
    
    def train_joint(self):
        self.mode = 'world'
    
    def compute(self, model: GridAtlas, model_input: dict, gt: dict) -> dict:
        if self.mode == 'world':
            return self.compute_world(model, model_input, gt)
        else:
            return self.compute_submap(model, model_input, gt)

    def compute_world(self, model:GridAtlas, model_input: dict, gt: dict) -> dict:
        sample_frame_ids = model_input['sample_frame_ids'][0, :, 0]
        gt_sdf = gt['sdf'][0]
        gt_sdf_valid = gt['sdf_valid'][0]
        gt_sdf_sign = gt['sdf_signs'][0]
        coords_frame = model_input['coords_frame'][0]
        assert coords_frame.ndim == 2 and gt_sdf.ndim == 2
        
        coords_world = coords_frame.clone()
        for kf_id in range(model.num_keyframes):
            idxs_select = torch.nonzero(sample_frame_ids == kf_id, as_tuple=False).squeeze(1)
            if idxs_select.numel() == 0: continue
            R_world_frame, t_world_frame = model.updated_kf_pose_in_world(kf_id)
            coords_world[idxs_select, :] = utils_geometry.transform_points_to(
                coords_frame[idxs_select, :],
                R_world_frame,
                t_world_frame
            )
        pred_sdf = model(coords_world)
        # SDF prediction loss
        sdf_constraint = torch.where(
            gt_sdf_valid == 1,
            pred_sdf - gt_sdf,
            torch.zeros_like(pred_sdf)
        )
        if self.loss_type == 'L2':
            sdf_loss = torch.mean(sdf_constraint**2)
        elif self.loss_type == 'L1':
            sdf_loss = torch.mean(torch.abs(sdf_constraint))
        else: 
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        loss_dict = {'sdf': sdf_loss * self.sdf_weight}
        # Sign prediction loss (free space)
        if self.sign_weight > 0:
            fs_upper_constraint = torch.where(
                gt_sdf_sign == 1,
                F.relu(pred_sdf - gt_sdf),
                torch.zeros_like(pred_sdf)
            )  # linear cost if exceeding bound
            fs_lower_constraint = torch.where(
                gt_sdf_sign == 1,
                F.relu(self.trunc_dist - pred_sdf),
                torch.zeros_like(pred_sdf)
            )  # Option 2: linear loss if predicting smaller than truncation distance
            fs_constraint = torch.maximum(
                fs_upper_constraint,
                fs_lower_constraint
            )
            loss_dict['free_space'] = torch.mean(fs_constraint) * self.sign_weight
        # Eikonal loss:
        if self.eik_weight > 0:
            raise ValueError("Eikonal loss not supported for submap training.")
        # Smoothness loss from GO-SURF, eq (10)
        if self.smooth_weight > 0:
            coords_1 = coords_world.clone()
            coords_2 = coords_1 + torch.normal(0, self.smooth_std, size=coords_1.shape).to(coords_1)
            coords_1.requires_grad_(True)
            coords_2.requires_grad_(True)
            grad1 = gradient3d(coords_1, model, method=self.grad_method, finite_diff_eps=self.finite_diff_eps)
            grad2 = gradient3d(coords_2, model, method=self.grad_method, finite_diff_eps=self.finite_diff_eps)
            # Only impose smoothness on near-surface points, which can be identified by those with valid GT sdf labels
            smooth_constraint = torch.where(
                gt_sdf_valid == 1,
                grad1 - grad2,
                torch.zeros_like(grad1)
            ) 
            loss_dict['smooth'] = torch.mean(smooth_constraint**2) * self.smooth_weight
        if self.pose_reg_weight > 0:
            for submap_id in range(model.num_submaps):
                submap = model.get_submap(submap_id)
                pose_loss_dict = compute_pose_regularization_loss(
                    submap, 
                    weight=self.pose_reg_weight
                )
                for key, value in pose_loss_dict.items():
                    loss_dict[f'submap{submap_id}_KF_{key}'] = value
                rot_norm = torch.sum(model.rotation_corrections[submap_id]**2)
                loss_dict[f'submap{submap_id}_l2_reg_R'] = self.pose_reg_weight * rot_norm
                tran_norm = torch.sum(model.translation_corrections[submap_id]**2)
                loss_dict[f'submap{submap_id}_l2_reg_t'] = self.pose_reg_weight * tran_norm
                
        # Stability loss
        # mu_loss_dict = compute_feature_stability_loss(submap, coords_submap, gt_sdf_valid)
        # loss_dict.update(mu_loss_dict)
        return loss_dict

    def compute_submap(self, model:GridAtlas, model_input: dict, gt: dict) -> dict:
        sample_submap_ids = model_input['sample_submap_ids'][0, :, 0]
        loss_dict = {}
        
        # Go through reconstruction for each submap
        for submap_id in range(model.num_submaps):
            submap = model.get_submap(submap_id)
            idxs_in_submap = torch.nonzero(sample_submap_ids == submap_id, as_tuple=False).squeeze(1)
            if idxs_in_submap.numel() == 0: 
                # logger.warning(f"Submap {submap_id} has no samples.")
                continue
            # Downselect input to only those in the submap
            coords_frame = model_input['coords_frame'][0, idxs_in_submap, :]
            sample_frame_ids = model_input['sample_frame_ids'][0, idxs_in_submap, 0]
            gt_sdf = gt['sdf'][0, idxs_in_submap, :]
            gt_sdf_valid = gt['sdf_valid'][0, idxs_in_submap, :]
            gt_sdf_sign = gt['sdf_signs'][0, idxs_in_submap, :]
            assert coords_frame.ndim == 2 and gt_sdf.ndim == 2
            # Compute coords in submap
            coords_submap = coords_frame.clone()
            for kf_id in range(model.num_keyframes):
                if model.submap_id_for_kf(kf_id) != submap_id: continue
                idxs_select = torch.nonzero(sample_frame_ids == kf_id, as_tuple=False).squeeze(1)
                if idxs_select.numel() == 0: continue
                R_submap_frame, t_submap_frame = model.updated_kf_pose_in_submap(kf_id, submap_id)
                coords_submap[idxs_select, :] = utils_geometry.transform_points_to(
                    coords_frame[idxs_select, :],
                    R_submap_frame,
                    t_submap_frame
                )
            # Compute losses in submap
            pred_sdf = submap(coords_submap)
            # SDF prediction loss
            sdf_constraint = torch.where(
                gt_sdf_valid == 1,
                pred_sdf - gt_sdf,
                torch.zeros_like(pred_sdf)
            )
            if self.loss_type == 'L2':
                sdf_loss = torch.mean(sdf_constraint**2)
            elif self.loss_type == 'L1':
                sdf_loss = torch.mean(torch.abs(sdf_constraint))
            else: 
                raise ValueError(f"Invalid loss type: {self.loss_type}")
            submap_loss_dict = {f'{submap_id}_sdf': sdf_loss * self.sdf_weight}
            # Sign prediction loss (free space)
            if self.sign_weight > 0:
                fs_upper_constraint = torch.where(
                    gt_sdf_sign == 1,
                    F.relu(pred_sdf - gt_sdf),
                    torch.zeros_like(pred_sdf)
                )  # linear cost if exceeding bound
                fs_lower_constraint = torch.where(
                    gt_sdf_sign == 1,
                    F.relu(self.trunc_dist - pred_sdf),
                    torch.zeros_like(pred_sdf)
                )  # Option 2: linear loss if predicting smaller than truncation distance
                fs_constraint = torch.maximum(
                    fs_upper_constraint,
                    fs_lower_constraint
                )
                submap_loss_dict[f'{submap_id}_free_space'] = torch.mean(fs_constraint) * self.sign_weight
            # Eikonal loss:
            if self.eik_weight > 0:
                raise ValueError("Eikonal loss not supported for submap training.")
            # Smoothness loss from GO-SURF, eq (10)
            if self.smooth_weight > 0:
                coords_1 = coords_submap.clone()
                coords_2 = coords_1 + torch.normal(0, self.smooth_std, size=coords_1.shape).to(coords_1)
                coords_1.requires_grad_(True)
                coords_2.requires_grad_(True)
                grad1 = gradient3d(coords_1, submap, method=self.grad_method, finite_diff_eps=self.finite_diff_eps)
                grad2 = gradient3d(coords_2, submap, method=self.grad_method, finite_diff_eps=self.finite_diff_eps)
                # Only impose smoothness on near-surface points, which can be identified by those with valid GT sdf labels
                smooth_constraint = torch.where(
                    gt_sdf_valid == 1,
                    grad1 - grad2,
                    torch.zeros_like(grad1)
                ) 
                submap_loss_dict[f'{submap_id}_smooth'] = torch.mean(smooth_constraint**2) * self.smooth_weight
            # Stability loss
            # mu_loss_dict = compute_feature_stability_loss(submap, coords_submap, gt_sdf_valid)
            # loss_dict.update(mu_loss_dict)
            # Pose regularization
            if self.pose_reg_weight > 0:
                # pose_loss_dict = compute_pose_trust_region_loss(
                #     submap, 
                #     thresh_rad=self.pose_thresh_rad,
                #     thresh_m=self.pose_thresh_m,
                #     weight=self.pose_reg_weight
                # )  # Trust region based regularization
                pose_loss_dict = compute_pose_regularization_loss(
                    submap, 
                    weight=self.pose_reg_weight
                )  # L2 pose regularization
                for key, value in pose_loss_dict.items():
                    submap_loss_dict[f'{submap_id}_{key}'] = value
            loss_dict.update(submap_loss_dict)

        return loss_dict
    

class MisoLossTracking(BaseLoss):
    """An implementation of the SDF-based tracking loss.
    """
    def __init__(
            self, 
            weight_sdf=1.0, 
            loss_type='L2', 
            trunc_dist=None, 
            gm_scale_sdf=1.0, 
            gm_scale_grad=None
        ):
        super().__init__()
        self.weight_sdf = weight_sdf
        self.loss_type = loss_type
        self.trunc_dist = trunc_dist
        self.gm_scale_sdf = gm_scale_sdf
        self.gm_scale_grad = gm_scale_grad  # not used yet 

    def compute(self, model: GridNet, model_input: dict, gt: dict) -> dict:
        coords_frame = model_input['coords_frame'][0]
        sample_frame_ids = model_input['sample_frame_ids'][0, :, 0]
        gt_sdf = gt['sdf'][0]
        assert coords_frame.ndim == 2 and gt_sdf.ndim == 2
        valid_mask = gt['sdf_valid'][0]
        # Truncate based on truncation distance
        if self.trunc_dist is not None:
            mask_trunc = torch.abs(gt_sdf) < self.trunc_dist
            valid_mask = torch.logical_and(valid_mask, mask_trunc)
            # logger.info(f"Pruning with trunc_dist {self.trunc_dist}: {torch.count_nonzero(valid_mask)}/{gt_sdf.shape[0]}.")
        assert valid_mask.shape == gt_sdf.shape
        # Transform coords from keyframe to world frame
        unique_frame_ids = np.unique(sample_frame_ids.detach().cpu().numpy()).tolist()
        coords_world = coords_frame.clone()
        for kf_id in unique_frame_ids:
            idxs_select = torch.nonzero(sample_frame_ids == kf_id, as_tuple=False).squeeze(1)
            if idxs_select.numel() == 0: continue
            R_world_frame, t_world_frame = model.updated_kf_pose_in_world(kf_id)
            coords_world[idxs_select, :] = utils_geometry.transform_points_to(
                coords_frame[idxs_select, :],
                R_world_frame,
                t_world_frame
            )
        pred_sdf = model(coords_world)
        # SDF prediction loss
        sdf_constraint = torch.where(
            valid_mask == 1,
            pred_sdf - gt_sdf,
            torch.zeros_like(pred_sdf)
        )
        if self.loss_type == 'L2':
            sdf_loss = torch.mean(sdf_constraint**2)
        elif self.loss_type == 'L1':
            sdf_loss = torch.mean(torch.abs(sdf_constraint))
        elif self.loss_type == 'GM':
            e = sdf_constraint.clone().detach()
            w = self.gm_scale_sdf / (self.gm_scale_sdf + e**2)**2
            sdf_loss = torch.mean(w * sdf_constraint**2)
            # Print historgram over e_abs
            # e_abs = torch.abs(e)
            # e_abs_np = e_abs.cpu().numpy()
            # hist, bin_edges = np.histogram(e_abs_np, bins=10)
            # bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            # bin_centers = [f"{x:.1e}" for x in bin_centers]
            # hist = 100 * hist / np.sum(hist)
            # hist = [f"{x:.1f}" for x in hist]
            # print(f"BIN: {bin_centers}\nPER: {hist}.")
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        loss_dict = {f'sdf_{self.loss_type}': self.weight_sdf * sdf_loss}
        return loss_dict
    

def miso_scale_sigma_helper(self, sdf, sigmoid_scale):
    input = - sdf / sigmoid_scale
    return torch.sigmoid(input)


def miso_loss_regression(
        pred, 
        targ, 
        valid_mask=None, 
        sample_weights=None,
        loss_type='L1'
    ):
    """Helper function to compute the regression loss.

    Args:
        pred (_type_): Predicted value (N, d)
        targ (_type_): Target value (N, d)
        valid_mask (_type_): boolean mask (N, 1)
        sample_weights (_type_): sample weights (N, 1)
        loss_type (_type_): L2, L1, Cosine
    """
    assert pred.shape == targ.shape
    num_samples = pred.shape[0]
    if valid_mask is None:
        valid_mask = torch.ones((num_samples, 1)).to(pred)
    if sample_weights is None:
        sample_weights = torch.ones((num_samples, 1)).to(pred)
    assert valid_mask.shape == (num_samples, 1)
    assert sample_weights.shape == (num_samples, 1)
    if loss_type == 'L2':
        # Compute L2 (MSE) loss
        loss_vec = torch.sum((pred - targ)**2, dim=1, keepdim=True)  # (N, 1)
    elif loss_type == 'L1':
        # Compute L1 (MAE) loss
        loss_vec = torch.sum(torch.abs(pred - targ), dim=1, keepdim=True)  # (N, 1)
    elif loss_type == 'Cosine':
        # Compute cosine similarity
        loss_vec = 1.0 - F.cosine_similarity(pred, targ, dim=1, eps=1e-8).unsqueeze(1)  # (N, 1)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    loss_vec = torch.where(
        valid_mask == 1,
        loss_vec,
        torch.zeros_like(loss_vec)
    )
    reg_loss = torch.mean(sample_weights * loss_vec)
    return reg_loss


def miso_loss_eikonal(
        model: BaseNet,
        coords_world, 
        gt_sdf,
        eik_trunc_dist,
        grad_method,
        finite_diff_eps,
    ):
    """Helper function to compute the Eikonal loss.

    Args:
        model (BaseNet): model to evaluate the Eikonal equation
        coords_world: sampled coordinates to evaluate the Eikonal equation
        gt_sdf: ground truth SDF values at sampled points
        eik_trunc_dist: optional filter to only apply Eikonal loss to points within a certain distance 
        grad_method: 
        finite_diff_eps: 
    """
    if eik_trunc_dist is not None:
        valid_mask = torch.abs(gt_sdf) < eik_trunc_dist
        valid_indices = torch.nonzero(valid_mask, as_tuple=False)[:, 0]
        x_eik = coords_world[valid_indices,:].clone()
    else:
        x_eik = coords_world.clone()
    x_eik.requires_grad_(True)
    gradient = gradient3d(x_eik, model, method=grad_method, finite_diff_eps=finite_diff_eps, create_graph=True)
    grad_constraint = gradient.norm(dim=-1) - 1
    return torch.mean(grad_constraint**2)


def miso_loss_free_space(
        pred_sdf, 
        gt_sdf, 
        gt_sdf_sign, 
        trunc_dist
    ):
    """Helper function to compute the free space loss.

    Args:
        pred_sdf: predicted SDF values
        gt_sdf: ground truth SDF values (storing upper bound in this case)
        gt_sdf_sign: ground truth SDF signs
        trunc_dist: truncation distance (i.e. lower bound for SDF)

    Returns:
        _type_: _description_
    """
    assert trunc_dist is not None
    fs_upper_constraint = torch.where(
        gt_sdf_sign == 1,
        F.relu(pred_sdf - gt_sdf),
        torch.zeros_like(pred_sdf)
    )  # linear cost if exceeding bound
    fs_lower_constraint = torch.where(
        gt_sdf_sign == 1,
        F.relu(trunc_dist - pred_sdf),
        torch.zeros_like(pred_sdf)
    )  # Option 2: linear loss if predicting smaller than truncation distance
    fs_constraint = torch.maximum(
        fs_upper_constraint,
        fs_lower_constraint
    )
    return torch.mean(fs_constraint) 


class MisoLossMappingBase(BaseLoss):
    """An implementation of the SDF-based mapping loss.
    """
    def __init__(
            self, 
            loss_type='L1', 
            weight_sdf=1.0,
            weight_eik=0.5,
            weight_fs=0,
            trunc_dist=0, 
            finite_diff_eps=1e-2, 
            grad_method='autograd',
            eik_trunc_dist=0.1,
            use_stability=False, 
            weight_clip=0
        ):
        super().__init__()
        self.loss_type = loss_type
        self.trunc_dist = trunc_dist
        self.weight_sdf = weight_sdf
        self.weight_eik = weight_eik
        self.weight_fs = weight_fs
        self.finite_diff_eps = finite_diff_eps
        self.grad_method = grad_method
        self.eik_trunc_dist = eik_trunc_dist
        self.use_stability = use_stability
        self.weight_clip = weight_clip
        logger.info("MisoLossMappingBase initialized with the following configuration:")
        logger.info(f"  - Loss function: {self.loss_type}")
        logger.info(f"  - Weight SDF: {self.weight_sdf}")
        logger.info(f"  - Weight Eik: {self.weight_eik}")
        logger.info(f"  - Weight FS: {self.weight_fs}")
        logger.info(f"  - Truncation distance: {self.trunc_dist}")
        logger.info(f"  - Finite difference epsilon: {self.finite_diff_eps}")
        logger.info(f"  - Gradient method: {self.grad_method}")
        logger.info(f"  - Eik truncation distance: {self.eik_trunc_dist}")
        logger.info(f"  - Use stability: {self.use_stability}")
        logger.info(f"  - Weight CLIP: {self.weight_clip}")

    def query_kf_pose(self, model: BaseNet, kf_id: int):
        raise NotImplementedError("This function should be implemented in the derived class.")
    
    def query_model(self, model, coords_world: torch.Tensor):
        model_out = model(coords_world)
        out_dict = {
            'sdf': model_out[:, [0]]
        }
        if self.weight_clip > 0:
            out_dict['clip'] = model_out[:, 1:]
        return out_dict
        
    def compute(self, model, model_input: dict, gt: dict) -> dict:
        coords_frame = model_input['coords_frame'][0]
        sample_frame_ids = model_input['sample_frame_ids'][0, :, 0]
        sample_weights = model_input['weights'][0]
        gt_sdf = gt['sdf'][0]
        gt_sdf_valid = gt['sdf_valid'][0]
        gt_sdf_sign = gt['sdf_signs'][0]
        assert coords_frame.ndim == 2 and gt_sdf.ndim == 2
        assert sample_weights.shape == gt_sdf.shape
        # Transform coords from keyframe to world frame
        unique_frame_ids = np.unique(sample_frame_ids.detach().cpu().numpy()).tolist()
        coords_world = coords_frame.clone()
        for kf_id in unique_frame_ids:
            idxs_select = torch.nonzero(sample_frame_ids == kf_id, as_tuple=False).squeeze(1)
            if idxs_select.numel() == 0: continue
            R_world_frame, t_world_frame = self.query_kf_pose(model, kf_id)
            coords_world[idxs_select, :] = utils_geometry.transform_points_to(
                coords_frame[idxs_select, :],
                R_world_frame,
                t_world_frame
            )
        pred_dict = self.query_model(model, coords_world)
        pred_sdf  = pred_dict['sdf']
        # Compute main SDF-based loss
        sdf_loss = miso_loss_regression(
            pred=pred_sdf,
            targ=gt_sdf,
            valid_mask=gt_sdf_valid,
            sample_weights=sample_weights,
            loss_type=self.loss_type
        )
        loss_dict = {f'sdf_{self.loss_type}': self.weight_sdf * sdf_loss}
        # Compute Eikonal loss
        if self.weight_eik > 0:
            assert not self.use_clip, "Eikonal loss not supported with CLIP."
            eik_loss = miso_loss_eikonal(
                model=model,
                coords_world=coords_world,
                gt_sdf=gt_sdf,
                eik_trunc_dist=self.eik_trunc_dist,
                grad_method=self.grad_method,
                finite_diff_eps=self.finite_diff_eps
            )
            loss_dict['eik'] = eik_loss * self.weight_eik
        # Free space bound loss
        if self.weight_fs > 0:
            fs_loss = miso_loss_free_space(
                pred_sdf=pred_sdf,
                gt_sdf=gt_sdf,
                gt_sdf_sign=gt_sdf_sign,
                trunc_dist=self.trunc_dist
            )
            loss_dict['free_space'] = fs_loss * self.weight_fs
        if self.use_stability:
            mu_loss_dict = compute_feature_stability_loss(model, coords_world)
            loss_dict.update(mu_loss_dict)
        if self.weight_clip > 0:
            clip_loss_dict = self.compute_clip(model, model_input, gt)
            loss_dict.update(clip_loss_dict)
        return loss_dict

    def compute_clip(self, model, model_input: dict, gt: dict) -> dict:
        coords_frame = model_input['clip_coords_frame'][0]
        sample_frame_ids = model_input['clip_sample_frame_ids'][0, :, 0]
        gt_clip = gt['clip_embeddings'][0]
        assert coords_frame.ndim == 2 and gt_clip.ndim == 2
        # Transform coords from keyframe to world frame
        unique_frame_ids = np.unique(sample_frame_ids.detach().cpu().numpy()).tolist()
        coords_world = coords_frame.clone()
        for kf_id in unique_frame_ids:
            idxs_select = torch.nonzero(sample_frame_ids == kf_id, as_tuple=False).squeeze(1)
            if idxs_select.numel() == 0: continue
            R_world_frame, t_world_frame = self.query_kf_pose(model, kf_id)
            coords_world[idxs_select, :] = utils_geometry.transform_points_to(
                coords_frame[idxs_select, :],
                R_world_frame,
                t_world_frame
            )
        pred_dict = self.query_model(model, coords_world)
        pred_clip  = pred_dict['clip']
        # Compute CLIP prediction loss
        clip_loss = miso_loss_regression(
            pred=pred_clip,
            targ=gt_clip,
            valid_mask=None,
            sample_weights=None,
            loss_type='L1'
        )
        return {
            'clip_L1': clip_loss * self.weight_clip
        }

    

class MisoLossMapping(MisoLossMappingBase):
    """For mapping within a single submap (GridNet).
    """
    def query_kf_pose(self, model, kf_id):
        assert isinstance(model, GridNet)
        return model.updated_kf_pose_from_key(f'KF{kf_id}')
    

class MisoLossFusion(MisoLossMappingBase):
    """For joint mapping over multiple submaps (GridAtlas).
    """
    def query_kf_pose(self, model, kf_id):
        assert isinstance(model, GridAtlas)
        return model.updated_kf_pose_in_world(kf_id)