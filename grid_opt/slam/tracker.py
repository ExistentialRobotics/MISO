import math
import torch
from copy import deepcopy
from typing import List
from pytorch3d.transforms.so3 import hat
from grid_opt.configs import *
from grid_opt.models.grid_net import GridNet
from grid_opt.datasets.submap_dataset import SubmapDataset
from grid_opt.loss import MisoLossTracking
import grid_opt.utils.utils as utils
import logging
logger = logging.getLogger(__name__)


class Tracker:
    """
    A class for tracking keyframes and submaps in a SLAM system.
    """
    def __init__(
            self, 
            model: GridNet,              # We always perform tracking in a single GridNet
            dataset: SubmapDataset,         
            cfg: dict,
        ):
        assert isinstance(model, GridNet), "Model must be an instance of GridNet."
        self.grid = model
        self.dataset = dataset
        self.train_loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)
        self.cfg = cfg
        cfg_track = cfg['tracking']
        self.lr = cfg_track['learning_rate']
        self.verbose = cfg_track['verbose']
        self.gm_scale_sdf = cfg_track['gm_scale_sdf']
        self.lm_lambda = cfg_track['lm_lambda']
        self.lm_max_iter = cfg_track['lm_max_iter']
        self.lm_tol_deg = cfg_track['lm_tol_deg']
        self.lm_tol_m = cfg_track['lm_tol_m']
        self.loss_type = cfg_track['loss_type']
        self.trunc_dist = cfg_track['trunc_dist']
        self.solver = cfg_track['solver']
        if self.solver == 'adam':
            self.loss_fn = MisoLossTracking(
                weight_sdf=1.0,
                loss_type=self.loss_type,
                trunc_dist=self.trunc_dist,
                gm_scale_sdf=self.gm_scale_sdf,
            )
        if 'disable' in cfg_track and cfg_track['disable']:
            self.disable = True
        else:
            self.disable = False
        logger.info("Tracker initialized with the following configuration:")
        logger.info(f"  - Verbose: {self.verbose}")
        logger.info(f"  - Solver: {self.solver}")
        logger.info(f"  - Loss type: {self.loss_type}")
        logger.info(f"  - Geman-McClure scale: {self.gm_scale_sdf}")
        logger.info(f"  - LM lambda: {self.lm_lambda}")
        logger.info(f"  - LM max iterations: {self.lm_max_iter}")
        logger.info(f"  - LM tolerance (degrees): {self.lm_tol_deg}")
        logger.info(f"  - LM tolerance (meters): {self.lm_tol_m}")
        logger.info(f"  - Truncation distance: {self.trunc_dist}")
        logger.info(f"  - Disable optimization: {self.disable}")
        self.latest_fov_overlap = 1.0
    
    def initialize_window(self, head_kf, tail_kf):
        """Initialize the keyframe poses in the grid in the window [head_kf, tail_kf),
        by propagating the odometry estimates contained in the dataset.
        """
        with torch.no_grad():
            for dst_id in range(head_kf, tail_kf):
                src_id = dst_id - 1
                assert src_id >= 0
                R_src, t_src = self.grid.updated_kf_pose_in_world(src_id)
                T_world_src = utils_geometry.pose_matrix(R_src, t_src)
                T_src_dst = self.dataset.get_odometry_at_pose(src_id).to(T_world_src)
                T_world_dst = T_world_src @ T_src_dst
                R_world_dst = T_world_dst[:3, :3]
                t_world_dst = T_world_dst[:3, [3]]
                self.grid.set_initial_kf_pose(dst_id, R_world_dst, t_world_dst)
    
    def track_window(self, optimize_kfs:List[int], iterations=10):
        """Track multiple keyframes in a window at once using standard pytorch optimizer.
        """
        print(f"\nTracking frames: {list(optimize_kfs)}.")
        # Only make the specified keyframes trainable
        self.grid.lock_feature()
        self.grid.unlock_pose()
        self.grid.lock_all_pose_indices()
        for kf_id in optimize_kfs:
            self.grid.unlock_pose_index(kf_id)
        # Only sample from the specified keyframes
        self.dataset.select_keyframes(optimize_kfs)
        # Config and optimize!
        cfg_copy = deepcopy(self.cfg)    
        cfg_train = cfg_copy['train']
        cfg_train['epochs'] = iterations
        cfg_train['learning_rate'] = self.lr
        cfg_train['verbose'] = self.verbose
        trainer = Trainer(
            cfg_train,
            self.grid,
            self.loss_fn,
            self.train_loader,
            None,
            self.cfg['device'],
            torch.float32
        )
        if self.verbose: self.grid.print_trainable_params()
        trainer.train()
        if self.verbose: self.grid.print_kf_pose_info()
    
    def track(self, optimize_kf: int):
        """Track a single keyframe using Levenberg-Marquardt optimization with a Geman-McClure loss.
        """
        if self.disable: 
            logger.info(f"Tracking disabled. Skipping tracking for keyframe {optimize_kf}.")
            return
        if self.solver == 'adam':
            self.track_window([optimize_kf], iterations=15)
        elif self.solver == 'lm':
            self.track_lm(optimize_kf)
        else:
            raise ValueError(f"Unknown solver: {self.solver}.")

    def track_lm(self, optimize_kf: int):
        for lm_step in range(self.lm_max_iter):
            lm_step_info = self.lm_step(optimize_kf)
            delta_deg = lm_step_info['delta_R_deg']
            delta_m = lm_step_info['delta_t_norm']
            gradnorm = lm_step_info['grad_norm']
            if self.verbose:
                print(f"LM step {lm_step}: delta_deg: {delta_deg:.1e}, delta_t_norm: {delta_m:.1e}, grad_norm: {gradnorm:.1e}")
            if delta_deg < self.lm_tol_deg and delta_m < self.lm_tol_m:
                break
        self.latest_fov_overlap = lm_step_info['fov_overlap']
        logger.info(f"LM tracking: frame {optimize_kf}, fov_overlap: {lm_step_info['fov_overlap']:.2f}, lm_steps={lm_step}.")
        if self.verbose: self.grid.print_kf_pose_info()
    
    def residual_weights(self, r: torch.Tensor):
        if self.loss_type == 'L2':
            w = torch.ones_like(r)
        elif self.loss_type == 'GM':
            w = self.gm_scale_sdf / (self.gm_scale_sdf + r**2)**2  # N, 1
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}.")
        return w
    
    def lm_step(self, optimize_kf:int):
        """
        Perform a single LM-based registration step to optimize the keyframe pose.
        """
        self.dataset.select_keyframes([optimize_kf])
        model_input, gt = utils.get_batch(self.train_loader, self.cfg['device'])
        coords_frame = model_input['coords_frame'][0]
        frame_ids = model_input['sample_frame_ids'][0]
        gt_sdf = gt['sdf'][0]
        gt_sdf_valid = gt['sdf_valid'][0]
        # Optional SDF-based truncation
        if self.trunc_dist is not None:
            num_bef_filter = gt_sdf.shape[0]
            valid_idxs = torch.nonzero(
                torch.abs(gt_sdf[:,0]) < self.trunc_dist, as_tuple=False).squeeze(1)
            coords_frame = coords_frame[valid_idxs, :]
            frame_ids = frame_ids[valid_idxs, :]
            gt_sdf = gt_sdf[valid_idxs,: ]
            gt_sdf_valid = gt_sdf_valid[valid_idxs, :]
            num_aft_filter = gt_sdf.shape[0]
        # logger.info(f"LM step for frame {optimize_kf} with {coords_frame.shape[0]} points.")
        assert torch.all(frame_ids == optimize_kf)
        assert torch.all(gt_sdf_valid == 1), "Only valid SDFs should be used for tracking."
        # Compute SDF gradient
        Rwf, twf = self.grid.updated_kf_pose_from_key(f'KF{optimize_kf}')
        Rwf, twf = Rwf.detach(), twf.detach()
        coords_world = utils_geometry.transform_points_to(coords_frame, Rwf, twf)
        mask_bnd = utils_geometry.coords_in_bound(coords_world, self.grid.bound)
        fov_overlap = float(torch.count_nonzero(mask_bnd)) / mask_bnd.numel()
        x = coords_world.clone()
        x.requires_grad = True
        grad_world = gradient3d(x, self.grid, method='autograd', create_graph=False).detach()  # N, 3
        # Compute the Jacobian
        Rxi = utils_geometry.transform_points_to(coords_frame, Rwf, torch.zeros_like(twf))
        Rxi_hat = hat(Rxi)  # N, 3, 3
        cT = torch.bmm(Rxi_hat, grad_world.unsqueeze(-1)).squeeze(-1)  # N, 3
        cTR = cT @ Rwf  # N, 3
        J = torch.cat((cTR, grad_world), dim=1)   # J = [JR, Jt] with dimension N, 6
        # Compute the residual
        sdf_pred = self.grid(coords_world)
        r = (sdf_pred - gt_sdf).detach()  # N, 1
        # Compute the weight based on Geman-McClure
        w = self.residual_weights(r)  # N, 1
        # Assemble and solve the LM normal equations
        WJ = w * J  # N, 6
        Wr = w * r  # N, 1
        H = J.T @ WJ + self.lm_lambda * torch.eye(6, device=J.device)  # 6, 6
        g = J.T @ Wr  # 6, 1
        delta = torch.linalg.solve(H, -g)
        delta_R, delta_t = delta[:3], delta[3:]  # 3, 1
        # Update the keyframe pose
        with torch.no_grad():
            kf_id = self.grid.pose_key_to_id(f'KF{optimize_kf}')
            self.grid.rotation_corrections[kf_id] += delta_R.squeeze()
            self.grid.translation_corrections[kf_id] += delta_t
        delta_R_rad, delta_t_norm = torch.linalg.norm(delta_R).item(), torch.linalg.norm(delta_t).item()
        delta_R_deg = math.degrees(delta_R_rad)
        g_norm = torch.linalg.norm(g).item()
        info = {
            'delta_R_deg': delta_R_deg,
            'delta_t_norm': delta_t_norm,
            'grad_norm': g_norm,
            'fov_overlap': fov_overlap,
        }
        return info

