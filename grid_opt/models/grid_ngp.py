import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sdf.netowrk import SDFNetwork
from pytorch3d.transforms import so3_exp_map
import grid_opt.utils.utils as utils
from .base_net import BaseNet
from .modules import MLPNet
from .grid_modules import *
import grid_opt.utils.utils_geometry as utils_geometry
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GridNGP(BaseNet):
    """
    An implementation similar to grid_net, but with the regular grid
    replaced by a hash grid implemented in torch-ngp. Right now, this
    code contains a subset of functionalities of the original grid_net.

    # This code assumes torch ngp is put on python path:
    # e.g. source PATH_TO_TORCH_NGP/put_torch_ngp_on_path.sh

    # FIXME: the impl has copied code from grid_net.
    """
    def __init__(self,
        cfg: dict, 
        device = 'cuda:0',
        dtype = torch.float32,
    ):
        super(GridNGP, self).__init__(cfg, device, dtype)        
        self.init_ngp(cfg)
        self.init_poses(cfg)
    
    def init_ngp(self, cfg):
        self.num_levels = cfg['grid']['n_levels']
        if 'base_resolution' not in cfg['grid']:
            logger.warning("base_resolution not found in cfg. Using default value of 16.")
            base_res = 16
        else:
            base_res = cfg['grid']['base_resolution']
        self.sdf_network = SDFNetwork(
            encoding='hashgrid',
            num_levels=cfg['grid']['n_levels'],
            base_resolution=base_res,
            level_dim=cfg['grid']['feature_dim'],   
            hidden_dim=cfg['decoder']['hidden_dim'],
            num_layers=cfg['decoder']['hidden_layers'],
            output_dim=cfg['decoder']['out_dim'],
        )

    def init_poses(self, cfg):
        """Initialize pose correction terms.
        The pose corrections can be optimized jointly with the feature grid, i.e., bundle adjustment.
        As an example, see PosedSdfLoss3D.
        """
        self.num_poses = cfg['pose']['num_poses']
        self.optimize_pose = cfg['pose']['optimize']
        self.rotation_corrections = torch.nn.Parameter(
            torch.zeros(self.num_poses, 3).float().to(self.device),
            requires_grad=self.optimize_pose
        )
        self.translation_corrections = torch.nn.Parameter(
            torch.zeros(self.num_poses, 3, 1).float().to(self.device),
            requires_grad=self.optimize_pose
        )
        self.pose_estimates_known = [False] * self.num_poses
        self.register_buffer('Rwk', utils_geometry.identity_rotations(self.num_poses).to(self.device))
        self.register_buffer('twk', torch.zeros(size=(self.num_poses, 3, 1), device=self.device))
        self.locked_pose_indices = set()
        self._pose_key_to_id = dict()
        logger.info(f"Initialized {self.num_poses} pose variables (optimize={self.optimize_pose}).")

    def lock_feature(self):
        for param in self.sdf_network.encoder.parameters():
            param.requires_grad = False

    def unlock_feature(self):
        for param in self.sdf_network.encoder.parameters():
            param.requires_grad = True
    
    def lock_pose(self):
        self.rotation_corrections.requires_grad_(False)
        self.translation_corrections.requires_grad_(False)
        self.lock_all_pose_indices()
     
    def unlock_pose(self):
        self.rotation_corrections.requires_grad_(True)
        self.translation_corrections.requires_grad_(True)
        self.unlock_all_pose_indices()
    
    def lock_pose_index(self, pose_index:int):
        self.locked_pose_indices.add(pose_index)

    def lock_all_pose_indices(self):
        self.locked_pose_indices = set(range(self.num_poses))

    def unlock_pose_index(self, pose_index:int):
        self.locked_pose_indices.remove(pose_index)

    def unlock_all_pose_indices(self):
        self.locked_pose_indices.clear()

    def pose_correction(self, kf_id: int):
        r = self.rotation_corrections[[kf_id], :]  # (1,3)
        t = self.translation_corrections[kf_id, :, :]  # (3,1)
        if kf_id in self.locked_pose_indices:
            r = r.clone().detach()
            t = t.clone().detach()
        return r, t

    def set_initial_kf_pose(self, kf_id: int, Rwk: torch.Tensor, twk: torch.Tensor, kf_key=None):
        """Set the initial guess for the keyframe pose.
        # TODO: the kf_id is currently a local consecutive index, but 
        this should be replaced by the key completely in the future. 
        We keep it for now to be compatible with the previous version.

        Args:
            kf_id (int): local consecutive index / ID of the keyframe
            Rwk (torch.Tensor): rotation
            twk (torch.Tensor): translation
            kf_key: An optional key associated with this pose. Defaults to None.
        """
        assert Rwk.shape == (3,3)
        assert twk.shape == (3,1)
        assert kf_id < self.num_poses, f"KF ID {kf_id} exceeds the number of poses {self.num_poses}!"
        self.pose_estimates_known[kf_id] = True
        self.Rwk[kf_id,: ,: ] = Rwk.to(self.device)
        self.twk[kf_id, :, :] = twk.to(self.device)
        # Reset perturbations to zero
        with torch.no_grad():
            self.rotation_corrections[kf_id, :].copy_(torch.zeros(3, device=self.device))
            self.translation_corrections[kf_id, :, :].copy_(torch.zeros(3, 1, device=self.device))
        if kf_key is not None:
            self._pose_key_to_id[kf_key] = kf_id
    
    def pose_key_to_id(self, kf_key):
        assert kf_key in self._pose_key_to_id, f"Key {kf_key} not found in pose key to ID mapping!"
        return self._pose_key_to_id[kf_key]
    
    def initial_kf_pose(self, kf_id: int):
        assert self.pose_estimates_known[kf_id], f"Initial pose estimate for KF {kf_id} is not available!"
        return self.Rwk[kf_id, :, :], self.twk[kf_id, :, :]
    
    def initial_kf_pose_in_world(self, kf_id: int):
        return self.initial_kf_pose(kf_id)
    
    def initial_kf_pose_from_key(self, kf_key):
        kf_id = self.pose_key_to_id(kf_key)
        return self.initial_kf_pose(kf_id)
    
    def updated_kf_pose(self, kf_id: int):
        Rwk, twk = self.initial_kf_pose_in_world(kf_id)
        Dr, Dt = self.pose_correction(kf_id)  
        return utils_geometry.apply_pose_correction(
            Rwk, twk, Dr, Dt
        )
    
    def updated_kf_pose_in_world(self, kf_id: int):  
        return self.updated_kf_pose(kf_id)
    
    def updated_kf_pose_from_key(self, kf_key):
        kf_id = self.pose_key_to_id(kf_key)
        return self.updated_kf_pose(kf_id)
    
    def query_feature(self, x):
        x = utils.normalize_coordinates(x, self.bound)
        return self.sdf_network.encoder(x)
    
    def forward(self, x):
        x = utils.normalize_coordinates(x, self.bound)
        return self.sdf_network(x)
    
    def params_at_level(self, level):
        # FIXME: right now this always return the full set of params!
        return list(self.sdf_network.parameters())
    
    def print_kf_pose_info(self):
        max_rot = torch.max(torch.linalg.norm(self.rotation_corrections, dim=1))
        max_tran = torch.max(torch.linalg.norm(self.translation_corrections.squeeze(2), dim=1))
        logger.info(f"GridNet KF pose corrections: max_rot={math.degrees(max_rot):.3f}deg, max_tran={max_tran:.3f}m.")
        
    def print_feature_info(self):
        logger.warning("Feature info not implemented yet for GridNGP.")
