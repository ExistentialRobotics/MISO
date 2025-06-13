# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from ..base_net import BaseNet
from .embedding import PostionalEncoding
import grid_opt.utils.utils_geometry as utils_geometry
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def fc_block(in_f, out_f):
    return torch.nn.Sequential(
        torch.nn.Linear(in_f, out_f),
        torch.nn.Softplus(beta=100)
    )


def init_weights(m, init_fn=torch.nn.init.xavier_normal_):
    if isinstance(m, torch.nn.Linear):
        init_fn(m.weight)


class iSDF(BaseNet):
    def __init__(
        self,
        cfg: dict, 
        device = 'cuda:0',
        dtype = torch.float32,
        positional_encoding=PostionalEncoding(),
        hidden_size=256,
        hidden_layers_block=1,
        scale_output=1.,
    ):
        super(iSDF, self).__init__(cfg, device, dtype)

        self.init_poses(cfg)

        self.scale_output = scale_output

        self.positional_encoding = positional_encoding
        embedding_size = self.positional_encoding.embedding_size

        self.in_layer = fc_block(embedding_size, hidden_size)

        hidden1 = [fc_block(hidden_size, hidden_size)
                   for _ in range(hidden_layers_block)]
        self.mid1 = torch.nn.Sequential(*hidden1)

        self.cat_layer = fc_block(
            hidden_size + embedding_size, hidden_size)

        hidden2 = [fc_block(hidden_size, hidden_size)
                   for _ in range(hidden_layers_block)]
        self.mid2 = torch.nn.Sequential(*hidden2)

        self.out_alpha = torch.nn.Linear(hidden_size, 1)

        self.apply(init_weights)

    def init_poses(self, cfg):
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
        self.Rwk = utils_geometry.identity_rotations(self.num_poses).to(self.device)
        self.twk = torch.zeros(size=(self.num_poses, 3, 1), device=self.device)
        logger.info(f"Initialized {self.num_poses} pose variables, optimize={self.optimize_pose}.")
    
    def lock_pose(self):
        self.rotation_corrections.requires_grad_(False)
        self.translation_corrections.requires_grad_(False)
     
    def unlock_pose(self):
        self.rotation_corrections.requires_grad_(True)
        self.translation_corrections.requires_grad_(True)
    
    def set_initial_kf_pose(self, kf_id: int, Rwk: torch.Tensor, twk: torch.Tensor):
        assert Rwk.shape == (3,3)
        assert twk.shape == (3,1)
        self.pose_estimates_known[kf_id] = True
        self.Rwk[kf_id,: ,: ] = Rwk.to(self.device)
        self.twk[kf_id, :, :] = twk.to(self.device)
    
    def initial_kf_pose_in_world(self, kf_id: int):
        assert self.pose_estimates_known[kf_id], f"Initial pose estimate for KF {kf_id} is not available!"
        return self.Rwk[kf_id, :, :], self.twk[kf_id, :, :]
    
    def updated_kf_pose_in_world(self, kf_id: int):
        Rwk, twk = self.initial_kf_pose_in_world(kf_id)
        return utils_geometry.apply_pose_correction(
            Rwk, twk, self.rotation_corrections[[kf_id], :], self.translation_corrections[kf_id, :, :]
        )
    
    def print_kf_pose_info(self):
        max_rot = torch.max(torch.linalg.norm(self.rotation_corrections, dim=1))
        max_tran = torch.max(torch.linalg.norm(self.translation_corrections.squeeze(2), dim=1))
        print(f"iSDF KF pose corrections: max_rot={max_rot:.2e}, max_tran={max_tran:.2e}")

    def forward(self, x, noise_std=None, pe_mask=None, sdf1=None):

        # Handle case when x has ndim > 2
        flattened = False
        if x.ndim > 2:
            flattened = True
            H, W = x.shape[0], x.shape[1]
            x = torch.reshape(x, (H * W, x.shape[2]))


        # YT: When x is 2D, let's pad with additional dimension
        # TODO: better way of handling this
        if x.shape[1] == 2:    
            N = x.shape[0]
            x = torch.column_stack((x, torch.zeros(N, 1, device=x.device)))

        x_pe = self.positional_encoding(x)
        if pe_mask is not None:
            x_pe = torch.mul(x_pe, pe_mask)

        fc1 = self.in_layer(x_pe)
        fc2 = self.mid1(fc1)
        fc2_x = torch.cat((fc2, x_pe), dim=-1)
        fc3 = self.cat_layer(fc2_x)
        fc4 = self.mid2(fc3)
        raw = self.out_alpha(fc4)

        if noise_std is not None:
            noise = torch.randn(raw.shape, device=x.device) * noise_std
            raw = raw + noise
        alpha = raw * self.scale_output

        if flattened:
            alpha = torch.reshape(alpha, (H, W, 1))

        return alpha
