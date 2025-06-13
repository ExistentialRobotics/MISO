import os
import math
import numpy as np
from typing import Tuple
from copy import deepcopy
import open3d as o3d
import torch
from torch import Tensor
from .base_net import BaseNet
import grid_opt.utils.utils as utils
import grid_opt.utils.utils_geometry as utils_geometry
import grid_opt.utils.utils_sdf as utils_sdf
from grid_opt.models.grid_net import GridNet
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GridAtlas(BaseNet):
    """ Modeling the scene as a collection of submaps, where each submap is 
    represented as a GridNet object.

    The current implementation assumes that keyframes and submaps are
    created in a sequential manner. Furthermore, the first keyframe added
    to each submap is used as the anchor keyframe for that submap.

    """
    def __init__(self,
        cfg: dict, 
        device = 'cuda:0',
        dtype = torch.float32,
    ):
        super(GridAtlas, self).__init__(cfg, device, dtype)  
        self.cfg = cfg
        self.submaps = torch.nn.ModuleList()
        self.rotation_corrections = torch.nn.ParameterList()
        self.translation_corrections = torch.nn.ParameterList()
        self.R_world_submap_list = []
        self.t_world_submap_list = []
        self._submap_anchor_kf = []
        self._kf_id_to_submap_id = []
        self._submap_id_to_kf_ids = dict()
        self.curr_submap_id = -1
        self.curr_kf_id = -1
    
    def lock_submap(self, submap_id: int):
        submap = self.get_submap(submap_id)
        submap.lock_feature()
        submap.lock_pose()

    def unlock_submap(self, submap_id: int):
        submap = self.get_submap(submap_id)
        submap.unlock_feature()
        submap.unlock_pose()
    
    def lock_submap_pose(self):
        logger.info(f"Lock submap pose variables in grid atlas.")
        for param in self.rotation_corrections:
            param.requires_grad_(False)
        for param in self.translation_corrections:
            param.requires_grad_(False)
    
    def lock_keyframe_pose(self):
        logger.info(f"Lock keyframe pose variables in grid atlas.")
        for submap_id in range(self.num_submaps):
            self.get_submap(submap_id).lock_pose()

    def unlock_submap_pose(self):
        logger.info(f"Unlock submap pose variables in grid atlas.")
        for param in self.rotation_corrections:
            param.requires_grad_(True)
        for param in self.translation_corrections:
            param.requires_grad_(True)
        
    def unlock_keyframe_pose(self):
        logger.info(f"Unlock keyframe pose variables in grid atlas.")
        for submap_id in range(self.num_submaps):
            self.get_submap(submap_id).unlock_pose()

    def print_keyframe_pose_info(self):
        for submap_id in range(self.num_submaps):
            submap = self.get_submap(submap_id)
            max_rot = torch.max(torch.linalg.norm(submap.rotation_corrections, dim=1))
            max_deg = math.degrees(max_rot)
            max_tran = torch.max(torch.linalg.norm(submap.translation_corrections.squeeze(2), dim=1))
            print(f"KF submap {submap_id} pose corrections: max_rot={max_deg:.2f}deg, max_tran={max_tran:.2f}m")
    
    def print_submap_pose_info(self):
        for submap_id in range(self.num_submaps):
            submap_rad = torch.linalg.norm(self.rotation_corrections[submap_id])
            submap_deg = math.degrees(submap_rad)
            print(f"Base submap {submap_id} pose corrections: rot={submap_deg:.2f}deg, tran={torch.linalg.norm(self.translation_corrections[submap_id]):.2f}m")

    def anchor_kf_for_submap(self, submap_id: int): 
        return self._submap_anchor_kf[submap_id]
    
    def add_kf(self, Rsk: torch.Tensor, tsk: torch.Tensor):
        """Add a keyframe to the current submap.

        Args:
            Rsk (torch.Tensor): Initial orientation of KF in submap
            tsk (torch.Tensor): Initial position of KF in submap
        """
        assert Rsk.shape == (3,3)
        assert tsk.shape == (3,1)
        assert self.curr_submap_id >= 0, "No submap is created yet. Create a submap first."
        submap_id = self.curr_submap_id
        kf_id_global = self.curr_kf_id + 1
        kf_id_submap = kf_id_global - self.anchor_kf_for_submap(self.curr_submap_id)
        logger.info(f"Add keyframe {kf_id_global} in submap {submap_id} (local id {kf_id_submap}).")
        self._kf_id_to_submap_id.append(submap_id)
        submap = self.get_submap(submap_id)
        submap.set_initial_kf_pose(kf_id_submap, Rsk, tsk, kf_key=f'KF{kf_id_global}')
        self._submap_id_to_kf_ids[submap_id].add(kf_id_global)
        self.curr_kf_id = kf_id_global
        return kf_id_global 

    def set_kf_pose(self, kf_id: int, Rsk: torch.Tensor, tsk: torch.Tensor):
        """Set the pose of a keyframe.

        Args:
            kf_id (int): Keyframe ID
            Rsk (torch.Tensor): New orientation of KF in submap
            tsk (torch.Tensor): New position of KF in submap
        """
        assert Rsk.shape == (3,3)
        assert tsk.shape == (3,1)
        submap_id = self.submap_id_for_kf(kf_id)
        submap = self.get_submap(submap_id)
        kf_id_submap = kf_id - self.anchor_kf_for_submap(submap_id)
        submap.set_initial_kf_pose(kf_id_submap, Rsk, tsk, kf_key=f'KF{kf_id}')
    
    def add_submap(self, local_bound: torch.Tensor, Rws: torch.Tensor, tws: torch.Tensor,
                   num_poses=1, optimize_poses=True):
        """Add a submap.

        Args:
            local_bound (torch.Tensor): Bound of this submap in submap frame.
            Rws (torch.Tensor): Initial orientation in world frame.
            tws (torch.Tensor): Initial position in world frame.
            num_poses (int, optional): Number of pose variables to initialize in submap. Defaults to 1.
            optimize_poses (bool, optional): Optimize pose variables in submap. Defaults to True.
        """
        assert Rws.shape == (3,3)
        assert tws.shape == (3,1)
        submap_id = len(self.submaps)
        cfg_model = deepcopy(self.cfg)
        cfg_model['grid']['bound'] = local_bound.numpy()
        cfg_model['pose']['num_poses'] = num_poses
        cfg_model['pose']['optimize'] = optimize_poses
        self.submaps.append(GridNet(cfg=cfg_model, device=self.device, dtype=self.dtype))
        self.R_world_submap_list.append(Rws.to(self.device))
        self.t_world_submap_list.append(tws.to(self.device))
        # Use the next keyframe ID as the anchor for the new submap
        anchor_kf = self.curr_kf_id + 1
        self._submap_anchor_kf.append(anchor_kf)
        self.rotation_corrections.append(torch.nn.Parameter(
            torch.zeros(1, 3).float().to(self.device), requires_grad=True
        ))
        self.translation_corrections.append(torch.nn.Parameter(
            torch.zeros(3, 1).float().to(self.device), requires_grad=True
        ))
        self.active_submaps = range(self.num_submaps)
        self.curr_submap_id = submap_id
        self._submap_id_to_kf_ids[submap_id] = set()
        self._submap_id_to_kf_ids[submap_id].add(anchor_kf)
        # Move inactive submaps out of GPU
        # for inactive_submap_id in range(self.curr_submap_id):
        #     self.get_submap(inactive_submap_id).to('cpu')
        logger.info(f"Added submap {submap_id} with anchor KF {anchor_kf} and bound:\n {local_bound.detach().cpu().numpy()}.")

    def set_submap_pose(self, submap_id: int, Rws: torch.Tensor, tws: torch.Tensor):
        """Set the pose of a submap.

        Args:
            submap_id (int): ID of submap.
            Rws (torch.Tensor): New orientation in world frame.
            tws (torch.Tensor): New position in world frame.
        """
        assert Rws.shape == (3,3)
        assert tws.shape == (3,1)
        with torch.no_grad():
            self.R_world_submap_list[submap_id].copy_(Rws.to(self.device))
            self.t_world_submap_list[submap_id].copy_(tws.to(self.device))
            # Since the base pose is updated, we will also reset 
            # the pose correction to zero.
            self.rotation_corrections[submap_id].copy_(torch.zeros_like(self.rotation_corrections[submap_id]))
            self.translation_corrections[submap_id].copy_(torch.zeros_like(self.translation_corrections[submap_id]))
    
    def set_submap_pose_correction(self, submap_id: int, R_delta: torch.Tensor, t_delta: torch.Tensor):
        """Set the pose correction for a submap.

        Args:
            submap_id (int): ID of submap.
            R_delta (torch.Tensor): rotation correction (1,3)
            t_delta (torch.Tensor): translation correction (3,1)
        """
        assert R_delta.shape == (1,3)
        assert t_delta.shape == (3,1)
        with torch.no_grad():
            self.rotation_corrections[submap_id].copy_(R_delta)
            self.translation_corrections[submap_id].copy_(t_delta)

    def set_active_submaps(self, active_submaps):
        self.active_submaps = active_submaps

    @property
    def num_submaps(self):
        return len(self.submaps)

    @property
    def num_active_submaps(self):
        return len(self.active_submaps)
    
    @property
    def num_keyframes(self):
        return self.curr_kf_id + 1
    
    def num_keyframes_in_submap(self, submap_id: int) -> int:
        """Number of keyframes in a submap.
        """
        return len(self._submap_id_to_kf_ids[submap_id])
    
    def submap_id_for_kf(self, kf_id: int):
        return self._kf_id_to_submap_id[kf_id]
    
    def submap_id_for_kf_batch(self, kf_ids: Tensor) -> Tensor:
        """Batch query for submap ID for a batch of keyframes.

        Args:
            kf_ids (Tensor): (N,) shaped tensor of keyframe IDs.

        Returns:
            Tensor: (N,) shaped tensor of submap IDs.
        """
        torch_kf_id_to_submap_id = torch.tensor(self._kf_id_to_submap_id, device=kf_ids.device)
        return torch_kf_id_to_submap_id[kf_ids]
    
    def initial_submap_pose(self, submap_id: int) -> Tuple[Tensor, Tensor]:
        """Initial submap pose in world frame.

        Args:
            submap_id (int): Submap Id

        Returns:
            tuple[Tensor, Tensor]: Rws (3,3), tws (3,1)
        """
        R_world_submap, t_world_submap = self.R_world_submap_list[submap_id], self.t_world_submap_list[submap_id]
        return R_world_submap, t_world_submap
    
    def updated_submap_pose(self, submap_id: int, device=None) -> Tuple[Tensor, Tensor]:
        """Optimized submap pose in world frame

        Args:
            submap_id (int): Submap id.

        Returns:
            tuple[Tensor, Tensor]: Rws (3,3), tws (3,1)
        """
        R_world_submap, t_world_submap = self.initial_submap_pose(submap_id)
        R_world_submap, t_world_submap = utils_geometry.apply_pose_correction(
            R=R_world_submap,
            t=t_world_submap,
            R_delta=self.rotation_corrections[submap_id],
            t_delta=self.translation_corrections[submap_id]
        )
        if device is not None:
            R_world_submap, t_world_submap = R_world_submap.to(device), t_world_submap.to(device)
        return R_world_submap, t_world_submap

    def initial_kf_pose_in_submap(self, kf_id:int, submap_id:int) -> Tuple[Tensor, Tensor]:
        """Initial KF pose in submap.

        Args:
            kf_id (int): keyframe ID.
            submap_id (int): submap ID. Should be the submap that contains this KF. Otherwise raises an error.

        Returns:
            tuple[Tensor, Tensor]: Rsk (3,3), tsk (3,1)
        """
        expect_submap_id = self.submap_id_for_kf(kf_id)
        assert expect_submap_id == submap_id, f"Wrong submap for KF {kf_id}! Expect {expect_submap_id}, got {submap_id}."
        kf_id_submap = kf_id - self.anchor_kf_for_submap(submap_id)
        submap = self.get_submap(submap_id)
        return submap.initial_kf_pose(kf_id_submap)
    
    def updated_kf_pose_in_submap(self, kf_id:int, submap_id:int) -> Tuple[Tensor, Tensor]:
        """Updated KF pose in submap.

        Args:
            kf_id (int): keyframe ID
            submap_id (int): submap ID. Should be the submap that contains this KF. Otherwise raises an error.

        Returns:
            tuple[Tensor, Tensor]: Rsk (3,3), tsk (3,1)
        """
        expect_submap_id = self.submap_id_for_kf(kf_id)
        assert expect_submap_id == submap_id, f"Wrong submap for KF {kf_id}! Expect {expect_submap_id}, got {submap_id}."
        kf_id_submap = kf_id - self.anchor_kf_for_submap(submap_id)
        submap = self.get_submap(submap_id)
        return submap.updated_kf_pose(kf_id_submap)
    
    def initial_kf_pose_in_world(self, kf_id: int) -> Tuple[Tensor, Tensor]:
        """Initial KF pose in world.

        Args:
            kf_id (int): Keyframe ID.

        Returns:
            tuple[Tensor, Tensor]: Rwk (3,3), twk (3,1)
        """
        submap_id = self.submap_id_for_kf(kf_id)
        R_submap_kf, t_submap_kf = self.initial_kf_pose_in_submap(kf_id, submap_id)
        R_world_submap, t_world_submap = self.initial_submap_pose(submap_id)
        return utils_geometry.transform_poses_to(R_submap_kf, t_submap_kf, R_world_submap, t_world_submap)
    
    def updated_kf_pose_in_world(self, kf_id: int) -> Tuple[Tensor, Tensor]:
        """Updated KF pose in world, accounting for both submap updates and keyframe update within submap

        Args:
            kf_id (int): Keyframe ID.

        Returns:
            tuple[Tensor, Tensor]: Rwk (3,3), twk (3,1)
        """
        submap_id = self.submap_id_for_kf(kf_id)
        R_submap_kf, t_submap_kf = self.updated_kf_pose_in_submap(kf_id, submap_id)
        R_world_submap, t_world_submap = self.updated_submap_pose(submap_id)
        return utils_geometry.transform_poses_to(R_submap_kf, t_submap_kf, R_world_submap, t_world_submap)

    def global_bound(self, device='cpu') -> torch.Tensor: 
        all_corners = []
        for submap_id in range(self.num_submaps):
            R_world_submap, t_world_submap = self.updated_submap_pose(submap_id)
            R_world_submap, t_world_submap = R_world_submap.to(device), t_world_submap.to(device)
            local_box = self.get_submap(submap_id).bound
            # Extract local corners
            x_min, x_max = local_box[0]
            y_min, y_max = local_box[1]
            z_min, z_max = local_box[2]
            corners_submap = torch.tensor([
                [x_min, y_min, z_min],
                [x_min, y_min, z_max],
                [x_min, y_max, z_min],
                [x_min, y_max, z_max],
                [x_max, y_min, z_min],
                [x_max, y_min, z_max],
                [x_max, y_max, z_min],
                [x_max, y_max, z_max]
            ], device=device)
            corners_world = utils_geometry.transform_points_to(corners_submap, R_world_submap, t_world_submap)
            all_corners.append(corners_world)
        all_corners = torch.cat(all_corners, dim=0)
        global_min = all_corners.min(dim=0).values  # Shape: (3,)
        global_max = all_corners.max(dim=0).values  # Shape: (3,)
        global_bnd = torch.stack([global_min, global_max], dim=1)
        return global_bnd

    @property
    def num_levels(self):
        return self.get_submap(0).num_levels
    
    def ignore_level(self, l):
        for submap in self.submaps:
            submap.ignore_level(l)
    
    def include_level(self, l):
        for submap in self.submaps:
            submap.include_level(l)
        
    def zero_features(self):
        for submap in self.submaps:
            submap.zero_features()
    
    def query_feature(self, x_world: torch.Tensor):
        # Separately probe each submap
        sum_feats = 0
        sum_weights = 0
        for submap_id in self.active_submaps:
            submap = self.get_submap(submap_id)
            # R_world_submap, t_world_submap = self.dataset.ground_truth_submap_pose(submap_id)
            R_world_submap, t_world_submap = self.updated_submap_pose(submap_id)
            x_submap = utils_geometry.transfrom_points_from(x_world, R_world_submap, t_world_submap) 
            mask_bnd = utils_geometry.coords_in_bound(x_submap, submap.bound) # N,1
            submap_feats = utils.grid_interp_regular(submap.features, x_submap, ignore_level=None)  # N, fdim
            sum_feats += mask_bnd * submap_feats
            sum_weights += mask_bnd
        sum_weights[sum_weights == 0] = 1
        sum_weights = sum_weights.float()
        mean_feats = sum_feats / sum_weights
        logger.debug(f"mean_feats.shape = {mean_feats.shape}")
        return mean_feats

    def forward(self, x_world:torch.Tensor, noise_std=0):
        mean_feats = self.query_feature(x_world)
        pred = utils.grid_decode(mean_feats, None, self.submaps[0].decoder, True)
        if noise_std > 0:
            noise = torch.randn(pred.shape, device=x_world.device) * noise_std
            pred = pred + noise
        return pred
    
    def get_submap(self, submap_id: int) -> GridNet:
        assert submap_id >= 0 and submap_id < self.num_submaps
        return self.submaps[submap_id]
    
    def check_submap_intersection(self, src_id: int, dst_id: int, overlap_thresh=1e-2):
        submap_src = self.get_submap(src_id)
        submap_dst = self.get_submap(dst_id)
        # check if any vertex in src map fall inside dst map
        corners_src = submap_src.features[-1].vertex_positions().to(self.device)
        R_world_src, t_world_src = self.updated_submap_pose(src_id)
        R_world_dst, t_world_dst = self.updated_submap_pose(dst_id)
        corners_world = utils_geometry.transform_points_to(corners_src, R_world_src, t_world_src)
        corners_dst = utils_geometry.transfrom_points_from(corners_world, R_world_dst, t_world_dst)
        mask_bnd = utils_geometry.coords_in_bound(corners_dst, submap_dst.bound)
        num_valid = torch.count_nonzero(mask_bnd)
        num_all = corners_src.shape[0]
        overlap_percentage = num_valid / num_all
        if num_valid > 0:
            logger.debug(f"Overlapping submap pair {src_id}-{dst_id} overlap percentage: {overlap_thresh: .2f}")
        return overlap_percentage > overlap_thresh
    
    def visualize(self, save_dir, resolution=512, visualize_sdf_plane=True):
        utils.cond_mkdir(save_dir)
        utils_sdf.save_mesh(
            self, 
            self.global_bound(),
            os.path.join(save_dir, f"mesh_final.ply"), 
            resolution=resolution
        )
        if visualize_sdf_plane:
            utils_sdf.visualize_sdf_plane(
                self, 
                self.global_bound(),
                resolution, 
                axis='z', 
                fig_path=os.path.join(save_dir, f"sdf_plane_final.png"),
                show_colorbar=False, show_title=False, hide_axis=True
            )
    
    def submap_obb_in_world(self, submap_id: int, use_pose='updated', color=[0,0,1]):
        submap = self.get_submap(submap_id)
        if use_pose == 'updated':
            R_world_submap, t_world_submap = self.updated_submap_pose(submap_id)
        elif use_pose == 'initial':
            R_world_submap, t_world_submap = self.initial_submap_pose(submap_id)
        else:
            raise ValueError(f"Unknown pose mode: {use_pose}")
        min_bound = submap.bound[:, 0].detach().cpu().numpy()
        max_bound = submap.bound[:, 1].detach().cpu().numpy()
        aabb_submap = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        points_submap = torch.from_numpy(np.asarray(aabb_submap.get_box_points())).float().to(R_world_submap)
        points_world = utils_geometry.transform_points_to(points_submap, R_world_submap, t_world_submap)
        obb_world = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(points_world.detach().cpu().numpy())) 
        obb_world.color = color
        # obb_submap =  o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(aabb_submap)
        # obb_world = obb_submap.rotate(R_world_submap.detach().cpu().numpy())
        # obb_world = obb_world.translate(t_world_submap.detach().cpu().numpy())
        # obb_world.color = color
        return obb_world
    
    def visualize_submaps(self, save_dir, use_pose='updated'):
        submap_meshes = []
        submap_pcds = []
        submap_boxs = []
        submap_poses = []
        for submap_id in range(self.num_submaps):
            submap_color = np.random.rand(3)
            submap = self.get_submap(submap_id)
            submap_dir = os.path.join(save_dir, f"submap{submap_id}")
            utils.cond_mkdir(submap_dir)
            mesh_tri = utils_sdf.save_mesh(
                submap, 
                submap.bound, 
                os.path.join(submap_dir, f"mesh_submap.ply"), 
                resolution=128
            )
            vertices = np.asarray(mesh_tri.vertices)
            faces = np.asarray(mesh_tri.faces)
            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
            mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
            if use_pose == 'gt':
                # R_world_submap, t_world_submap = self.dataset.ground_truth_submap_pose(submap_id)
                raise NotImplementedError
            elif use_pose == 'updated':
                R_world_submap, t_world_submap = self.updated_submap_pose(submap_id)
            elif use_pose == 'initial':
                R_world_submap, t_world_submap = self.initial_submap_pose(submap_id)
            else:
                raise ValueError(f"Unknown pose mode: {use_pose}")
            
            T_world_submap = np.eye(4)
            T_world_submap[:3,:3] = R_world_submap.detach().cpu().numpy()
            T_world_submap[:3, 3] = t_world_submap.detach().cpu().numpy().flatten()
            mesh_tri.apply_transform(T_world_submap)
            mesh_o3d.transform(T_world_submap)
            o3d_output_path = os.path.join(submap_dir, f"mesh_submap_transformed.ply")
            o3d.io.write_triangle_mesh(o3d_output_path, mesh_o3d)
            pcd = mesh_o3d.sample_points_uniformly(50000)
            pcd.paint_uniform_color(submap_color)
            submap_meshes.append(mesh_tri)
            submap_pcds.append(pcd)
            min_bound = submap.bound[:, 0].detach().cpu().numpy()
            max_bound = submap.bound[:, 1].detach().cpu().numpy()
            aabb_submap = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            obb_submap =  o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(aabb_submap)
            obb_world = obb_submap.rotate(R_world_submap.detach().cpu().numpy())
            obb_world = obb_world.translate(t_world_submap.detach().cpu().numpy())
            obb_world.color = submap_color
            submap_boxs.append(obb_world)
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            coord.transform(T_world_submap)
            coord.paint_uniform_color(submap_color)
            submap_poses.append(coord)

            # 2) visualize feature norm of each submap
            feats = submap.features[0].feature.squeeze().permute([3, 2, 1, 0])
            xsize, ysize, zsize, fdim = feats.shape
            grid = feats[:, :, zsize//2, :].detach().cpu().numpy()
            utils.visualize_grid_magnitude(grid, fig_path=os.path.join(submap_dir, f"featnorm_coarse.png"))

            feats = submap.features[1].feature.squeeze().permute([3, 2, 1, 0])
            xsize, ysize, zsize, fdim = feats.shape
            grid = feats[:, :, zsize//2, :].detach().cpu().numpy()
            utils.visualize_grid_magnitude(grid, fig_path=os.path.join(submap_dir, f"featnorm_fine.png"))

            # 3) visualize feature stability of each submap
            for level in range(submap.num_levels):
                grid_mu = submap.feature_stability[level].feature
                grid_mu = grid_mu.squeeze(0).permute([3, 2, 1, 0]) 
                xsize, ysize, zsize, fdim = grid_mu.shape
                for z_idx in range(zsize):
                    grid = grid_mu[:, :, z_idx, :].detach().cpu().numpy()
                    fig_path = os.path.join(submap_dir, f"mu_level{level}_z{z_idx}.png")
                    utils.visualize_grid_magnitude(grid, fig_path, log_scale=False)

        # combined_mesh = trimesh.util.concatenate(submap_meshes)
        # combined_mesh.export(join(args.save_dir, 'mesh_combined.ply'), file_type='ply')
        o3d.visualization.draw_geometries(submap_boxs + submap_poses)
    
    def params_for_submap_pose(self, submap_id):
        return [self.rotation_corrections[submap_id], self.translation_corrections[submap_id]]
    
    def params_for_all_submap_poses(self):
        return [*self.rotation_corrections, *self.translation_corrections]
    
    def params_for_all_kf_poses(self):
        params = []
        for submap_id in range(self.num_submaps):
            params += self.get_submap(submap_id).params_for_poses()
        return params
    
    def params_for_all_features(self):
        params = []
        for submap_id in range(self.num_submaps):
            params += self.get_submap(submap_id).params_for_features()
        return params
    
    def params_at_level(self, level):
        params = []
        for submap in self.submaps:
            params += submap.params_at_level(level)
        return params
    
    def precompute_coordinates_for_alignment(self, norm_thresh=1e-5):
        logger.info(f"Precomputing coordinates for alignment with norm threshold {norm_thresh:.1e}.")
        self._coords_for_alignment = dict()
        for level in range(self.num_levels):
            for submap_id in range(self.num_submaps):
                submap = self.get_submap(submap_id)
                coords = submap.features[level].vertex_positions().to(submap.device)
                feature = submap.query_feature(coords)
                featnorm_from = torch.linalg.norm(feature, dim=1, keepdim=True).detach()
                mask_feat = featnorm_from > norm_thresh
                valid_indices = torch.nonzero(mask_feat, as_tuple=False)[:, 0]
                coords_valid = coords[valid_indices, :]
                logger.info(f"Submap {submap_id} level {level} has {coords_valid.shape[0]}/{coords.shape[0]} valid coordinates.")
                key = f"submap{submap_id}_level{level}"
                self._coords_for_alignment[key] = coords_valid.detach()
        
    def coordinates_for_alignment(self, submap_id: int, level: int):
        assert submap_id >= 0 and submap_id < self.num_submaps
        assert level >= 0 and level < self.num_levels
        key = f"submap{submap_id}_level{level}"
        if key not in self._coords_for_alignment:
            raise ValueError(f"Coordinates for alignment not found for submap {submap_id} and level {level}. Did you call precompute_coordinates_for_alignment()?")
        return self._coords_for_alignment[key]
