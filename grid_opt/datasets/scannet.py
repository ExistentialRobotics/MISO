import torch
import grid_opt.utils.utils as utils
import grid_opt.utils.utils_sdf as utils_sdf
import grid_opt.utils.utils_geometry as utils_geometry
from grid_opt.utils.utils_sample import origin_dirs_W, ray_dirs_C
from grid_opt.utils.utils_sample import sample_pixels, get_batch_data, sample_along_rays
from grid_opt.models.encoder import EncoderObservation
from grid_opt.datasets.submap_dataset import SubmapDataset
import math
import numpy as np
import open3d as o3d
cosSim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ScanNet(SubmapDataset):

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        # dataset
        self.data_root = self.config['dataset']['path']
        self.inv_depth_scale = 1. / self.config['dataset']['depth_scale']
        self.bounds_method = self.config['dataset']['bounds_method']
        self.trunc_dist = self.config['dataset']['trunc_dist']
        self.normal_trunc_dist = 0.29365022    # iSDF normal based bounds (not used) 
        self.device = cfg['device']

        # Camera
        self.set_scannet_cam_params(self.config['dataset']['intrinsics_file'])

        self.dirs_C = ray_dirs_C(
            1, self.H, self.W, self.fx, self.fy, self.cx, self.cy, self.device, depth_type="z"
        )

        # Sampling
        self.max_depth = self.config["sample"]["depth_range"][1]
        self.min_depth = self.config["sample"]["depth_range"][0]
        self.dist_behind_surf = self.config["sample"]["dist_behind_surf"]
        self.n_rays = self.config["sample"]["n_rays"]
        self.n_strat_samples = self.config["sample"]["n_strat_samples"]
        self.n_surf_samples = self.config["sample"]["n_surf_samples"]
        self.batch_size = self.config["train"]["batch_size"]
        self.n_kframe = self.batch_size // (self.n_rays * (self.n_strat_samples + self.n_surf_samples))
        self.voxel_size = self.config['dataset']['voxel_size']  # voxel downsample

        # load selected keyframe idx
        kframe_select_file = f"{self.data_root}/sample_idxs.txt"
        self.online = self.config['dataset']['online'] and os.path.exists(kframe_select_file)
        
        if self.online:
            self.kframe_idxs = []
            with open(kframe_select_file, 'r') as f:
                for line in f:
                    tmp = []
                    line_data = line.strip().split()
                    for item in line_data:
                        tmp.append(int(item))
                    self.kframe_idxs.append(np.array(tmp))
        
        kframe_data_file = f"{self.data_root}/frame_data.pt"
        kframe_data = torch.load(kframe_data_file)
        self.depth_all = kframe_data['depth_batch']
        self.T_WC_all = kframe_data['T_WC_batch']
        self.T_WC_all_true = self.T_WC_all
        utils.check_tensor(self.T_WC_all_true)
        self.norm_all = kframe_data['norm_batch']
        self.total_kframe = self.depth_all.shape[0]
        self.n_kframe = min(self.n_kframe, self.total_kframe)
        # Load odometry from color ICP
        icp_odom_file = f"{self.data_root}/cam_poses_icp.npy"
        assert os.path.exists(icp_odom_file), f"ICP odometry file {icp_odom_file} does not exist."
        self.T_WC_all_odom = torch.from_numpy(np.load(icp_odom_file)).float()
        assert self.T_WC_all_odom.shape == self.T_WC_all_true.shape, "ICP odometry shape mismatch."

        submap_data_file = f"{self.data_root}/submaps.pt"
        if os.path.exists(submap_data_file):
            submap_data = torch.load(submap_data_file)
            # M * 6: [x, y, z, x_len, y_len, z_len] 
            # (x, y, z) is the center of the submap
            self.submaps = submap_data['submaps']
            # N * 2, each keyframe can be associated with at most two submaps
            # the elements are submap id, -1 means non submap association
            self.submap_assoc = submap_data['kframe_submap_assoc']
            self.keyframe_to_submap = self.submap_assoc[:, 0].detach().cpu().numpy().tolist()
            assert -1 not in self.keyframe_to_submap
            self.num_submaps = self.submaps.shape[0]
        assert 'anchor_kfs' in self.config['dataset'], "List of anchor KFs must be provided."
        self.anchor_kfs = self.config['dataset']['anchor_kfs']
        assert len(self.anchor_kfs) == self.num_submaps, "Number of anchor KFs must match number of submaps."
        
        self._getitem_from_single_submap = None
        self.set_gt_kf_poses_in_world()
        self.set_gt_submap_poses_in_world()
        self.set_gt_kf_poses_in_submap()
        self.set_odometry_kf_poses_from_icp()
        self.set_noisy_submap_poses_in_world()
        self.set_noisy_kf_poses_in_submap()
        logger.info(f"Constructed ScanNet dataset from {self.data_root}. KFs={self.total_kframe}, submaps={self.num_submaps}.")
        self.kframes_selected = None

    @property
    def num_kfs(self):
        return self.total_kframe
    
    def get_odometry_at_pose(self, src_id):
        """ Obtain the odometry from src_id to src_id+1.
        """
        T_odom_src = self.T_WC_all_odom[src_id, :, :]
        T_odom_dst = self.T_WC_all_odom[src_id+1, :, :]
        # note(yulun): uncomment below to use ground truth odometry
        # T_odom_src = self.T_WC_all_true[src_id, :, :]
        # T_odom_dst = self.T_WC_all_true[src_id+1, :, :]
        T_src_dst = torch.linalg.inv(T_odom_src) @ T_odom_dst
        return T_src_dst
    
    def sampled_points_at_kf(self, kf_id):
        old_select_keyframes = self.kframes_selected
        self.select_keyframes([kf_id])
        input_dict, _ = self.__getitem__(0)
        self.select_keyframes(old_select_keyframes)
        return input_dict['coords_frame']
    
    def select_keyframes(self, kframes_selected):
        self.kframes_selected = kframes_selected

    def unselect_keyframes(self):
        self.kframes_selected = None
    
    def submap_id_for_kf(self, kf_id: int):
        return self.keyframe_to_submap[kf_id]

    def getitem_from_single_submap(self, submap_id):
        self._getitem_from_single_submap = submap_id
    
    def getitem_from_all_submaps(self):
        self._getitem_from_single_submap = None

    def set_gt_kf_poses_in_world(self):
        self.R_world_kf_gt = utils_geometry.identity_rotations(self.total_kframe)  # N, 3, 3
        self.t_world_kf_gt = np.zeros((self.total_kframe, 3, 1))                   # N, 3, 1
        for kf_id in range(self.total_kframe):
            T = self.T_WC_all_true[kf_id, :, :]
            self.R_world_kf_gt[kf_id, :, :] = T[:3,:3]
            self.t_world_kf_gt[kf_id, :, 0] = T[:3, 3].detach().cpu().numpy()
        self.t_world_kf_gt = torch.from_numpy(self.t_world_kf_gt).float().to(self.R_world_kf_gt)
        utils.check_tensor(self.R_world_kf_gt)
        utils.check_tensor(self.t_world_kf_gt)

    def set_gt_submap_poses_in_world(self):
        self.R_world_submap_gt = utils_geometry.identity_rotations(self.num_submaps)          # N, 3, 3
        self.t_world_submap_gt = torch.zeros((self.num_submaps, 3, 1))                        # N, 3, 1
        self.submap_center_in_world_gt = np.zeros((self.num_submaps, 3, 1))
        # Use corresponding anchor KF as pose for each submap
        for submap_id in range(self.num_submaps):
            anchor_kf = self.anchor_kfs[submap_id]
            R_anchor, t_anchor = self.R_world_kf_gt[anchor_kf, :, :], self.t_world_kf_gt[anchor_kf, :, :]
            submap_center = self.submaps[submap_id, :3].detach().cpu().numpy().flatten()  # (3,)
            self.R_world_submap_gt[submap_id, :, :] = R_anchor
            self.t_world_submap_gt[submap_id, :, 0] = t_anchor.flatten()
            self.submap_center_in_world_gt[submap_id, :, 0] = submap_center
        self.submap_center_in_world_gt = torch.from_numpy(self.submap_center_in_world_gt).float()
    
    def set_gt_kf_poses_in_submap(self):
        self.R_submap_kf_gt_dict = {}
        self.t_submap_kf_gt_dict = {}
        # For each submap, get ALL keyframe poses expressed in this submap frame
        # We can do better, because technically each submap only needs to track its own KFs.
        for submap_id in range(self.num_submaps):
            R_world_submap = self.R_world_submap_gt[submap_id, :, :]  # 3,3
            t_world_submap = self.t_world_submap_gt[submap_id, :, :]  # 3,1
            R_submap_kf, t_submap_kf = utils_geometry.transform_poses_from(
                self.R_world_kf_gt,
                self.t_world_kf_gt,
                R_world_submap,
                t_world_submap
            )
            self.R_submap_kf_gt_dict[submap_id] = R_submap_kf
            self.t_submap_kf_gt_dict[submap_id] = t_submap_kf
            utils.check_tensor(R_submap_kf)
            utils.check_tensor(t_submap_kf)

    def set_noisy_submap_poses_in_world(self):
        # logger.info(f"Submap pose errors {error_meter:.3f} m, {error_rad:.3f} rad.")
        # sample rotation and translation perturbations 
        # translation_noises = utils_geometry.fixed_length_translations(self.num_submaps, length=0.30).unsqueeze(-1)  # N, 3, 1
        # rotation_noises = utils_geometry.fixed_angle_rotations(self.num_submaps, rad=math.radians(5))   # N, 3, 3
        # Do not perturb first submap pose
        # translation_noises[0,:] = 0.0  
        # rotation_noises[0,:,:] = torch.eye(3)
        # # simulated noisy frame pose estimates
        # self.t_world_submap = self.t_world_submap_gt + translation_noises  # (num_frames, 3, 1)
        # self.R_world_submap = torch.matmul(self.R_world_submap_gt, rotation_noises)  # (num_frame, 3, 3)
        self.R_world_submap = utils_geometry.identity_rotations(self.num_submaps)  # N, 3, 3
        self.t_world_submap = torch.zeros((self.num_submaps, 3, 1))                # N, 3, 1
        for submap_id in range(self.num_submaps):
            anchor_kf = self.anchor_kfs[submap_id]
            R_anchor, t_anchor = self.R_world_kf[anchor_kf, :, :], self.t_world_kf[anchor_kf, :, :]
            self.R_world_submap[submap_id, :, :] = R_anchor 
            self.t_world_submap[submap_id, :, 0] = t_anchor.flatten() 

    def set_noisy_kf_poses_in_submap(self):
        # logger.info(f"KF pose errors in submaps {error_meter:.3f} m, {error_rad:.3f} rad.")
        # Do this for every submap
        self.R_submap_kf_dict = {}
        self.t_submap_kf_dict = {}
        for submap_id in range(self.num_submaps):
            # sample rotation and translation perturbations 
            # translation_noises = utils_geometry.fixed_length_translations(self.total_kframe, length=error_meter).unsqueeze(-1)  # N, 3, 1
            # rotation_noises = utils_geometry.fixed_angle_rotations(self.total_kframe, rad=error_rad)   # N, 3, 3
            # # Do not perturb first KF pose (Do we need to do this?)
            # translation_noises[0,:] = 0.0  
            # rotation_noises[0,:,:] = torch.eye(3)
            # t_submap_kf = self.t_submap_kf_gt_dict[submap_id] + translation_noises
            # R_submap_kf = torch.matmul(self.R_submap_kf_gt_dict[submap_id], rotation_noises)
            R_world_submap = self.R_world_submap[submap_id, :, :]  # 3,3
            t_world_submap = self.t_world_submap[submap_id, :, :]  # 3,1
            R_submap_kf, t_submap_kf = utils_geometry.transform_poses_from(
                self.R_world_kf,
                self.t_world_kf,
                R_world_submap,
                t_world_submap
            )
            self.R_submap_kf_dict[submap_id] = R_submap_kf
            self.t_submap_kf_dict[submap_id] = t_submap_kf

    def set_odometry_kf_poses_from_icp(self):
        self.R_world_kf = self.T_WC_all_odom[:, :3, :3]
        self.t_world_kf = self.T_WC_all_odom[:, :3, [3]]
        assert self.R_world_kf.shape == (self.total_kframe, 3, 3)
        assert self.t_world_kf.shape == (self.total_kframe, 3, 1)
    
    def set_odometry_kf_poses_in_world(self, error_meter=0, error_rad=0):
        logger.info(f"Cumulative odometry pose error {error_meter:.3f} m, {math.degrees(error_rad):.3f} deg.")
        # sample rotation and translation perturbations 
        translation_noises = utils_geometry.fixed_length_translations(self.total_kframe-1, length=error_meter).unsqueeze(-1)  # N, 3, 1
        rotation_noises = utils_geometry.fixed_angle_rotations(self.total_kframe-1, rad=error_rad)   # N, 3, 3
        # simulate noisy odometry
        odom_T_list = []
        for src_id in range(self.total_kframe-1):
            dst_id = src_id + 1
            R_world_src, t_world_src = self.R_world_kf_gt[src_id, :, :], self.t_world_kf_gt[src_id, :, :]
            R_world_dst, t_world_dst = self.R_world_kf_gt[dst_id, :, :], self.t_world_kf_gt[dst_id, :, :]
            T_world_src = utils_geometry.pose_matrix(R_world_src, t_world_src)
            T_world_dst = utils_geometry.pose_matrix(R_world_dst, t_world_dst)
            T_src_dst = torch.linalg.inv(T_world_src) @ T_world_dst
            T_src_dst[:3,:3] = T_src_dst[:3,:3] @ rotation_noises[src_id, :, :]
            T_src_dst[:3,3]  = T_src_dst[:3,3] + translation_noises[src_id, :, :].flatten()
            odom_T_list.append(T_src_dst)
        # propagate noisy odometry to get noisy KF poses
        self.R_world_kf = utils_geometry.identity_rotations(self.total_kframe)     # N, 3, 3
        self.t_world_kf = torch.zeros((self.total_kframe, 3, 1))                   # N, 3, 1
        curr_id = 0
        R_world_curr, t_world_curr = self.R_world_kf_gt[0, :, :], self.t_world_kf_gt[0, :, :]
        T_world_curr = utils_geometry.pose_matrix(R_world_curr, t_world_curr)
        while True:
            self.R_world_kf[curr_id, :, :] = T_world_curr[:3,:3]
            self.t_world_kf[curr_id, :, 0] = T_world_curr[:3, 3]
            next_id = curr_id + 1
            if next_id >= self.total_kframe:
                break
            T_curr_next = odom_T_list[curr_id]
            T_world_curr = T_world_curr @ T_curr_next
            curr_id = next_id

    def set_odometry_kf_poses_in_world_simple(self, error_meter=0, error_rad=0):
        logger.info(f"Simple odometry pose error {error_meter:.3f} m, {math.degrees(error_rad):.3f} deg.")
        # sample rotation and translation perturbations 
        translation_noises = utils_geometry.fixed_length_translations(self.total_kframe, length=error_meter).unsqueeze(-1)  # N, 3, 1
        rotation_noises = utils_geometry.fixed_angle_rotations(self.total_kframe, rad=error_rad)   # N, 3, 3
        # Do not perturb first submap pose
        translation_noises[0,:] = 0.0  
        rotation_noises[0,:,:] = torch.eye(3)
        # simulated noisy frame pose estimates
        self.t_world_kf = self.t_world_kf_gt + translation_noises  # (num_frames, 3, 1)
        self.R_world_kf = torch.matmul(self.R_world_kf_gt, rotation_noises)  # (num_frame, 3, 3)

    def true_submap_pose_in_world(self, submap_id):
        Rwm = self.R_world_submap_gt[submap_id, :, :]  # 3,3
        twm = self.t_world_submap_gt[submap_id, :, :]  # 3,1
        return Rwm, twm
    
    def noisy_submap_pose_in_world(self, submap_id):
        Rwm = self.R_world_submap[submap_id, :, :]  # 3,3
        twm = self.t_world_submap[submap_id, :, :]  # 3,1
        return Rwm, twm
    
    def true_kf_pose_in_world(self, kf_id):
        Rwk = self.R_world_kf_gt[kf_id, :, :]  # 3,3
        twk = self.t_world_kf_gt[kf_id, :, :]  # 3,1
        return Rwk, twk
    
    def noisy_kf_pose_in_world(self, kf_id):
        # submap_id = self.submap_id_for_kf(kf_id)
        # Rws, tws = self.noisy_submap_pose_in_world(submap_id)
        # Rsk, tsk = self.noisy_kf_pose_in_submap(kf_id)
        # return utils_geometry.transform_poses_to(Rsk, tsk, Rws, tws)
        Rwk = self.R_world_kf[kf_id, :, :]  # 3,3
        twk = self.t_world_kf[kf_id, :, :]  # 3,1
        return Rwk, twk

    def noisy_kf_pose_in_submap(self, kf_id):
        submap_id = self.submap_id_for_kf(kf_id)
        Rsk = self.R_submap_kf_dict[submap_id][kf_id, :, :]
        tsk = self.t_submap_kf_dict[submap_id][kf_id, :, :]
        return Rsk, tsk
    
    def sanity_check_noisy_poses(self):
        for kf_id in range(self.total_kframe):
            submap_id = self.submap_id_for_kf(kf_id)
            Rsk, tsk = self.noisy_kf_pose_in_submap(kf_id)
            Rwk, twk = self.noisy_kf_pose_in_world(kf_id)
            Rws, tws = self.noisy_submap_pose_in_world(submap_id)
            Rwk2, twk2 = utils_geometry.transform_poses_to(
                Rsk, tsk, Rws, tws
            )
            rot_err = (Rwk - Rwk2).norm().item()
            tran_err = (twk - twk2).norm().item()
            # assert rot_err < 1e-5, f"Large rot error {rot_err}"
            # assert tran_err < 1e-5, f"Large tran error {tran_err}"
            logger.debug(f"Checked noisy pose for KF {kf_id}.")

    def local_bound_from_submap_center(self, submap_id:int) -> torch.Tensor:
        dim = self.submaps[submap_id, 3:].detach().cpu().numpy()
        x_len, y_len, z_len = dim[0], dim[1], dim[2]
        bound = np.asarray([[-x_len/2, x_len/2], [-y_len/2, y_len/2], [-z_len/2, z_len/2]])
        return torch.from_numpy(bound).float()
        # input_dict, gt_dict = self.__getitem__(0)
        # assert input_dict['submap_idxs'].ndim == 2
        # assert input_dict['coords_submap'].ndim == 2
        # indices_m = input_dict['submap_idxs'][:, 0]
        # indices = torch.nonzero(indices_m == submap_id, as_tuple=False).squeeze(1)
        # if indices.numel() == 0:
        #     raise ValueError(f"Failed to sample points for submap {submap_id}!")
        # points_submap = input_dict['coords_submap'][indices, :].detach().cpu().numpy()
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points_submap)
        # aabb = pcd.get_axis_aligned_bounding_box()
        # min_bound = torch.tensor(aabb.get_min_bound())  # Shape (3,)
        # max_bound = torch.tensor(aabb.get_max_bound())  # Shape (3,)
        # bound = torch.stack([min_bound, max_bound], dim=1)
        # assert bound.shape == (3,2)
        # buffer = 1.5
        # bound[:,0] = bound[:, 0] - buffer
        # bound[:,1] = bound[:, 1] + buffer
        # return bound

    def gt_submap_aabb_in_world(self, submap_id:int, color=[0,1,0]):
        center = self.submap_center_in_world_gt[submap_id, :, 0]
        bound = self.local_bound_from_submap_center(submap_id)
        min_bound = center + bound[:, 0]
        max_bound = center + bound[:, 1]
        # Convert to open3d aabb format
        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        aabb.color = color
        return aabb

    def compute_submap_local_bound(self, submap_id:int):
        Rws, tws = self.true_submap_pose_in_world(submap_id)
        # Rsw, tsw = Rws.T, -Rws.T @ tws
        aabb_world = self.gt_submap_aabb_in_world(submap_id)
        points_world = torch.from_numpy(np.asarray(aabb_world.get_box_points())).float().to(Rws)
        points_submap = utils_geometry.transfrom_points_from(points_world, Rws, tws)
        min_bound, _ = torch.min(points_submap, dim=0)
        max_bound, _ = torch.max(points_submap, dim=0)
        bound_local = torch.stack([min_bound, max_bound], dim=1)
        # print(bound_local)
        return bound_local

    def set_scannet_cam_params(self, file):
        info = {}
        with open(file, 'r') as f:
            for line in f.read().splitlines():
                split = line.split(' = ')
                info[split[0]] = split[1]
        self.fx = float(info['fx_depth'])
        self.fy = float(info['fy_depth'])
        self.cx = float(info['mx_depth'])
        self.cy = float(info['my_depth'])
        self.H = int(info['depthHeight'])
        self.W = int(info['depthWidth'])

    def sample_points(
        self,
        depth_batch,
        T_WC_batch,
        norm_batch=None,
        active_loss_approx=None,
        n_rays=None,
        dist_behind_surf=None,
        n_strat_samples=None,
        n_surf_samples=None,
    ):
        """
        Sample points by first sampling pixels, then sample depths along
        the backprojected rays.
        """
        if n_rays is None:
            n_rays = self.n_rays
        if dist_behind_surf is None:
            dist_behind_surf = self.dist_behind_surf
        if n_strat_samples is None:
            n_strat_samples = self.n_strat_samples
        if n_surf_samples is None:
            n_surf_samples = self.n_surf_samples

        n_frames = depth_batch.shape[0]
        if active_loss_approx is None:
            indices_b, indices_h, indices_w = sample_pixels(
                n_rays, n_frames, self.H, self.W, device=self.device)
        else:
            # indices_b, indices_h, indices_w = \
            #     active_sample.active_sample_pixels(
            #         n_rays, n_frames, self.H, self.W, device=self.device,
            #         loss_approx=active_loss_approx,
            #         increments_single=self.increments_single
            #     )
            raise Exception('Active sampling not currently supported.')

        get_masks = active_loss_approx is None
        (
            dirs_C_sample,
            depth_sample,
            norm_sample,
            T_WC_sample,
            binary_masks,
            indices_b,
            indices_h,
            indices_w
        ) = get_batch_data(
            depth_batch,
            T_WC_batch,
            self.dirs_C,
            indices_b,
            indices_h,
            indices_w,
            norm_batch=norm_batch,
            get_masks=get_masks,
        )

        max_depth = depth_sample + dist_behind_surf
        pc, z_vals = sample_along_rays(
            T_WC_sample,
            self.min_depth,
            max_depth,
            n_strat_samples,
            n_surf_samples,
            dirs_C_sample,
            gt_depth=depth_sample,
            grad=False,
        )

        sample_pts = {
            "depth_batch": depth_batch,
            "pc": pc,
            "z_vals": z_vals,
            "indices_b": indices_b,
            "indices_h": indices_h,
            "indices_w": indices_w,
            "dirs_C_sample": dirs_C_sample,
            "depth_sample": depth_sample,
            "T_WC_sample": T_WC_sample,
            "norm_sample": norm_sample,
            "binary_masks": binary_masks,
        }
        return sample_pts
    
    def __getitem__(self, index):
        # Notes:
        # self.depth_all: (num_keyframes, H, W)

        
        # load images and T_WC batch
        if self.online:
            # use only selected keyframes
            kframes_selected = self.kframe_idxs[index]
            
            depth_batch = self.depth_all[kframes_selected]
            T_WC_select = self.T_WC_all[kframes_selected]
            norm_batch = self.norm_all[kframes_selected]
        else:
            # use all keyframes
            depth_batch = self.depth_all
            T_WC_select = self.T_WC_all
            norm_batch = self.norm_all

            # random select kframe
            if self.kframes_selected is None:
                kframe_idxs = np.random.choice(len(depth_batch), self.n_kframe, replace=False)
            else:
                n_kframe = min(self.n_kframe, len(self.kframes_selected))
                kframe_idxs = np.random.choice(self.kframes_selected, n_kframe, replace=False)       
            
            logger.debug(f"n_kframe={self.n_kframe}, total_kframes={len(depth_batch)}")
            depth_batch = depth_batch[kframe_idxs]
            T_WC_select = T_WC_select[kframe_idxs]
            norm_batch = norm_batch[kframe_idxs]

        sample = self.sample_points(
            depth_batch, T_WC_select, norm_batch=norm_batch)

        pc = sample["pc"]
        z_vals = sample["z_vals"]
        dirs_C_sample = sample["dirs_C_sample"]
        depth_sample = sample["depth_sample"]
        T_WC_sample = sample["T_WC_sample"]
        norm_sample = sample["norm_sample"]

        bounds, grad_vec = compute_bounds(
            self.bounds_method,
            dirs_C_sample,
            depth_sample,
            T_WC_sample,
            z_vals,
            pc,
            self.normal_trunc_dist,
            norm_sample,
            do_grad=True,
        )

        # Filter out nan values in coords
        num_before_filter = pc.shape[0]
        pc_is_nan = torch.isnan(pc)
        valid_indices = (~torch.any(pc_is_nan.view(pc_is_nan.shape[0], -1), dim=1)).nonzero(as_tuple=True)[0]
        pc = pc[valid_indices]
        num_after_filter = pc.shape[0]
        logger.debug(f"{num_after_filter}/{num_before_filter} points left after nan filtering.")
        norm_sample = norm_sample[valid_indices]
        bounds = bounds[valid_indices]
        grad_vec = grad_vec[valid_indices]

        # Get correspondence between each sample and submap
        sampled_kframes = np.asarray(kframe_idxs)
        indices_b = sample["indices_b"].unsqueeze(-1)[valid_indices]   # (total_num_rays, 1)
        indices_k = torch.from_numpy(sampled_kframes[indices_b.squeeze().cpu().numpy()]).unsqueeze(-1)
        num_samples_per_ray = self.n_surf_samples + self.n_strat_samples
        indices_k = indices_k.unsqueeze(1).expand(-1, num_samples_per_ray, -1)
        indices_k = indices_k.reshape(-1,1).squeeze()   # association between sample and keyframe (total_num_samples,)
        indices_m = self.submap_assoc[indices_k, 0]     # association between sample and submaps (total_num_samples,)
        logger.debug(f"submap_assoc = {self.submap_assoc.shape}")
        logger.debug(f"indices_k {indices_k.shape}")
        logger.debug(f"indices_m {indices_m.shape}")
        
        # Convert pc from world to submap frames
        pc_world = pc.reshape(-1, 3)
        # Convert pc from world to kf and submap frames
        # pc in KF frame is GT
        # pc in submap frame and pc in world frame is subject to error
        pc_world_input = pc_world.clone()
        pc_kf = pc_world.clone()
        pc_submap = pc_world.clone()
        for kf_id in range(self.total_kframe):
            indices = torch.nonzero(indices_k == kf_id, as_tuple=False).squeeze(1)
            if indices.numel() == 0:
                continue
            Rwk, twk = self.true_kf_pose_in_world(kf_id)
            coords_world = pc_world[indices, :].to(Rwk)
            coords_kf = utils_geometry.transfrom_points_from(coords_world, Rwk, twk)
            pc_kf[indices, :] = coords_kf.to(pc_kf)
            # Convert to submap frame using the initial (noisy submap poses)
            # Currently, this is only needed by the encoder initialization
            R_submap_kf, t_submap_kf = self.noisy_kf_pose_in_submap(kf_id)
            coords_submap = utils_geometry.transform_points_to(coords_kf, R_submap_kf, t_submap_kf)
            pc_submap[indices, :] = coords_submap.to(pc_submap)
            # Convert to world frame using noisy KF poses
            R_world_kf_noisy, t_world_kf_noisy = self.noisy_kf_pose_in_world(kf_id)
            coords_world_noisy = utils_geometry.transform_points_to(coords_kf, R_world_kf_noisy, t_world_kf_noisy)
            pc_world_input[indices, :] = coords_world_noisy.to(pc_world_input)

        # TODO: Extend the getitem output to include normals
        gt_sdf = bounds.reshape(-1, 1)
        num_samples = gt_sdf.shape[0]
        coords_frame = pc_kf
        sample_frame_ids = indices_k.unsqueeze(1).long()
        weights = torch.ones_like(gt_sdf)
        gt_sdf_valid = torch.abs(gt_sdf) < self.trunc_dist
        gt_sdf_signs = torch.zeros_like(gt_sdf)
        gt_sdf_signs[gt_sdf < -self.trunc_dist] = -1
        gt_sdf_signs[gt_sdf > self.trunc_dist] = 1
        assert gt_sdf_valid.shape == (num_samples, 1)
        assert gt_sdf_signs.shape == (num_samples, 1)

        # Support optional voxel downsampling
        # FIXME: This is inefficient. Instead, any voxel downsampling could
        # be down offline for once
        if self.voxel_size is not None:
            down_idx = utils_geometry.voxel_down_sample_torch(
                coords_frame.detach().cpu(), self.voxel_size)
            coords_frame = coords_frame[down_idx, :]
            sample_frame_ids = sample_frame_ids[down_idx]
            weights = weights[down_idx, :]
            gt_sdf = gt_sdf[down_idx, :]
            gt_sdf_valid = gt_sdf_valid[down_idx, :]
            gt_sdf_signs = gt_sdf_signs[down_idx, :]

        input_dict = {
            'coords_frame': coords_frame,
            'sample_frame_ids': sample_frame_ids,
            'weights': weights,
        }
        gt_dict = {
            'sdf': gt_sdf,
            'sdf_valid': gt_sdf_valid,
            'sdf_signs': gt_sdf_signs,
        }

        return input_dict, gt_dict

    def get_encoder_observation_global(self, device='cpu', trunc_dist=0.15):
        model_input, gt = self.__getitem__(0)
        model_input, gt = utils.prepare_batch(model_input, gt, device)
        coords_world = model_input['coords']
        gt_sdf = gt['sdf']
        return EncoderObservation(
            coords_world=coords_world,
            gt_sdf=gt_sdf,
            gt_sdf_sign=utils_sdf.sign_mask_from_gt_sdf(gt_sdf, trunc_dist),
            gt_sdf_valid=utils_sdf.valid_mask_from_gt_sdf(gt_sdf, trunc_dist)
        )

    def get_encoder_observation_submaps(self, device='cpu', trunc_dist=0.15):
        output_obs_list = []
        for submap_id in range(self.num_submaps):
            self.getitem_from_single_submap(submap_id)
            model_input, gt = self.__getitem__(0)
            model_input, gt = utils.prepare_batch(model_input, gt, device)
            gt_sdf = gt['sdf']
            obs = EncoderObservation(
                coords_world=model_input['coords_submap'],
                gt_sdf=gt_sdf,
                gt_sdf_sign=utils_sdf.sign_mask_from_gt_sdf(gt_sdf, trunc_dist),
                gt_sdf_valid=utils_sdf.valid_mask_from_gt_sdf(gt_sdf, trunc_dist)
            )
            output_obs_list.append(obs)
        self.getitem_from_all_submaps()
        return output_obs_list
    
    def get_points_for_submap(self, submap_id, num_batches=1, trunc_dist=1e-3) -> np.ndarray:
        coords_list = []
        self.getitem_from_single_submap(submap_id)
        for _ in range(num_batches):
            model_input, gt = self.__getitem__(0)
            gt_sdf = gt['sdf']
            mask_valid = torch.abs(gt_sdf) < trunc_dist
            valid_indices = torch.nonzero(mask_valid, as_tuple=False)[:, 0]
            coords_list.append(model_input['coords_submap'][valid_indices, :])
        coords = torch.cat(coords_list, dim=0)
        self.getitem_from_all_submaps()
        return coords.detach().cpu().numpy()
    
    def __len__(self):
        if self.online:
            return len(self.kframe_idxs)
        else:
            # return (self.H * self.W) // self.n_rays
            # return int(self.total_kframe / self.n_kframe) + 1
            return 1
    

def bounds_ray(depth_sample, z_vals, dirs_C_sample, T_WC_sample, do_grad):
    bounds = depth_sample[:, None] - z_vals
    z_to_euclidean_depth = dirs_C_sample.norm(dim=-1)
    bounds = z_to_euclidean_depth[:, None] * bounds

    grad = None
    if do_grad:
        grad = grad_ray(T_WC_sample, dirs_C_sample, z_vals.shape[1] - 1)

    return bounds, grad


def bounds_normal(
    depth_sample, z_vals, dirs_C_sample, norm_sample, normal_trunc_dist,
    T_WC_sample, do_grad,
):
    ray_bounds = bounds_ray(depth_sample, z_vals, dirs_C_sample)

    costheta = torch.abs(cosSim(-dirs_C_sample, norm_sample))

    # only apply correction out to truncation distance
    sub = normal_trunc_dist * (1. - costheta)
    normal_bounds = ray_bounds - sub[:, None]

    trunc_ixs = ray_bounds < normal_trunc_dist
    trunc_vals = (ray_bounds * costheta[:, None])[trunc_ixs]
    normal_bounds[trunc_ixs] = trunc_vals

    grad = None
    if do_grad:
        grad = grad_ray(T_WC_sample, dirs_C_sample, z_vals.shape[1] - 1)

    return normal_bounds, grad


def grad_ray(T_WC_sample, dirs_C_sample, n_samples):
    """ Returns the negative of the viewing direction vector """
    _, dirs_W = origin_dirs_W(T_WC_sample, dirs_C_sample)
    grad = - dirs_W[:, None, :].repeat(1, n_samples, 1)

    return grad


def bounds_pc(pc, z_vals, depth_sample, do_grad=True):
    with torch.set_grad_enabled(False):
        surf_pc = pc[:, 0]
        diff = pc[:, :, None] - surf_pc
        dists = diff.norm(dim=-1)
        dists, closest_ixs = dists.min(axis=-1)
        behind_surf = z_vals > depth_sample[:, None]
        dists[behind_surf] *= -1
        bounds = dists

        grad = None
        if do_grad:
            ix1 = torch.arange(
                diff.shape[0])[:, None].repeat(1, diff.shape[1])
            ix2 = torch.arange(
                diff.shape[1])[None, :].repeat(diff.shape[0], 1)
            grad = diff[ix1, ix2, closest_ixs]
            grad = grad[:, 1:]
            grad = grad / grad.norm(dim=-1)[..., None]
            # flip grad vectors behind the surf
            grad[behind_surf[:, 1:]] *= -1

    return bounds, grad


def compute_bounds(
    method,
    dirs_C_sample,
    depth_sample,
    T_WC_sample,
    z_vals,
    pc,
    normal_trunc_dist,
    norm_sample,
    do_grad=True,
):
    """ do_grad: compute approximate gradient vector. """
    assert method in ["ray", "normal", "pc"]

    if method == "ray":
        bounds, grad = bounds_ray(
            depth_sample, z_vals, dirs_C_sample, T_WC_sample, do_grad
        )

    elif method == "normal":
        bounds, grad = bounds_normal(
            depth_sample, z_vals, dirs_C_sample,
            norm_sample, normal_trunc_dist, T_WC_sample, do_grad)

    else:
        bounds, grad = bounds_pc(pc, z_vals, depth_sample, do_grad)

    return bounds, grad



    