import os
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from grid_opt.utils.utils_geometry import *
from grid_opt.datasets.submap_dataset import SubmapDataset
import logging
logger = logging.getLogger(__name__)

class PosedSdf3DLidar(SubmapDataset):
    """
    A dataset that generates SDF samples from downsampled LiDAR frames.
    Each frame is considered as surface points, and additional samples 
    (near-surface, free-space, behind-surface) are generated with truncation.
    """
    def __init__(
        self,
        lidar_folder,
        pose_file_gt,
        pose_file_init,
        frame_batchsize=2**10,
        frame_samples=2**10,
        near_surface_n=4,
        near_surface_std=0.1,
        free_space_n=2,
        behind_surface_n=1,
        trunc_dist=0.50,
        num_frames=None,
        distance_std=0.,
        min_dist_ratio=0.30,
        voxel_size=0.03,
        min_z = -3.0,
        max_z = 60.0,
        min_range=1.5, 
        max_range=60.0,
        adaptive_range=True
    ):

        super().__init__()
        self.lidar_folder = lidar_folder
        self.frame_batchsize = frame_batchsize
        self.frame_samples = frame_samples
        self.near_surface_n = near_surface_n
        self.near_surface_std = near_surface_std
        self.free_space_n = free_space_n
        self.behind_surface_n = behind_surface_n
        self.trunc_dist = trunc_dist
        self.distance_std = distance_std
        self.min_dist_ratio = min_dist_ratio
        self.max_range_hehind_surface = 4 * self.near_surface_std   # Set as suggested by PIN-SLAM (Table II)
        self.adaptive_range = adaptive_range
        self.voxel_size = voxel_size
        self.min_z = min_z
        self.max_z = max_z
        self.min_range = min_range
        self.max_range = max_range

        # 1) Read pose files
        self.pose_list_gt = self.read_poses(pose_file_gt)
        self.pose_list_init = self.read_poses(pose_file_init)
        num_poses = min(len(self.pose_list_gt), len(self.pose_list_init))
        self.pose_list_gt = self.pose_list_gt[:num_poses]
        self.pose_list_init = self.pose_list_init[:num_poses]
        logger.info(f"Read {num_poses} poses from files.")

        # 2) Collect all .pcd (or .ply) files in the given folder
        pcd_files = sorted([f for f in os.listdir(lidar_folder) if f.endswith('.pcd') or f.endswith('.ply')])
        # If num_frames is specified and smaller than total .pcd files, truncate the list
        if num_frames is not None and num_frames < len(pcd_files):
            pcd_files = pcd_files[:num_frames]

        # number of pose lines or number of .pcd files
        n_usable = min(len(self.pose_list_gt), len(pcd_files))
        if n_usable == 0:
            raise ValueError("No usable frames. Check your data.")
        print(f"Dataset has {n_usable} usable frames.")
        self._num_frames = n_usable

        # Create tensors to store ground-truth (GT) poses for each frame
        self.R_world_frame_gt = torch.zeros((self._num_frames, 3, 3), dtype=torch.float32)
        self.t_world_frame_gt = torch.zeros((self._num_frames, 3, 1), dtype=torch.float32)
        self.R_world_frame = self.R_world_frame_gt.clone()
        self.t_world_frame = self.t_world_frame_gt.clone()
        for i in range(self._num_frames):
            R, t = self.pose_list_gt[i]
            self.R_world_frame_gt[i] = torch.from_numpy(R)
            self.t_world_frame_gt[i] = torch.from_numpy(t)
            R, t = self.pose_list_init[i]
            self.R_world_frame[i] = torch.from_numpy(R)
            self.t_world_frame[i] = torch.from_numpy(t)

        # 3) Load each .pcd LiDAR frame, downsample it, and convert points to global coordinates
        self.frames_lidar = []
        
        for idx in tqdm(range(n_usable), desc="Loading Lidar frames"):
            pcd_name = pcd_files[idx]
            Rf_gt, tf_gt = self.pose_list_gt[idx]

            # Read the point cloud
            pcd = o3d.io.read_point_cloud(os.path.join(lidar_folder, pcd_name))
            pts = np.asarray(pcd.points, dtype=np.float32)
            if pts.shape[0] == 0:
                logger.warning(f"{pcd_name} has 0 points. Skipping.")
                continue
            pts_torch = torch.from_numpy(pts).float()

            # Adaptive voxel downsampling
            if self.adaptive_range:
                pc_max_bound, _ = torch.max(pts_torch, dim=0)
                pc_min_bound, _ = torch.min(pts_torch, dim=0)
                min_x_range = min(torch.abs(pc_max_bound[0]), torch.abs(pc_min_bound[0]))
                min_y_range = min(torch.abs(pc_max_bound[1]), torch.abs(pc_min_bound[1]))
                max_x_y_min_range = max(min_x_range, min_y_range)
                crop_max_range = min(self.max_range, 2.0 * max_x_y_min_range)
            else:
                crop_max_range = self.max_range
            adapt_voxel_m = (
                crop_max_range / self.max_range
            ) * self.voxel_size
            down_idx = voxel_down_sample_torch(pts_torch, adapt_voxel_m)
            pts_torch = pts_torch[down_idx]
            # logger.info(f"Adaptive voxel downsampling frame {idx}: voxel_size={adapt_voxel_m:.2f}, num_points {pts_torch.shape[0]}.")

            # Cropping
            pts_torch, _ = crop_points(
                pts_torch,
                None,
                self.min_z,
                self.max_z,
                self.min_range,
                crop_max_range,
            )
            # logger.info(f"Cropping frame {idx}:, min_z={self.min_z}, max_z={self.max_z}, min_range={self.min_range}, max_range={crop_max_range}.")
            pts_down = pts_torch.detach().cpu().numpy()

            # Transform the downsampled points into the global coordinate frame
            pts_global = (Rf_gt @ pts_down.T) + tf_gt
            pts_global = pts_global.T

            # ---------------------------
            #  Bounding box filtering
            # ---------------------------
            # mask_x = (pts_global[:, 0] >= x_min) & (pts_global[:, 0] <= x_max)
            # mask_y = (pts_global[:, 1] >= y_min) & (pts_global[:, 1] <= y_max)
            # mask_z = (pts_global[:, 2] >= z_min) & (pts_global[:, 2] <= z_max)
            # mask = mask_x & mask_y & mask_z
            # pts_global_filtered = pts_global[mask]
            # pts_down_filtered = pts_down[mask]
            pts_global_filtered = pts_global
            pts_down_filtered = pts_down
            if pts_global_filtered.shape[0] == 0:
                logger.warning(f"{pcd_name} has no points in the specified bounding box. Skipping.")
                continue

            # Store relevant info for each frame
            self.frames_lidar.append({
                "R": Rf_gt,
                "t": tf_gt,
                "points_local": pts_down_filtered,  
                "points_global": pts_global_filtered 
            })

        # 4) For each frame, generate SDF samples (surface, near-surface, free-space, behind-surface)
        self.frames_data = []
        self.sample_frames()
        
        # 6) Getitem settings
        self._selected_kfs = None

        logger.info((f"Constructed dataset with settings: \nnear_surface_std={near_surface_std}, \nnear_surface_n={near_surface_n},"
                    f"\nfree_space_n={free_space_n}, \nbehind_surface_n={behind_surface_n}, \ntrunc_dist={trunc_dist}, "
                    f"\ndistance_std={distance_std}, \nmin_dist_ratio={min_dist_ratio}, \nvoxel_size={voxel_size:.3f}, " 
                    f"\nframe_batchsize={frame_batchsize}, \nframe_samples={frame_samples}."))
    
    @property
    def num_kfs(self):
        return self._num_frames
    
    def sampled_points_at_kf(self, kf_id):
        frame_data = self.frames_data[kf_id]
        return frame_data['points_frame']
    
    def read_poses(self, pose_file):
        T_list = read_kitti_format_poses(pose_file)
        pose_list = []
        for T in T_list:
            R = T[:3, :3]
            t = T[:3, 3:]
            pose_list.append((R, t))
        print(f"Read {len(pose_list)} poses from file:", pose_file)
        return pose_list
    
    def get_odometry_at_pose(self, src_id):
        """ Obtain the odometry from src_id to src_id+1.
        """
        R_odom_src, t_odom_src = self.noisy_kf_pose_in_world(src_id)
        T_odom_src = pose_matrix(R_odom_src, t_odom_src)
        R_odom_dst, t_odom_dst = self.noisy_kf_pose_in_world(src_id+1)
        T_odom_dst = pose_matrix(R_odom_dst, t_odom_dst)
        T_src_dst = torch.linalg.inv(T_odom_src) @ T_odom_dst
        return T_src_dst

    
    def distance_weight_func(self, dists, dist_weight_scale=0.8):
        """A weighting function for distance samples as in PIN-SLAM.
        """
        weights = 1 + dist_weight_scale * 0.5 - (dists / self.max_range) * dist_weight_scale
        assert weights.shape == dists.shape
        assert weights.min() >= 0.0
        return weights

    
    def sample_frames(self):
        """
        Generate SDF samples for each frame and store them in self.frames_data.
        """
        for frame_id in tqdm(range(self._num_frames), desc="Sampling frames"):
            # Ground-truth pose for the current frame
            Rwf_gt = self.R_world_frame_gt[frame_id].numpy()
            twf_gt = self.t_world_frame_gt[frame_id].numpy().flatten()

            # LiDAR center (the origin in the world frame) for this frame
            eye = twf_gt

            # Surface points in global coordinates
            pts_surface = self.frames_lidar[frame_id]["points_global"]
            n_surf = pts_surface.shape[0]
            if n_surf == 0:
                logger.warning(f"Frame {frame_id} has no points after downsample!")
                continue

            # Subsample the surface points to a maximum of self.frame_samples
            n_keep = min(self.frame_samples, n_surf)
            perm_indices = np.random.permutation(n_surf)
            pts_surface = pts_surface[perm_indices[:n_keep], :]
            n_surf = pts_surface.shape[0]

            # Distance from 'eye' to each surface point
            dist_surface = np.linalg.norm(pts_surface - eye[None, :], axis=1, keepdims=True)
            # Surface points have SDF = 0
            sdf_surface = np.zeros((n_surf, 1), dtype=np.float32)
            weights_surface = self.distance_weight_func(dist_surface)
            sample_points_all = [pts_surface]
            sample_sdfs_all = [sdf_surface]
            sample_weights_all = [weights_surface]
            sample_signs_all = [np.zeros_like(sdf_surface)]

            # =====================
            #  Near-surface points
            # =====================
            if self.near_surface_n > 0:
                repeated_surf = np.repeat(pts_surface, self.near_surface_n, axis=0)
                repeated_dist = np.repeat(dist_surface, self.near_surface_n, axis=0)
                directions = repeated_surf - eye[None, :]
                directions /= (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
                displacement = np.random.randn(repeated_surf.shape[0], 1) * self.near_surface_std
                d_near = repeated_dist + displacement
                near_surf_points = eye[None, :] + directions * d_near
                sdf_near = (repeated_dist - d_near).astype(np.float32)
                weights_near = self.distance_weight_func(repeated_dist)
                sample_points_all.append(near_surf_points)
                sample_sdfs_all.append(sdf_near)
                sample_weights_all.append(weights_near)
                sample_signs_all.append(np.zeros_like(sdf_near))

            # =====================
            #  Free-space points
            # =====================
            if self.free_space_n > 0:
                repeated_surf_fs = np.repeat(pts_surface, self.free_space_n, axis=0)
                repeated_dist_fs = np.repeat(dist_surface, self.free_space_n, axis=0)
                directions_fs = repeated_surf_fs - eye[None, :]
                directions_fs /= (np.linalg.norm(directions_fs, axis=1, keepdims=True) + 1e-8)
                max_dist_ratio_per_ray = 1.0 - self.trunc_dist / repeated_dist_fs
                ratio_range_per_ray = np.maximum(max_dist_ratio_per_ray - self.min_dist_ratio, 1e-2)
                dist_ratio = self.min_dist_ratio + np.random.rand(repeated_surf_fs.shape[0], 1) * ratio_range_per_ray
                displacement_fs = (dist_ratio - 1.0) * repeated_dist_fs
                d_fs = repeated_dist_fs + displacement_fs
                free_space_points = eye[None, :] + directions_fs * d_fs
                sdf_fs = -displacement_fs.astype(np.float32)
                sample_points_all.append(free_space_points)
                sample_sdfs_all.append(sdf_fs)
                sample_weights_all.append(np.ones_like(sdf_fs))
                sample_signs_all.append(np.ones_like(sdf_fs))

            # =====================
            #  Behind-surface points
            # =====================
            if self.behind_surface_n > 0:
                repeated_surf_bh = np.repeat(pts_surface, self.behind_surface_n, axis=0)
                repeated_dist_bh = np.repeat(dist_surface, self.behind_surface_n, axis=0)
                directions_bh = repeated_surf_bh - eye[None, :]
                directions_bh /= (np.linalg.norm(directions_bh, axis=1, keepdims=True) + 1e-8)
                displacement_bh =  self.near_surface_std +  np.random.rand(repeated_surf_bh.shape[0], 1) * (self.max_range_hehind_surface - 2 * self.near_surface_std) 
                d_bh = repeated_dist_bh + displacement_bh
                behind_surface_points = eye[None, :] + directions_bh * d_bh
                sdf_bh = -displacement_bh.astype(np.float32)
                sample_points_all.append(behind_surface_points)
                sample_sdfs_all.append(sdf_bh)
                sample_weights_all.append(np.ones_like(sdf_bh))
                sample_signs_all.append(-np.ones_like(sdf_bh))

            # =====================
            #  Merge all samples
            # =====================
            points_world_np = np.concatenate(sample_points_all, axis=0)
            sdfs_np = np.concatenate(sample_sdfs_all, axis=0)
            weights_np = np.concatenate(sample_weights_all, axis=0)
            signs_np = np.concatenate(sample_signs_all, axis=0)

            # Optionally add noise to the surface points themselves
            if self.distance_std > 1e-12:
                raise ValueError("Noise on surface points not implemented yet.")

            # Convert these samples from world to the frame's local coordinate system
            Rwf_gt_t = torch.from_numpy(Rwf_gt).float()
            twf_gt_t = torch.from_numpy(twf_gt).float().unsqueeze(-1)
            points_frame_np = transfrom_points_from(
                torch.from_numpy(points_world_np).float(),
                Rwf_gt_t,
                twf_gt_t
            ).numpy()

            # Only use samples within a certain truncation distance as direct supervision
            sdfs_valid_np = (np.abs(sdfs_np) < self.trunc_dist).astype(np.float32)

            # -------------------------------
            # Make free-space samples invalid
            # -------------------------------
            # n_near_total = n_surf * self.near_surface_n
            # n_fs_total = n_surf * self.free_space_n
            # # Free space index range:
            # fs_start = n_surf + n_near_total
            # fs_end = n_surf + n_near_total + n_fs_total
            # sdfs_valid_np[fs_start:fs_end] = 0.0  # mark free-space as invalid

            # Store the sampled data in a dictionary
            frame_data = {
                "points_frame": torch.from_numpy(points_frame_np).float(),
                "points_world_gt": torch.from_numpy(points_world_np).float(),
                "sdfs": torch.from_numpy(sdfs_np).float(),
                "sdfs_valid": torch.from_numpy(sdfs_valid_np).float(),
                "signs": torch.from_numpy(signs_np).float(),
                "weights": torch.from_numpy(weights_np).float()
            }
            self.frames_data.append(frame_data)

    def true_kf_pose_in_world(self, kf_id):
        Rwk = self.R_world_frame_gt[kf_id, :, :]  # 3,3
        twk = self.t_world_frame_gt[kf_id, :, :]  # 3,1
        return Rwk, twk
    
    def noisy_kf_pose_in_world(self, kf_id):
        Rwk = self.R_world_frame[kf_id, :, :]  # 3,3
        twk = self.t_world_frame[kf_id, :, :]  # 3,1
        return Rwk, twk
    
    def __len__(self):
        """
        Returns:
            (int): Approximate number of "batches" if we fetch 'frame_batchsize' from each frame.
        """
        augment_factor = 1 + self.near_surface_n + self.free_space_n + self.behind_surface_n
        # return augment_factor * self.frame_samples // self.frame_batchsize
        return 1
        
    def select_keyframes(self, kf_ids):
        self._selected_kfs = list(kf_ids)

    def unselect_keyframes(self):
        self._selected_kfs = None
    
    def getitem_world(self, index):
        """
        Returns:
            input_dict, gt_dict: Two dictionaries containing input and ground-truth data.
        """
        points_frame = []
        sdfs = []
        sdfs_valid = []
        sdfs_sign = []
        weights = []
        sample_frame_ids = []
        kf_list = range(self._num_frames)
        if self._selected_kfs is not None:
            kf_list = list(set(kf_list).intersection(self._selected_kfs))

        # Sample from each frame_data
        for frame_id in kf_list:
            frame_data = self.frames_data[frame_id]
            frame_size = frame_data["points_frame"].shape[0]
            frame_batchsize = min(self.frame_batchsize, frame_size)

            # Randomly pick a subset of points for this batch
            selected_indices = np.random.choice(frame_size, size=frame_batchsize, replace=False)
            cur_points_frame = frame_data["points_frame"][selected_indices, :]
            cur_sdfs = frame_data["sdfs"][selected_indices, :]
            cur_sdfs_valid = frame_data["sdfs_valid"][selected_indices, :]
            cur_sdfs_sign = frame_data["signs"][selected_indices, :]
            cur_weights = frame_data["weights"][selected_indices, :]
            cur_frame_id = torch.tensor(np.ones(frame_batchsize) * frame_id).long().reshape(-1, 1)

            points_frame.append(cur_points_frame)
            sdfs.append(cur_sdfs)
            sdfs_valid.append(cur_sdfs_valid)
            sdfs_sign.append(cur_sdfs_sign)
            weights.append(cur_weights)
            sample_frame_ids.append(cur_frame_id)

        points_frame = torch.cat(points_frame, dim=0)
        sdfs = torch.cat(sdfs, dim=0)
        sdfs_valid = torch.cat(sdfs_valid, dim=0)
        sdfs_sign = torch.cat(sdfs_sign, dim=0)
        weights = torch.cat(weights, dim=0)
        sample_frame_ids = torch.cat(sample_frame_ids, dim=0)

        input_dict = {
            'coords_frame': points_frame,
            'sample_frame_ids': sample_frame_ids,
            'weights': weights
        }
        gt_dict = {
            'sdf': sdfs,
            'sdf_valid': sdfs_valid,
            'sdf_signs': sdfs_sign,
        }
        return input_dict, gt_dict
    
    def __getitem__(self, index):
        return self.getitem_world(index)

    def get_bounding_box_of_all_frames(self):
        """
        Compute and return the bounding boxes (min/max in x, y, z) 
        for all frames in both world coordinates and their local frame coordinates.
        """
        if not self.frames_data:
            print("Warning: frames_data is empty. Did you call sample_frames()?")
            return None

        all_world_points = []
        all_frame_points = []
        for frame_data in self.frames_data:
            all_world_points.append(frame_data["points_world_gt"].numpy())
            all_frame_points.append(frame_data["points_frame"].numpy())

        all_world_points_np = np.concatenate(all_world_points, axis=0)
        all_frame_points_np = np.concatenate(all_frame_points, axis=0)

        xyz_min_world = all_world_points_np.min(axis=0)
        xyz_max_world = all_world_points_np.max(axis=0)
        xyz_min_frame = all_frame_points_np.min(axis=0)
        xyz_max_frame = all_frame_points_np.max(axis=0)

        return {
            "world": {
                "min": xyz_min_world,
                "max": xyz_max_world
            },
            "frame": {
                "min": xyz_min_frame,
                "max": xyz_max_frame
            }
        }