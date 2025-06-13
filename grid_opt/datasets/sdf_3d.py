import math
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
import pysdf
import open3d as o3d
import random
import grid_opt.utils.utils_geometry as utils_geometry

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Sdf3D(Dataset):
    """3D SDF dataset where one could access the GT SDF values everywhere (on surface, in free space, in occupied space).
    Currently implemented using pysdf, as done in torch_ngp.
    """
    def __init__(self, meshfile, batch_size=2**16, total_samples=2**20, 
                 normalize=False, surface_stddev=0.1, bound_buffer=0.5,
                 trunc_dist=None):
        super().__init__()
        self.meshfile = meshfile
        self.mesh = trimesh.load(meshfile)
        self.normalize = normalize
        if normalize:
            logger.info(f"Normalizing mesh to be between [-1,1].")
            # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
            vs = self.mesh.vertices
            vmin = vs.min(0)
            vmax = vs.max(0)
            v_center = (vmin + vmax) / 2
            v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
            vs = (vs - v_center[None, :]) * v_scale
            self.mesh.vertices = vs
            self.bound = np.asarray([[-1.,1.], [-1.,1.], [-1.,1.]])
            assert self.bound.shape == (3,2)
            self.surface_stddev = 0.01
            self.v_scale = v_scale
        else:
            min_bound = np.reshape(np.min(self.mesh.vertices, axis=0), (-1, 1))
            max_bound = np.reshape(np.max(self.mesh.vertices, axis=0), (-1, 1))
            self.bound = np.concatenate((min_bound-bound_buffer, max_bound+bound_buffer), axis=1)
            self.surface_stddev = surface_stddev
            logger.info(f"Using original mesh bound.")
        logger.debug(f"mesh: {self.mesh.vertices.shape} {self.mesh.faces.shape}")

        if not self.mesh.is_watertight:
            logger.warning(f"Mesh is not watertight! SDF maybe incorrect.")
            # print(f"Mesh bound: \n {self.bound}")

        self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.trunc_dist = trunc_dist
        assert self.total_samples % 8 == 0, "total_samples must be divisible by 8."
        self.resample()


    def __len__(self):
        return self.total_samples // self.batch_size + 1
    

    def _sample_uniformly(self, k):
        xs = np.reshape(np.random.uniform(self.bound[0,0], self.bound[0,1], k), (k,1))
        ys = np.reshape(np.random.uniform(self.bound[1,0], self.bound[1,1], k), (k,1))
        zs = np.reshape(np.random.uniform(self.bound[2,0], self.bound[2,1], k), (k,1))
        return np.concatenate([xs, ys, zs], axis=1)
    

    def resample(self):
        # online sampling
        sdfs = np.zeros((self.total_samples, 1))
        # surface
        points_surface = self.mesh.sample(self.total_samples * 7 // 8)
        # perturb surface
        points_surface[self.total_samples // 2:] += self.surface_stddev * np.random.randn(self.total_samples * 3 // 8, 3)
        # random
        points_uniform = self._sample_uniformly(self.total_samples // 8)
        points = np.concatenate([points_surface, points_uniform], axis=0).astype(np.float32)

        sdfs[self.total_samples // 2:] = self.sdf_fn(points[self.total_samples // 2:])[:,None].astype(np.float32)
        # Take care of invalid sdf samples
        sdfs_valid = np.abs(sdfs) < 1e10
        logger.info(f"Sampled {np.sum(sdfs_valid)} out of {self.total_samples} valid sdf values.")

        # Further clipping SDF based on distance to surface
        if self.trunc_dist is not None:
            logger.info(f"Truncating dataset at maximum distance: {self.trunc_dist}.")
            assert self.trunc_dist > 0
            sdfs_valid = np.abs(sdfs) < self.trunc_dist
            # For points in the clipped region (ie, too far from surface), preserve sign information
            sdf_signs = np.zeros_like(sdfs)
            pos_idxs = np.nonzero(np.logical_and(sdfs > self.trunc_dist, np.abs(sdfs) < 1e10))[0]
            neg_idxs = np.nonzero(np.logical_and(sdfs < -self.trunc_dist, np.abs(sdfs) < 1e10))[0]
            sdf_signs[pos_idxs, :] = 1
            sdf_signs[neg_idxs, :] = -1
        else:
            sdf_signs = np.zeros_like(sdfs)
        
        self.coords = torch.from_numpy(points).float()
        self.sdfs = torch.from_numpy(sdfs).float()
        self.sdf_valid = torch.from_numpy(sdfs_valid).float()
        self.sdf_signs = torch.from_numpy(sdf_signs).float()
    

    def __getitem__(self, idx):
        selected_indices = np.random.choice(self.total_samples, size=self.batch_size)
        input_dict = {'coords': self.coords[selected_indices, :]}
        gt_dict = {
            'sdf': self.sdfs[selected_indices, :],
            'sdf_valid': self.sdf_valid[selected_indices, :], 
            'sdf_sign': self.sdf_signs[selected_indices, :]
        }

        return input_dict, gt_dict

    
    def visualize_open3d(self, sdf_tol=1e-3, downsample_voxelsize=0.05):
        valid_indices = torch.where(self.sdf_valid == 1)[0]
        points_np = self.coords[valid_indices, :].detach().cpu().numpy()
        sdf_np = self.sdfs[valid_indices, :].detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        pos_idxs = np.nonzero(sdf_np > sdf_tol)[0]
        neg_idxs = np.nonzero(sdf_np < -sdf_tol)[0]
        sur_idxs = np.nonzero(np.abs(sdf_np) < sdf_tol)[0]
        colors_np = np.zeros_like(points_np)
        colors_np[pos_idxs, :] = [1, 0, 0]  # Positive SDF: Red
        colors_np[neg_idxs, :] = [0, 0, 1]  # Negative SDF: Blue
        colors_np[sur_idxs, :] = [0, 0, 0]  # Surface: Black
        pcd.colors = o3d.utility.Vector3dVector(colors_np)
        pcd_down = pcd.voxel_down_sample(voxel_size=downsample_voxelsize)
        pcd_list = [pcd_down]
        
        # Visualize sign supervision, if used
        if self.trunc_dist is not None:
            pos_sign_indices = torch.where(self.sdf_signs == 1)[0]
            pos_sign_points_np = self.coords[pos_sign_indices, :].detach().cpu().numpy()
            sign_pcd = o3d.geometry.PointCloud()
            sign_pcd.points = o3d.utility.Vector3dVector(pos_sign_points_np)
            pos_sign_colors_np = np.zeros_like(pos_sign_points_np)
            pos_sign_colors_np[:, :] = [0, 1, 0]   # Points with positive sign: Green
            sign_pcd.colors = o3d.utility.Vector3dVector(pos_sign_colors_np)
            sign_pcd_down = sign_pcd.voxel_down_sample(voxel_size=downsample_voxelsize)
            pcd_list.append(sign_pcd_down)

        o3d.visualization.draw_geometries(pcd_list)

    def get_inflated_bound(self):
        assert not self.normalize
        return self.bound
    

class PosedSdf3D(Dataset):
    """A SDF dataset obtained by simulating random camera views.
    """
    def __init__(self, meshfile, frame_batchsize=2**14, frame_samples=2**14, num_frames=64, 
                 near_surface_n=2, near_surface_std=0.05, free_space_n=1, trunc_dist=0.15,
                 frame_std_rad=0.0, frame_std_meter=0.0, distance_std=0.0):
        super().__init__()
        self.meshfile = meshfile
        self.mesh = trimesh.load(meshfile)
        min_bound = np.reshape(np.min(self.mesh.vertices, axis=0), (-1, 1))
        max_bound = np.reshape(np.max(self.mesh.vertices, axis=0), (-1, 1))
        self.bound = np.concatenate((min_bound, max_bound), axis=1)
        self.near_surface_std = near_surface_std
        self.near_surface_n = near_surface_n
        self.free_space_n = free_space_n
        self.trunc_dist = trunc_dist
        logger.debug(f"Using original mesh bound: \n {self.bound}")
        logger.debug(f"mesh: {self.mesh.vertices.shape} {self.mesh.faces.shape}")
        self.frame_std_rad = frame_std_rad
        self.frame_std_meter = frame_std_meter
        self.distance_std = distance_std

        if not self.mesh.is_watertight:
            logger.warning(f"{meshfile}: Mesh is not watertight! SDF maybe incorrect.")
            # print(f"Mesh bound: \n {self.bound}")

        self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)
        self.num_frames = num_frames
        self.frame_samples = frame_samples
        self.frame_batchsize = frame_batchsize

        # sample GT frame centers and orientations
        camera_position_bound = self.bound 
        self.R_world_frame_gt = utils_geometry.wrapped_gaussian_rotations(self.num_frames, std_rad=1.0)  # N, 3, 3
        self.t_world_frame_gt = np.zeros((self.num_frames, 3, 1))
        # For camera position, make sure it is properly in free space
        for frame_id in range(self.num_frames):
            while True:
                sample_position = utils_geometry.uniform_translations(1, camera_position_bound) # 1, 3
                sdf_at_position = self.sdf_fn(sample_position).astype(np.float32)[0]
                if sdf_at_position > 0.3 and sdf_at_position < 1e10:
                    break
            self.t_world_frame_gt[frame_id, :, 0] = sample_position.flatten()
        self.t_world_frame_gt = torch.from_numpy(self.t_world_frame_gt).float()
        self.sample_frames()
        # Resample noisy pose estimates
        self.resample_poses()

    def __len__(self):
        augment_factor = 1 + self.free_space_n + self.near_surface_n
        return augment_factor * self.frame_samples // self.frame_batchsize
    
    def sample_frames(self):
        """Use the ray casting function in open3d to simulate camera views.
        """
        mesh = o3d.io.read_triangle_mesh(self.meshfile)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        # Generate local observations
        self.frames = []
        for frame_id in range(self.num_frames):
            twf_gt = self.t_world_frame_gt[frame_id, :, :]  # 3,1
            Rwf_gt = self.R_world_frame_gt[frame_id, :, :]  # 3,3
            twc = twf_gt.numpy().flatten()
            Rwc = Rwf_gt.numpy()
            eye = twc
            forward = -Rwc[:, 2]  # Negative z-axis in world coordinates
            center = eye + forward
            up = Rwc[:, 1]        # Y-axis in world coordinates
            rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                fov_deg=90,
                center=center,
                eye=eye,
                up=up,
                width_px=640,
                height_px=480,
            )
            ans = scene.cast_rays(rays)
            
            # Prepare for sampling
            points_world_list = []
            sdfs_list = []
            sdfs_valid_list = []
            signs_list = []

            # Part I. Sample intersection points 
            # Source: https://www.open3d.org/docs/release/tutorial/geometry/ray_casting.html
            hit = ans['t_hit'].isfinite()
            points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
            points_np = points.numpy()
            n_hit = points_np.shape[0]
            assert n_hit > 0, f"Frame {frame_id} has no hit point!"
            n_keep = min(self.frame_samples, n_hit)
            perm_indices = np.random.permutation(n_hit)
            points_hit_gt = points_np[perm_indices[:n_keep], :]
            sdfs = self.sdf_fn(points_hit_gt)[:,None].astype(np.float32)
            sdfs_valid = np.abs(sdfs) < 1e10
            # Simulate Gaussian noise to the distance
            distances_gt = np.linalg.norm(points_hit_gt - twc, axis=1, keepdims=True)  # N, 1
            distances_gt = np.where(distances_gt > 1e-6, distances_gt, 1e-6 * np.ones_like(distances_gt))
            directions_gt = (points_hit_gt - twc) / distances_gt  # N, 3
            distances_perturbed = distances_gt + np.random.randn(distances_gt.shape[0], 1) * self.distance_std
            points_hit = twc + directions_gt * distances_perturbed
            points_world_list.append(points_hit)
            sdfs_list.append(sdfs)
            sdfs_valid_list.append(sdfs_valid)
            signs_list.append(np.zeros((n_keep, 1)))

            # Part II. Perturb to get near surface samples
            n_surf = points_hit.shape[0]
            distances = np.linalg.norm(points_hit - twc, axis=1, keepdims=True)  # N, 1
            distances = np.where(distances > 1e-6, distances, 1e-6 * np.ones_like(distances))
            directions = (points_hit - twc) / distances  # N, 3
            repeated_dist = np.repeat(distances, self.near_surface_n, axis=0)  # 2N, 1
            repeated_dir  = np.repeat(directions, self.near_surface_n, axis=0)  # 2N, 3
            displacement = np.random.randn(n_surf * self.near_surface_n, 1) * self.near_surface_std  # 2N, 1
            near_surf_points = twc + repeated_dir * (repeated_dist + displacement)
            points_world_list.append(near_surf_points)
            sdfs_list.append(-displacement)
            sdfs_valid_list.append(np.ones((n_surf * self.near_surface_n, 1)))
            signs_list.append(np.zeros((n_surf * self.near_surface_n, 1)))
            
            # Part III. Free space samples
            min_dist_ratio = 0.01
            max_dist_ratio = 0.99
            repeated_dist = np.repeat(distances, self.free_space_n, axis=0)  # N, 1
            repeated_dir  = np.repeat(directions, self.free_space_n, axis=0)  # N, 3
            dist_ratio = min_dist_ratio + np.random.rand(n_surf * self.free_space_n, 1) * (max_dist_ratio - min_dist_ratio)
            displacement = (dist_ratio - 1.0) * repeated_dist
            displacement = np.minimum(displacement, -self.trunc_dist)  # make sure sample outside of truncation region
            free_space_coords = twc + repeated_dir * (repeated_dist + displacement)
            points_world_list.append(free_space_coords)
            sdfs_list.append(-displacement)
            sdfs_valid_list.append(np.zeros((n_surf * self.free_space_n, 1)))
            signs_list.append(np.ones((n_surf * self.free_space_n, 1)))

            # Part IV. Assemble all samples
            points_world_np = np.concatenate(points_world_list, axis=0)
            points_world = torch.from_numpy(points_world_np).float()
            points_frame = utils_geometry.transfrom_points_from(points_world, Rwf_gt, twf_gt)
            sdfs_np = np.concatenate(sdfs_list, axis=0)
            sdfs_valid_np = np.concatenate(sdfs_valid_list, axis=0)
            signs_np = np.concatenate(signs_list, axis=0)
            frame_data = {
                'R_world_frame_gt': Rwf_gt,
                't_world_frame_gt': twf_gt,
                # 'R_world_frame': R_world_frame,  # Generated by the self.resample_poses()
                # 't_world_frame': t_world_frame,
                'points_frame': points_frame,
                'sdfs': torch.from_numpy(sdfs_np).float(),
                'sdfs_valid': torch.from_numpy(sdfs_valid_np).float(),
                'signs': torch.from_numpy(signs_np).float()
            }
            # for key, val in frame_data.items():
            #     print(f"{key}: {val.shape}") 
            self.frames.append(frame_data)

    def resample_poses(self):
        # sample rotation and translation perturbations 
        translation_noises = utils_geometry.gaussian_translations(self.num_frames, stddev=self.frame_std_meter).unsqueeze(-1)  # N, 3, 1
        rotation_noises = utils_geometry.wrapped_gaussian_rotations(self.num_frames, std_rad=self.frame_std_rad)   # N, 3, 3
        # Do not perturb first pose
        translation_noises[0,:] = 0.0  
        rotation_noises[0,:,:] = torch.eye(3)
        # simulated noisy frame pose estimates
        self.t_world_frame = self.t_world_frame_gt + translation_noises  # (num_frames, 3, 1)
        self.R_world_frame = torch.matmul(self.R_world_frame_gt, rotation_noises)  # (num_frame, 3, 3)
        for frame_id in range(self.num_frames):
            self.frames[frame_id]['R_world_frame'] = self.R_world_frame[frame_id, :, :]
            self.frames[frame_id]['t_world_frame'] = self.t_world_frame[frame_id, :, :]

    def true_kf_pose_in_world(self, kf_id):
        Rwk = self.R_world_frame_gt[kf_id, :, :]  # 3,3
        twk = self.t_world_frame_gt[kf_id, :, :]  # 3,1
        return Rwk, twk
    
    def noisy_kf_pose_in_world(self, kf_id):
        Rwk = self.R_world_frame[kf_id, :, :]
        twk = self.t_world_frame[kf_id, :, :]
        return Rwk, twk

    def __getitem__(self, index):
        # For each frame, sample frame_batch_size samples
        points_frame = []
        points_world_noisy = []
        points_world_gt = []
        sdfs = []
        sdfs_valid = []
        sdfs_sign = []
        frame_indices = []
        frame_begin_index = 0
        for frame_id in range(self.num_frames):
            frame_data = self.frames[frame_id]
            frame_size = frame_data['points_frame'].shape[0]
            frame_batchsize = min(self.frame_batchsize, frame_size)
            selected_indices = np.random.choice(frame_size, size=frame_batchsize)
            frame_end_index = frame_begin_index + frame_batchsize
            points_frame.append(frame_data['points_frame'][selected_indices, :])
            sdfs.append(frame_data['sdfs'][selected_indices, :])
            sdfs_valid.append(frame_data['sdfs_valid'][selected_indices, :])
            sdfs_sign.append(frame_data['signs'][selected_indices, :])
            frame_indices.append(torch.tensor([frame_begin_index, frame_end_index]).long().reshape(1, 2))
            frame_begin_index = frame_end_index
            points_world_noisy.append(utils_geometry.transform_points_to(points_frame[-1], self.R_world_frame[frame_id, :, :], self.t_world_frame[frame_id, :, :]))
            points_world_gt.append(utils_geometry.transform_points_to(points_frame[-1], self.R_world_frame_gt[frame_id, :, :], self.t_world_frame_gt[frame_id, :, :]))

        input_dict = {
            'R_world_frame': self.R_world_frame,
            't_world_frame': self.t_world_frame,
            'coords_frame': torch.concat(points_frame, dim=0),
            'coords_world_noisy': torch.concat(points_world_noisy, dim=0),
            'coords_world_gt': torch.concat(points_world_gt, dim=0),
            'frame_indices': torch.concat(frame_indices, dim=0)
        }
        gt_dict = {
            'sdf': torch.concat(sdfs, dim=0),
            'sdf_valid': torch.concat(sdfs_valid, dim=0),
            'sdf_signs': torch.concat(sdfs_sign, dim=0),
            'R_world_frame_gt': self.R_world_frame_gt,   # Include for debugging purpose
            't_world_frame_gt': self.t_world_frame_gt
        }
        return input_dict, gt_dict

    def get_inflated_bound(self):
        min_bound = self.bound[:,0].reshape(-1,1)
        max_bound = self.bound[:,1].reshape(-1,1)
        return np.concatenate((min_bound-0.5, max_bound+0.5), axis=1)

    def visualize_open3d(self):
        sdf_tol = 1e-6
        mesh = o3d.io.read_triangle_mesh(self.meshfile)
        pcd_list = []
        coord_list = []
        for i, frame_data in enumerate(self.frames):
            gt_sdf = frame_data['sdfs'].numpy()
            gt_sign = frame_data['signs'].numpy()
            pcd = o3d.geometry.PointCloud()
            # Recover points in world frame
            points_world = utils_geometry.transform_points_to(frame_data['points_frame'], frame_data['R_world_frame'], frame_data['t_world_frame']).numpy()
            pcd.points = o3d.utility.Vector3dVector(points_world)
            # Color points based on SDF labels
            pcd.paint_uniform_color([0,0,0])
            colors = np.asarray(pcd.colors)
            colors[np.nonzero(gt_sdf > sdf_tol)[0]] = [1, 0, 0]  # Positive SDF as red
            colors[np.nonzero(gt_sdf < -sdf_tol)[0]] = [0, 0, 1]  # Negative SDF as blue
            colors[np.nonzero(gt_sign == 1)[0]] = [0, 1, 0]  # Free space samples as green
            pcd.colors = o3d.utility.Vector3dVector(colors)
            downpcd = pcd.voxel_down_sample(voxel_size=0.01)
            pcd_list.append(downpcd)
            
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            Twc = np.eye(4)
            Twc[:3, :3] = frame_data['R_world_frame_gt'].numpy()
            Twc[:3,  3] = frame_data['t_world_frame_gt'].numpy().flatten()
            coord.transform(Twc)
            coord_list.append(coord)
        
        o3d.visualization.draw_geometries(pcd_list + coord_list + [mesh])


class BatchedSdf3D(Dataset):
    """A collection of multiple 3d SDF datasets.
    """
    def __init__(self, filelist, batch_size=2**16, total_samples=2**20, normalize=False, surface_stddev=0.1, trunc_dist=None):
        super().__init__()
        self.num_datasets = len(filelist)
        self.len = 0
        self.datasets = []
        for dataset_idx in range(self.num_datasets):
            dataset = Sdf3D(
                meshfile=filelist[dataset_idx],
                batch_size=batch_size,
                total_samples=total_samples,
                normalize=normalize,
                surface_stddev=surface_stddev,
                trunc_dist=trunc_dist
            )
            self.len += len(dataset)
            self.datasets.append(dataset)
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        dataset_index = np.random.choice(self.num_datasets, size=1)[0]
        dataset = self.datasets[dataset_index]
        inner_index = np.random.choice(len(dataset), size=1)[0]
        input_dict, gt_dict = dataset[inner_index]
        input_dict['dataset_index'] = torch.tensor(dataset_index).long()
        return input_dict, gt_dict


class BatchPosedSdf3D(Dataset):
    def __init__(self, filelist, frame_batchsize=2**14, frame_samples=2**14, num_frames=64, 
                 near_surface_n=2, near_surface_std=0.05, free_space_n=1, trunc_dist=0.15,
                 frame_std_rad=0.0, frame_std_meter=0.0, distance_std=0.0, resample_poses_freq=None):
        super().__init__()
        self.num_datasets = len(filelist)
        self.len = 0
        self.datasets = []
        for dataset_idx in range(self.num_datasets):
            dataset = PosedSdf3D(
                meshfile=filelist[dataset_idx],
                frame_batchsize=frame_batchsize,
                frame_samples=frame_samples,
                num_frames=num_frames,
                near_surface_n=near_surface_n,
                near_surface_std=near_surface_std,
                free_space_n=free_space_n,
                trunc_dist=trunc_dist,
                frame_std_rad=frame_std_rad,
                frame_std_meter=frame_std_meter,
                distance_std=distance_std
            )
            # assert len(dataset) == 1, f"Expect len(dataset)=1, get {len(dataset)}"
            self.len += len(dataset)
            self.datasets.append(dataset)
        self.getitem_count = 0
        self.resample_poses_freq = resample_poses_freq
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        self.getitem_count += 1
        if self.resample_poses_freq is not None and self.getitem_count % self.resample_poses_freq == 0:
            self.resample_poses()
        dataset_index = np.random.choice(self.num_datasets, size=1)[0]
        dataset = self.datasets[dataset_index]
        inner_index = np.random.choice(len(dataset), size=1)[0]
        input_dict, gt_dict = dataset[inner_index]
        input_dict['dataset_index'] = torch.tensor(dataset_index).long()
        input_dict['dataset_bound'] = torch.from_numpy(dataset.get_inflated_bound()).float()
        return input_dict, gt_dict
    
    def resample_poses(self):
        logger.info(f"Resample noisy poses!")
        for dataset in self.datasets:
            dataset.resample_poses()