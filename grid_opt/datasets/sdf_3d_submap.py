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


class SubmapSdf3D(Dataset):
    """A SDF dataset where the scene is decomposed into multiple submaps.
    Within each submap, multiple camera observations are simulated.
    """
    def __init__(self, meshfile, Nx, Ny, 
                 near_surface_n=2, free_space_n=1, submap_size=16, 
                 max_ray_dist=3.0, distance_std=0, trunc_dist=0.15,
                 submap_err_meter=0.0, submap_err_rad=0.0, 
                 submap_full_overlap=False):
        self.meshfile = meshfile
        self.mesh = o3d.io.read_triangle_mesh(self.meshfile)
        self.mesh_tri = trimesh.load(meshfile)
        self.sdf_fn = pysdf.SDF(self.mesh_tri.vertices, self.mesh_tri.faces)
        self.num_submaps = Nx * Ny
        self.submap_size = submap_size
        self.max_ray_dist = max_ray_dist
        self.near_surface_std = 0.05
        self.near_surface_n = near_surface_n
        self.free_space_n = free_space_n
        self.trunc_dist = trunc_dist
        self.frame_samples = 2**14
        self.submap_batchsize = 2**18
        self.distance_std = distance_std
        self.submap_err_meter = submap_err_meter
        self.submap_err_rad = submap_err_rad
        # Part I-a. partition AABB of mesh into multiple regions.
        bbox = self.mesh.get_axis_aligned_bounding_box()
        self.min_bound = np.array(bbox.min_bound)
        self.max_bound = np.array(bbox.max_bound)
        x_step = (self.max_bound[0] - self.min_bound[0]) / Nx
        y_step = (self.max_bound[1] - self.min_bound[1]) / Ny
        z_min, z_max = self.min_bound[2], self.max_bound[2]  # Z range is unchanged
        self.submap_bounds = []
        for i in range(Nx):
            for j in range(Ny):
                x_min = self.min_bound[0] + i * x_step
                x_max = x_min + x_step
                y_min = self.min_bound[1] + j * y_step
                y_max = y_min + y_step
                if submap_full_overlap:
                    submap_bound = self.inflated_global_bound(buffer=0.0)  # In debug mode, we make it simple where each submap is the entire scene.
                else:
                    submap_bound = np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])
                self.submap_bounds.append(submap_bound)
        # Part I-b. get ground truth pose for each submap
        self.R_world_submap_gt = utils_geometry.identity_rotations(self.num_submaps)  # N, 3, 3
        self.t_world_submap_gt = np.zeros((self.num_submaps, 3, 1))
        for submap_id in range(self.num_submaps):
            submap_bound = self.submap_bounds[submap_id]
            submap_center = np.mean(submap_bound, axis=1)
            self.t_world_submap_gt[submap_id, :, 0] = submap_center.flatten()
        self.t_world_submap_gt = torch.from_numpy(self.t_world_submap_gt).float()

        # Part II. Sample cameras within each submap
        self.submaps = []
        for submap_id in range(self.num_submaps):
            submap_bound = self.submap_bounds[submap_id]
            R_world_submap_gt = self.R_world_submap_gt[submap_id, :, :]
            t_world_submap_gt = self.t_world_submap_gt[submap_id, :, :]
            self.submaps.append(self.generate_submap(submap_bound, R_world_submap_gt, t_world_submap_gt))

        # Part III. sample noise perturbation on submap poses
        self.resample_poses()

        self.getitem_from_submaps = range(self.num_submaps)
    
    def getitem_from_single_submap(self, submap_id):
        """When calling __getitem__, will only sample observations from the specified submap.
        """
        self.getitem_from_submaps = [submap_id]
    
    def getitem_from_subset_of_submaps(self, submap_ids):
        """When calling __getitem__, will only sample observations from a subset of submaps.
        """
        self.getitem_from_submaps = submap_ids
    
    def resample_poses(self):
        # sample rotation and translation perturbations 
        translation_noises = utils_geometry.fixed_length_translations(self.num_submaps, length=self.submap_err_meter).unsqueeze(-1)  # N, 3, 1
        rotation_noises = utils_geometry.fixed_angle_rotations(self.num_submaps, rad=self.submap_err_rad)   # N, 3, 3
        # Do not perturb first submap pose
        translation_noises[0,:] = 0.0  
        rotation_noises[0,:,:] = torch.eye(3)
        # simulated noisy frame pose estimates
        self.t_world_submap = self.t_world_submap_gt + translation_noises  # (num_frames, 3, 1)
        self.R_world_submap = torch.matmul(self.R_world_submap_gt, rotation_noises)  # (num_frame, 3, 3)
        for submap_id in range(self.num_submaps):
            self.submaps[submap_id]['R_world_submap'] = self.R_world_submap[submap_id, :, :]
            self.submaps[submap_id]['t_world_submap'] = self.t_world_submap[submap_id, :, :]
    
    def inflated_global_bound(self, buffer=0.3):
        global_bound = np.zeros((3,2))
        global_bound[:,0] = self.min_bound - buffer
        global_bound[:,1] = self.max_bound + buffer
        return global_bound

    def ground_truth_submap_pose(self, submap_id):
        Rwm = self.R_world_submap_gt[submap_id, :, :]  # 3,3
        twm = self.t_world_submap_gt[submap_id, :, :]  # 3,1
        return Rwm, twm
    
    def noisy_submap_pose(self, submap_id):
        Rwm = self.R_world_submap[submap_id, :, :]  # 3,3
        twm = self.t_world_submap[submap_id, :, :]  # 3,1
        return Rwm, twm
    
    def generate_submap(self, submap_bound, R_world_submap_gt, t_world_submap_gt):
        # Part I. Sample GT frame centers and orientations, and record in the submap frame
        # For camera position, make sure it is properly in free space
        camera_position_bound = submap_bound
        R_world_frame_gt = utils_geometry.wrapped_gaussian_rotations(self.submap_size, std_rad=1.0)  # N, 3, 3
        t_world_frame_gt = np.zeros((self.submap_size, 3, 1))
        for frame_id in range(self.submap_size):
            while True:
                sample_position = utils_geometry.uniform_translations(1, camera_position_bound) # 1, 3
                sdf_at_position = self.sdf_fn(sample_position).astype(np.float32)[0]
                if sdf_at_position > 0.3 and sdf_at_position < 1e10:
                    break
            t_world_frame_gt[frame_id, :, 0] = sample_position.flatten()
        t_world_frame_gt = torch.from_numpy(t_world_frame_gt).float()
        # Get camera poses in submap frame
        R_submap_frame_gt, t_submap_frame_gt = utils_geometry.transform_poses_from(
            R_world_frame_gt, t_world_frame_gt, R_world_submap_gt, t_world_submap_gt
        )

        # Part II. Sample each camera frame
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))
        frames = []
        for frame_id in range(self.submap_size):
            twf_gt = t_world_frame_gt[frame_id, :, :]  # 3,1
            Rwf_gt = R_world_frame_gt[frame_id, :, :]  # 3,3
            tmf_gt = t_submap_frame_gt[frame_id, :, :] # 3,1
            Rmf_gt = R_submap_frame_gt[frame_id, :, :] # 3,3
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
            
            # Part II-a. Sample intersection points 
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
            # Only keep those that are within the maximum distance bound
            valid_dist_indices = np.nonzero(distances_gt < self.max_ray_dist)[0]
            points_hit = points_hit[valid_dist_indices, :]
            sdfs = sdfs[valid_dist_indices, :]
            sdfs_valid = sdfs_valid[valid_dist_indices, :]
            n_keep = points_hit.shape[0]
            logger.debug(f"Keeping {n_keep} points within distance threshold.")
            points_world_list.append(points_hit)
            sdfs_list.append(sdfs)
            sdfs_valid_list.append(sdfs_valid)
            signs_list.append(np.zeros((n_keep, 1)))
            
            # Part II-b. Perturb to get near surface samples
            if self.near_surface_n > 0:
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

            # Part II-c. Free space samples
            if self.free_space_n > 0:
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

            # Part II-d. Assemble all samples
            points_world_np = np.concatenate(points_world_list, axis=0)
            points_world = torch.from_numpy(points_world_np).float()
            points_frame = utils_geometry.transfrom_points_from(points_world, Rwf_gt, twf_gt)
            points_submap = utils_geometry.transform_points_to(points_frame, Rmf_gt, tmf_gt)
            sdfs_np = np.concatenate(sdfs_list, axis=0)
            sdfs_valid_np = np.concatenate(sdfs_valid_list, axis=0)
            signs_np = np.concatenate(signs_list, axis=0)
            frame_data = {
                'R_world_frame_gt': Rwf_gt,
                't_world_frame_gt': twf_gt,
                'R_submap_frame_gt': Rmf_gt,
                't_submap_frame_gt': tmf_gt,
                'points_frame': points_frame,
                'points_submap': points_submap,
                'sdfs': torch.from_numpy(sdfs_np).float(),
                'sdfs_valid': torch.from_numpy(sdfs_valid_np).float(),
                'signs': torch.from_numpy(signs_np).float()
            }
            frames.append(frame_data)

        submap_data = {
            'frames': frames,
            'world_frame_bound': submap_bound,
            'R_world_submap_gt': R_world_submap_gt,
            't_world_submap_gt': t_world_submap_gt
        }
        return submap_data

    def visualize_open3d(self):
        submap_frame_list = []
        submap_aabb_list = []
        submap_pcd_list = []
        for submap_id in range(self.num_submaps):
            # Visualize submap AABB
            submap_color = np.random.rand(3)
            submap_bound = self.submap_bounds[submap_id]
            aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=submap_bound[:,0], max_bound=submap_bound[:,1])
            aabb.color = submap_color
            submap_aabb_list.append(aabb)
            # Visualize all points in the submap
            points_world = []
            submap = self.submaps[submap_id]
            R_world_submap = submap['R_world_submap_gt']
            t_world_submap = submap['t_world_submap_gt']
            for frame in submap['frames']:
                points_world.append(
                    utils_geometry.transform_points_to(frame['points_submap'], R_world_submap, t_world_submap).numpy()
                )
                # Visualize the corresponding camera frame
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
                Twc = np.eye(4)
                Twc[:3, :3] = frame['R_world_frame_gt'].numpy()
                Twc[:3,  3] = frame['t_world_frame_gt'].numpy().flatten()
                coord.transform(Twc)
                coord.paint_uniform_color(submap_color)
                submap_frame_list.append(coord)
            points_world = np.concatenate(points_world, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_world)
            pcd.paint_uniform_color(submap_color)
            downpcd = pcd.voxel_down_sample(voxel_size=0.05)
            submap_pcd_list.append(downpcd)

        o3d.visualization.draw_geometries(submap_frame_list + submap_aabb_list + submap_pcd_list)

    
    def visualize_submap_open3d(self, submap_id):
        submap = self.submaps[submap_id]
        points_submap = [frame['points_submap'].numpy() for frame in submap['frames']]
        points_submap = np.concatenate(points_submap, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_submap)
        aabb = pcd.get_axis_aligned_bounding_box()
        aabb.color = [0, 0, 0]
        origin_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        o3d.visualization.draw_geometries([pcd, aabb, origin_coord])

    
    def compute_submap_local_bound(self, submap_id: int) -> torch.Tensor:
        submap = self.submaps[submap_id]
        points_submap = [frame['points_submap'].numpy() for frame in submap['frames']]
        points_submap = np.concatenate(points_submap, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_submap)
        aabb = pcd.get_axis_aligned_bounding_box()
        local_bound = np.zeros((3,2))
        local_bound[:,0] = aabb.min_bound
        local_bound[:,1] = aabb.max_bound
        return torch.from_numpy(local_bound).float()

    def __len__(self):
        # TODO (less priority): this is not accurate due to subsampling with batch size. 
        return self.num_submaps
    
    def __getitem__(self, index):
        # For each submap, sample submap_batch_size samples.
        submap_id = np.random.choice(self.getitem_from_submaps, size=1)[0]
        submap = self.submaps[submap_id]
        submap_frames = submap['frames']
        # Accumulate points from all frames in this submap
        points_list = [frame['points_submap'] for frame in submap_frames]
        sdfs_list = [frame['sdfs'] for frame in submap_frames]
        sdfs_valid_list = [frame['sdfs_valid'] for frame in submap_frames]
        signs_list = [frame['signs'] for frame in submap_frames]
        
        points_submap = torch.concat(points_list, dim=0)
        sdfs = torch.concat(sdfs_list, dim=0)
        sdfs_valid = torch.concat(sdfs_valid_list, dim=0)
        sdfs_sign = torch.concat(signs_list, dim=0)
        submap_size = points_submap.shape[0]
        batch_size = min(submap_size, self.submap_batchsize)
        selected_indices = np.random.choice(submap_size, size=batch_size, replace=False)

        input_dict = {
            'submap_index': torch.tensor(submap_id).long(),
            'coords_submap': points_submap[selected_indices, :],
            'R_world_submap': self.R_world_submap,
            't_world_submap': self.t_world_submap
        }

        gt_dict = {
            'sdf': sdfs[selected_indices, :],
            'sdf_valid': sdfs_valid[selected_indices, :],
            'sdf_signs': sdfs_sign[selected_indices, :],
            'R_world_submap_gt': self.R_world_submap_gt,   # include for debugging
            't_world_submap_gt': self.t_world_submap_gt    # include for debugging
        }
        return input_dict, gt_dict

