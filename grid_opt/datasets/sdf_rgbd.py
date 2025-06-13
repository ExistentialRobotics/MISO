from os.path import join
import os
import glob
from tqdm import tqdm
import numpy as np
import cv2
import torch
import open3d as o3d
from torchvision import transforms
from grid_opt.utils.utils_data import BGRtoRGB, DepthScale, DepthFilter, \
    pointcloud_from_depth_torch, estimate_pointcloud_normals
from grid_opt.utils.utils_sample import origin_dirs_W, ray_dirs_C, \
    sample_pixels, get_batch_data, sample_along_rays
import grid_opt.utils.utils as utils
import grid_opt.utils.utils_data as utils_data
import grid_opt.utils.utils_geometry as utils_geometry
from grid_opt.datasets.submap_dataset import SubmapDataset
cosSim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PosedSdfRgbd(SubmapDataset):
    """
    A dataset that generates SDF samples from RGB-D Sequence.
    """
    def __init__(
            self, 
            dataset_root: str,
            num_input_frames: int, 
            cam_params: utils_data.CameraParameters,
            min_depth=0.07,
            max_depth=12.0,
            voxel_size=None,
            n_rays=2**10,
            dist_behind_surf=0.1,
            n_strat_samples=3,
            n_surf_samples=4,
            trunc_dist=0.30,
            frame_downsample=1,
            device='cuda:0',
            use_clip=False
        ):
        super().__init__()
        self.dataset_root = dataset_root
        
        self._num_total = num_input_frames
        self._cam_params = cam_params
        self.min_depth = min_depth  # minimum depth for sampling
        self.max_depth = max_depth  # maximum depth for sampling
        self.voxel_size = voxel_size  # voxel downsampling size
        self.n_rays = n_rays  # number of rays to sample
        self.dist_behind_surf = dist_behind_surf  # distance behind surface for sampling
        self.n_strat_samples = n_strat_samples
        self.n_surf_samples = n_surf_samples
        self.trunc_dist = trunc_dist  # truncation distance for SDF
        self.frame_downsample = frame_downsample
        self.bounds_method = 'ray'
        self.normal_trunc_dist = 0.30  # iSDF normal based bounds (not used) 
        self.device = device
        self.use_clip = use_clip

        # 0) Setup
        self.setup()
        # 1) Read gt pose files
        self.load_gt_pose()
        # 2) Read ICP pose files
        self.load_icp_pose()
        # 2) Sample frames
        self.load_rgbd()
        # 3) Process CLIP features
        if use_clip: self.process_clip_features()

        self._selected_kfs = None
        self._getitem_cnt = 0
        logger.info((f"Constructed RGBD dataset from {self._num_frames}/{self._num_total} frames:\n"
                        f"Pixels per frame: {self.n_rays}, "
                        f"Distance behind surface: {self.dist_behind_surf}, "
                        f"Stratified samples: {self.n_strat_samples}, "
                        f"Near surface samples: {self.n_surf_samples}, "
                        f"Truncation distance: {self.trunc_dist}, "
                        f"Frame downsample: {self.frame_downsample}."))
    
    
    def __len__(self):
        return 1


    @property
    def num_kfs(self) -> int:
        return self._num_frames
    

    def sampled_points_at_kf(self, kf_id):
        self.select_keyframes([kf_id])
        model_input, _ = self.__getitem__(0)
        self.unselect_keyframes()
        return model_input['coords_frame']
    

    def get_odometry_at_pose(self, src_id):
        #FIXME: right now we are using GT pose to compute odometry
        # R_odom_src, t_odom_src = self.true_kf_pose_in_world(src_id)
        R_odom_src, t_odom_src = self.noisy_kf_pose_in_world(src_id)
        # print('Odometry source pose:', R_odom_src, t_odom_src)
        # print('Odometry source ICP pose:', R_odom_src_icp, t_odom_src_icp)
        T_odom_src = utils_geometry.pose_matrix(R_odom_src, t_odom_src)
        # R_odom_dst, t_odom_dst = self.true_kf_pose_in_world(src_id+1)
        R_odom_dst, t_odom_dst = self.noisy_kf_pose_in_world(src_id+1)
        T_odom_dst = utils_geometry.pose_matrix(R_odom_dst, t_odom_dst)
        T_src_dst = torch.linalg.inv(T_odom_src) @ T_odom_dst
        return T_src_dst

    
    def select_keyframes(self, kf_ids):
        self._selected_kfs = list(kf_ids)


    def unselect_keyframes(self):
        self._selected_kfs = None


    def true_kf_pose_in_world(self, kf_id):
        Rwk = self.R_world_frame_gt[kf_id, :, :]  # 3,3
        twk = self.t_world_frame_gt[kf_id, :, :]  # 3,1
        return Rwk, twk

    def noisy_kf_pose_in_world(self, kf_id):
        Rwk = self.R_world_frame[kf_id, :, :]  # 3,3
        twk = self.t_world_frame[kf_id, :, :]  # 3,1
        return Rwk, twk
    
    def setup(self):
        self.rgb_transform = transforms.Compose([BGRtoRGB()])
        self.depth_transform = transforms.Compose(
            [DepthScale(1.0/self._cam_params.depth_scale),
            DepthFilter(self.max_depth)]
        )
        self.dirs_C = ray_dirs_C(
            1, 
            self._cam_params.H, 
            self._cam_params.W, 
            self._cam_params.fx, 
            self._cam_params.fy, 
            self._cam_params.cx, 
            self._cam_params.cy, 
            self.device, 
            depth_type="z"
        )
    
    
    def load_gt_pose(self):
        self.pose_list_gt = []
        gt_pose_dir = join(self.dataset_root, 'frames', 'pose')
        pose_pths = sorted(glob.glob(join(gt_pose_dir, "*.txt")))
        for idx in range(0, self._num_total, self.frame_downsample):
            pose_filename = join(gt_pose_dir, f"{idx}.pose.txt")
            pose = np.loadtxt(pose_filename).reshape(4, 4)
            assert utils_geometry.check_numpy_pose_matrix(pose)
            R = pose[:3, :3]
            t = pose[:3, 3:]
            self.pose_list_gt.append((R, t))
        self._num_frames = len(self.pose_list_gt)
        self.R_world_frame_gt = torch.zeros((self._num_frames, 3, 3), dtype=torch.float32)
        self.t_world_frame_gt = torch.zeros((self._num_frames, 3, 1), dtype=torch.float32)
        for i in range(self._num_frames):
            R, t = self.pose_list_gt[i]
            self.R_world_frame_gt[i] = torch.from_numpy(R)
            self.t_world_frame_gt[i] = torch.from_numpy(t)

    def load_icp_pose(self):
        pose_file = join(self.dataset_root, 'poses_color_icp.txt')
        if (os.path.exists(pose_file) == False):
            logger.warning(f"ICP pose file {pose_file} does not exist. Using GT instead.")
            self.R_world_frame = self.R_world_frame_gt.clone()
            self.t_world_frame = self.t_world_frame_gt.clone()
            return
        T_list = utils_geometry.read_kitti_format_poses(pose_file)
        assert len(T_list) == self._num_frames
        self.R_world_frame = torch.zeros((self._num_frames, 3, 3), dtype=torch.float32)
        self.t_world_frame = torch.zeros((self._num_frames, 3, 1), dtype=torch.float32)
        for i in range(self._num_frames):
            T = T_list[i]
            R = T[:3, :3]
            t = T[:3, 3:]
            self.R_world_frame[i] = torch.from_numpy(R)
            self.t_world_frame[i] = torch.from_numpy(t)

    
    def load_rgbd(self):
        depth_dir = join(self.dataset_root, 'frames', 'depth')
        color_dir = join(self.dataset_root, 'frames', 'color')
        kf_id = 0
        depth_batch, T_WC_batch, norm_batch = [], [], []
        for index in tqdm(range(0, self._num_total, self.frame_downsample), desc="Loading RGB-D frames"):
            depth_filename = join(depth_dir, f"{index}.depth.pgm")
            color_filename = join(color_dir, f"{index}.color.jpg")
            depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)
            image = cv2.imread(color_filename)
            depth = self.depth_transform(depth)
            image = self.rgb_transform(image)
            # estimate normals        
            depth = torch.from_numpy(depth).float().cuda()
            pc = pointcloud_from_depth_torch(
                depth, self._cam_params.fx, self._cam_params.fy, self._cam_params.cx, self._cam_params.cy)
            normals = estimate_pointcloud_normals(pc)
            
            R, t = self.true_kf_pose_in_world(kf_id)
            T_WC = utils_geometry.pose_matrix(R, t).cuda()
            depth_batch.append(depth)
            T_WC_batch.append(T_WC)
            norm_batch.append(normals)
            
            kf_id += 1
        self._depth_batch = torch.stack(depth_batch)
        self._T_WC_batch = torch.stack(T_WC_batch)
        self._norm_batch = torch.stack(norm_batch)
    

    def sample_points(
        self,
        depth_batch,
        T_WC_batch,
        norm_batch=None,
        active_loss_approx=None,
    ):
        """
        Sample points by first sampling pixels, then sample depths along
        the backprojected rays.
        """
        n_rays = self.n_rays
        dist_behind_surf = self.dist_behind_surf
        n_strat_samples = self.n_strat_samples
        n_surf_samples = self.n_surf_samples

        n_frames = depth_batch.shape[0]
        if active_loss_approx is None:
            indices_b, indices_h, indices_w = sample_pixels(
                n_rays, n_frames, 
                self._cam_params.H, self._cam_params.W, 
                device=self.device
            )
        else:
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
        

    def process_clip_features(self):
        self.clip_data = []
        depth_dir = join(self.dataset_root, 'frames', 'depth')
        clip_dir = join(self.dataset_root, 'frames', 'clip', 'embeds')
        kf_id = 0
        for index in tqdm(range(0, self._num_total, self.frame_downsample), desc="Processing CLIP features"):
            clip_filename = join(clip_dir, f"{index}.color.emb")
            clip_dict = torch.load(clip_filename)
            clip_emb = clip_dict['pixel_embeds'].float()  # (Hc, Wc, 768), e.g., (24, 32, 768)
            clip_color_img = utils.feature_grid_to_rgb(clip_emb, normalize=False)  # (24, 32, 3)
            depth_filename = join(depth_dir, f"{index}.depth.pgm")
            depth_img = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)  # (Hd, Wd), e.g., (480, 640)
            depth = self.depth_transform(depth_img)
            depth = torch.from_numpy(depth).float()
            depth_pcd = pointcloud_from_depth_torch(
                depth,
                self._cam_params.fx,
                self._cam_params.fy,
                self._cam_params.cx,
                self._cam_params.cy
            )
            depth_pcd = depth_pcd.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            clip_pcd = torch.nn.functional.interpolate(
                depth_pcd, 
                size=(clip_emb.shape[0], clip_emb.shape[1]), 
                mode='nearest'
            )
            clip_pcd = clip_pcd.squeeze().permute(1, 2, 0).cpu()  # (24, 32, 3)
            
            clip_pts = clip_pcd.reshape(-1, 3)
            clip_emb = clip_emb.reshape(-1, 768)
            clip_colors = clip_color_img.reshape(-1, 3)
            assert clip_pts.shape[0] == clip_emb.shape[0], f"Clip points and embeddings do not match: {clip_pts.shape[0]} vs {clip_emb.shape[0]}"
            assert clip_pts.shape[0] == clip_colors.shape[0], f"Clip points and colors do not match: {clip_pts.shape[0]} vs {clip_colors.shape[0]}"
            # filter out zero depth points 
            norm = np.linalg.norm(clip_pts, axis=1, keepdims=False)
            clip_pts = clip_pts[norm > 1e-5, :]
            clip_emb = clip_emb[norm > 1e-5, :]
            clip_clr = clip_colors[norm > 1e-5, :]
            # visualize
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(clip_pts)
            # pcd.colors = o3d.utility.Vector3dVector(clip_clr)
            # o3d.visualization.draw_geometries([pcd])

            frame_data = {
                "points_frame": clip_pts,
                "embeddings": clip_emb, 
                "colors": clip_clr,
            }
            self.clip_data.append(frame_data)

    
    def getitem_clip(self, index):
        points_frame = []
        embeddings = []
        sample_frame_ids = []
        kf_list = range(self._num_frames)
        if self._selected_kfs is not None:
            kf_list = list(set(kf_list).intersection(self._selected_kfs))
        
        # Sample from each frame_data
        for frame_id in kf_list:
            frame_data = self.clip_data[frame_id]
            frame_size = frame_data["points_frame"].shape[0]
            cur_points_frame = frame_data["points_frame"]
            cur_embeddings = frame_data["embeddings"]
            cur_frame_id = torch.tensor(np.ones(frame_size) * frame_id).long().reshape(-1, 1)
            points_frame.append(cur_points_frame)
            embeddings.append(cur_embeddings)
            sample_frame_ids.append(cur_frame_id)
        
        points_frame = torch.cat(points_frame, dim=0)
        embeddings = torch.cat(embeddings, dim=0)
        sample_frame_ids = torch.cat(sample_frame_ids, dim=0)
        input_dict = {
            'clip_coords_frame': points_frame,
            'clip_sample_frame_ids': sample_frame_ids,
        }
        gt_dict = {
            'clip_embeddings': embeddings,
        }
        return input_dict, gt_dict

    
    def getitem_sdf(self, index):
        depth_batch = self._depth_batch
        T_WC_select = self._T_WC_batch
        norm_batch = self._norm_batch
        if self._selected_kfs is None:
            kframe_idxs = range(self._num_frames)
        else:
            kframe_idxs = self._selected_kfs               
        depth_batch = depth_batch[kframe_idxs]
        T_WC_select = T_WC_select[kframe_idxs]
        norm_batch = norm_batch[kframe_idxs]

        sample = self.sample_points(
            depth_batch, 
            T_WC_select, 
            norm_batch=norm_batch
        )
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
        logger.debug(f"indices_k {indices_k.shape}")

        # Convert pc from world to submap frames
        pc_world = pc.reshape(-1, 3)
        pc_kf = pc_world.clone()
        for kf_id in kframe_idxs:
            indices = torch.nonzero(indices_k == kf_id, as_tuple=False).squeeze(1)
            if indices.numel() == 0:
                continue
            Rwk, twk = self.true_kf_pose_in_world(kf_id)
            coords_world = pc_world[indices, :].to(Rwk)
            coords_kf = utils_geometry.transfrom_points_from(coords_world, Rwk, twk)
            pc_kf[indices, :] = coords_kf.to(pc_kf)

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
        # TODO: Extend the getitem output to include normals

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


    def __getitem__(self, index):
        input_dict, gt_dict = self.getitem_sdf(index)
        if self.use_clip:
            clip_input, clip_gt = self.getitem_clip(index)
            input_dict.update(clip_input)
            gt_dict.update(clip_gt)
        return input_dict, gt_dict
    

    def compute_scene_obb(self):
        model_input, gt = self.__getitem__(0)
        model_input, gt = utils.prepare_batch(model_input, gt, self.device)
        coords_frame = model_input['coords_frame']
        sample_frame_ids = model_input['sample_frame_ids'][:, 0]
        # Transform coords from keyframe to world frame
        unique_frame_ids = np.unique(sample_frame_ids.detach().cpu().numpy()).tolist()
        coords_world = coords_frame.clone()
        for kf_id in unique_frame_ids:
            idxs_select = torch.nonzero(sample_frame_ids == kf_id, as_tuple=False).squeeze(1)
            if idxs_select.numel() == 0: continue
            R_world_frame, t_world_frame = self.true_kf_pose_in_world(kf_id)
            coords_world[idxs_select, :] = utils_geometry.transform_points_to(
                coords_frame[idxs_select, :],
                R_world_frame.to(coords_world),
                t_world_frame.to(coords_world)
            )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords_world.cpu().numpy())
        pcd = pcd.voxel_down_sample(voxel_size=0.05)
        return pcd.get_minimal_oriented_bounding_box()





# =====================
#  Helper functions
# ===================== 

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