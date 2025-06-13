from torch.utils.data import Dataset
import torch
from grid_opt.utils.utils_sample import origin_dirs_W, ray_dirs_C
from grid_opt.utils.utils_sample import sample_pixels, get_batch_data, sample_along_rays
from grid_opt.utils.utils_sample import pointcloud_from_depth_torch, estimate_pointcloud_normals
import numpy as np
import cv2
from torchvision import transforms
cosSim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)


class ReplicaCAD(Dataset):

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        # dataset
        self.data_root = self.config['dataset']['path']
        self.online = self.config['dataset']['online']
        self.inv_depth_scale = 1. / self.config['dataset']['depth_scale']
        self.bounds_method = self.config['loss']['bounds_method']
        self.trunc_distance = self.config['loss']['trunc_distance']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Camera
        self.W = self.config["dataset"]["camera"]["w"]
        self.H = self.config["dataset"]["camera"]["h"]
        self.fx = self.config["dataset"]["camera"]["fx"]
        self.fy = self.config["dataset"]["camera"]["fy"]
        self.cx = self.config["dataset"]["camera"]["cx"]
        self.cy = self.config["dataset"]["camera"]["cy"]

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

        # load selected keyframe idx
        kframe_select_file = f"{self.data_root}/sample_idxs.txt"
        self.kframe_idxs = []
        with open(kframe_select_file, 'r') as f:
            for line in f:
                tmp = []
                line_data = line.strip().split()
                for item in line_data:
                    tmp.append(int(item))
                self.kframe_idxs.append(np.array(tmp))
        # load kframe data
        kframe_data_file = f"{self.data_root}/frame_data.pt"
        kframe_data = torch.load(kframe_data_file)
        self.depth_all = kframe_data['depth_batch']
        self.T_WC_all = kframe_data['T_WC_batch']
        self.norm_all = kframe_data['norm_batch']

        # load camera poses
        traj_file = self.data_root + "/traj.txt"
        self.Ts = np.loadtxt(traj_file).reshape(-1, 4, 4)

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
            self.trunc_distance,
            norm_sample,
            do_grad=True,
        )

        if norm_sample is not None:
            input_dict = {'coords': pc.reshape(-1, 3), 'normals': norm_sample.reshape(-1, 3)}
        else:
            input_dict = {'coords': pc.reshape(-1, 3)}
        gt_dict = {'sdf': bounds.reshape(-1, 1),
                   'grad_vec': grad_vec,
                   'sdf_valid': bounds.reshape(-1, 1) < 1e10}
        return input_dict, gt_dict

    def __len__(self):
        if self.online:
            return len(self.kframe_idxs)
        else:
            return (self.H * self.W) // self.n_rays
    

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



    
