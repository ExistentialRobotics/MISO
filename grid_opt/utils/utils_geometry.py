import math
import numpy as np
import torch
from pytorch3d.transforms import so3_exp_map, so3_relative_angle, matrix_to_axis_angle
from .utils import normalize_last_dim
from typing import List
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def coords_in_bound(coords: torch.Tensor, bound: torch.Tensor):
    """Test if input coords are within specified bound.

    Args:
        coords (torch.Tensor): N, 3 set of 3D points
        bound (torch.Tensor): 3, 2 tensor where rows specify min, max bound for the x, y, z dimensions.

    Returns:
        torch.Tensor: N, 1 boolean mask
    """
    inside_min = coords >= bound[:, 0]  # Points greater than or equal to min bounds
    inside_max = coords <= bound[:, 1]  # Points less than or equal to max bounds
    mask = (inside_min & inside_max).all(dim=1).unsqueeze(1)
    N = mask.shape[0]
    num_true = torch.count_nonzero(mask)
    logger.debug(f"{num_true} / {N} points are in bound.")
    return mask

def batch_transform_to_world_frame(
    coords_frame: torch.Tensor,
    frame_indices: torch.Tensor,
    R_world_frame_input: torch.Tensor,     # num_frames, 3, 3
    t_world_frame_input: torch.Tensor,     # num_frames, 3, 1
    rotation_corrections: torch.Tensor,    # num_frames, 3
    translation_corrections: torch.Tensor  # num_frames, 3, 1
):
    num_frames = R_world_frame_input.shape[0]
    assert t_world_frame_input.shape[0] == num_frames
    assert rotation_corrections.shape[0] == num_frames
    assert translation_corrections.shape[0] == num_frames
    # Obtain corrected camera poses
    R_corr = so3_exp_map(rotation_corrections)
    R_corr[0, :, :] = torch.eye(3)  # First pose is anchored 
    t_corr = translation_corrections.clone()
    t_corr[0, :, :] = 0.0
    R_world_frame = torch.matmul(R_world_frame_input, R_corr)
    t_world_frame = t_world_frame_input + t_corr
    # Obtain query coordinates in the world frame
    coords_world = []
    for frame_id in range(num_frames):
        idx_begin = frame_indices[frame_id, 0]
        idx_end = frame_indices[frame_id, 1]
        coords_world.append(transform_points_to(
            coords_frame[idx_begin:idx_end, :],
            R_world_frame[frame_id, :, :],
            t_world_frame[frame_id, :, :]
        ))
    coords_world = torch.concat(coords_world, dim=0)
    return coords_world

def pose_matrix(R: torch.Tensor, t: torch.Tensor):
    """Convert rotation and translation to 4x4 pose matrix.

    Args:
        R (torch.Tensor): 3, 3 rotation matrix
        t (torch.Tensor): 3, 1 translation vector

    Returns:
        torch.Tensor: 4, 4 pose matrix
    """
    assert R.shape == (3,3)
    assert t.shape == (3,1)
    pose = torch.eye(4).to(R)
    pose[:3, :3] = R
    pose[:3, [3]] = t
    return pose

def apply_pose_correction(
        R: torch.Tensor,                # 3, 3
        t: torch.Tensor,                # 3, 1
        R_delta: torch.Tensor,          # 1, 3
        t_delta: torch.Tensor,          # 3, 1
    ):
    """Apply pose perturbation.

    Args:
        R (torch.Tensor): Input rotation 3,3
        t (torch.Tensor): Input translation 3,1
        R_delta (torch.Tensor): delta R 1,3
        t_delta (torch.Tensor): delta t 3,1

    Returns:
        R 3,3 ; t 3,1
    """
    assert R.shape == (3,3)
    assert t.shape == (3,1)
    assert R_delta.shape == (1,3)
    assert t_delta.shape == (3,1)
    return torch.matmul(R, so3_exp_map(R_delta)[0]), t + t_delta

def get_pose_correction(
        R: torch.Tensor,             # 3, 3
        t: torch.Tensor,             # 3, 1
        Rnew: torch.Tensor,          # 3, 3
        tnew: torch.Tensor,          # 3, 1
    ):
    """Get pose perturbation to go from R, t to Rnew, tnew,
    i.e., Rnew = R Exp(R_delta), tnew = t + t_delta.
    Args:
        R (torch.Tensor): Input rotation 3,3
        t (torch.Tensor): Input translation 3,1
        Rnew (torch.Tensor): New rotation 3,3
        tnew (torch.Tensor): New translation 3,1
    return R_delta 1,3 ; t_delta 3,1
    """
    t_delta = tnew - t
    R_delta = (R.T @ Rnew).unsqueeze(0)
    R_delta = matrix_to_axis_angle(R_delta)
    return R_delta, t_delta

def uniform_translations(k, bound):
    xs = np.reshape(np.random.uniform(bound[0,0], bound[0,1], k), (k,1))
    ys = np.reshape(np.random.uniform(bound[1,0], bound[1,1], k), (k,1))
    zs = np.reshape(np.random.uniform(bound[2,0], bound[2,1], k), (k,1))
    return torch.from_numpy(np.concatenate([xs, ys, zs], axis=1)).float()

def gaussian_translations(k, stddev):
    return torch.from_numpy(np.random.normal(size=(k, 3)) * stddev).float()

def fixed_length_translations(k, length):
    directions = torch.from_numpy(np.random.normal(size=(k, 3))).float()
    directions = normalize_last_dim(directions)
    return length * directions

def identity_rotations(n):
    R = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(n, 1, 1)
    assert R.shape == (n, 3, 3)
    return R

def wrapped_gaussian_rotations(n, std_rad=0.1):
    """Generate n 3D rotations according to zero-mean wrapped Gaussian distribution.
    Returns:
        torch.Tensor: N, 3, 3 tensor of N rotations
    """
    tangent_vectors = torch.from_numpy(np.random.normal(size=(n,3)) * std_rad).float()
    return so3_exp_map(tangent_vectors)

def fixed_angle_rotations(n, rad):
    axis = torch.from_numpy(np.random.normal(size=(n, 3))).float()
    axis = normalize_last_dim(axis)
    axis_angle = rad * axis
    return so3_exp_map(axis_angle)

def chordal_to_radian(d):
    return 2 * np.arcsin(d/(2*np.sqrt(2)))

def chordal_to_degree(d):
    return math.degrees(chordal_to_radian(d))

def rotation_rmse(R1, R2):
    """Compute root mean squared error in DEGREE between two sets of rotations.
    For now, assume R1 and R2 are aligned in the same global frame.
    TODO: option to align R1 and R2 before computing distance.

    Args:
        R1 (Tensor): N, 3, 3 
        R2 (Tensor): N, 3, 3
    """
    angles_cos = so3_relative_angle(R1, R2, cos_bound=1e-12, cos_angle=True).detach().cpu().numpy()  # (N, )
    angles_rad = np.arccos(np.clip(angles_cos, -1, 1))
    rmse_rad = np.sqrt(np.mean(angles_rad**2))
    rmse_deg = math.degrees(rmse_rad)
    return rmse_deg

def rotation_mean_error(R1, R2):
    """Compute mean angular error in DEGREE between two sets of rotations.
    For now, assume R1 and R2 are aligned in the same global frame.
    TODO: option to align R1 and R2 before computing distance.

    Args:
        R1 (Tensor): N, 3, 3 
        R2 (Tensor): N, 3, 3
    """
    angles_cos = so3_relative_angle(R1, R2, cos_bound=1e-12, cos_angle=True).detach().cpu().numpy()  # (N, )
    angles_rad = np.arccos(np.clip(angles_cos, -1, 1))
    err_rad = np.mean(np.abs(angles_rad))
    err_deg = math.degrees(err_rad)
    return err_deg

def translation_rmse(t1, t2):
    """Compute root mean squared error between two sets of translations.
    For now, assume t1 and t2 are aligned in the same global frame.
    TODO: option to align t1 and t2 before computing distance.

    Args:
        t1 (Tensor): N, 3, 1
        t2 (Tensor): N, 3, 1
    """
    dists = torch.linalg.vector_norm(t1.squeeze() - t2.squeeze(), dim=1)
    return torch.sqrt(torch.mean(dists**2)).item()

def translation_mean_error(t1, t2):
    """Compute mean error between two sets of translations.
    For now, assume t1 and t2 are aligned in the same global frame.
    TODO: option to align t1 and t2 before computing distance.

    Args:
        t1 (Tensor): N, 3, 1
        t2 (Tensor): N, 3, 1
    """
    dists = torch.linalg.vector_norm(t1.squeeze() - t2.squeeze(), dim=1)
    return torch.mean(torch.abs(dists)).item()
    
def transform_points_to(points_src, R_dst_src, t_dst_src):
    """
    Args:
        points_src (torch.Tensor): N, 3 tensor in src frame. Each row is a point.
        R_dst_src (torch.Tensor): 3, 3 rotation from src to dst
        t_dst_src (torch.Tensor): 3, 1 translation from src to dst
    Output: 
        points_dst (Torch.Tensor): N, 3 tensor in dst frame
    """
    assert R_dst_src.shape == (3,3)
    assert t_dst_src.shape == (3,1)
    return points_src @ (R_dst_src.T) + t_dst_src.T

def transfrom_points_from(points_dst, R_dst_src, t_dst_src):
    """
    Args:
        points_dst (torch.Tensor): N, 3 tensor in dst frame. Each row is a point.
        R_dst_src (torch.Tensor): 3, 3 rotation from src to dst
        t_dst_src (torch.Tensor): 3, 1 translation from src to dst
    Output: 
        points_src (Torch.Tensor): N, 3 tensor in src frame
    """
    assert R_dst_src.shape == (3,3)
    assert t_dst_src.shape == (3,1)
    R_src_dst = R_dst_src.T
    t_src_dst = - R_dst_src.T @ t_dst_src
    return transform_points_to(points_dst, R_src_dst, t_src_dst)

def transform_poses_to(R_src_frames, t_src_frames, R_dst_src, t_dst_src):
    """ 
    Transform a collection of poses (separately represented as rotations and translations)
    to a new frame.
    Args:
        R_src_frames (torch.Tensor): N, 3, 3 rotations in src frame.
        t_src_frames (torch.Tensor): N, 3, 1 translations in src frame.
        R_dst_src (torch.Tensor): 3, 3 rotation from src to dst
        t_dst_src (torch.Tensor): 3, 1 translation from src to dst
    Output: 
        R_dst_frames (torch.Tensor): N, 3, 3 rotations in dst frame.
        t_dst_frames (torch.Tensor): N, 3, 1 translations in dst frame.
    """
    assert R_src_frames.ndim == 3 or R_src_frames.ndim == 2
    assert t_src_frames.ndim == 3 or t_src_frames.ndim == 2
    assert R_dst_src.shape == (3,3)
    assert t_dst_src.shape == (3,1)
    R_dst_frames = torch.matmul(R_dst_src, R_src_frames)
    t_dst_frames = torch.matmul(R_dst_src, t_src_frames) + t_dst_src
    return R_dst_frames, t_dst_frames

def transform_poses_from(R_dst_frames, t_dst_frames, R_dst_src, t_dst_src):
    """ 
    Args:
        R_dst_frames (torch.Tensor): N, 3, 3 rotations in dst frame.
        t_dst_frames (torch.Tensor): N, 3, 1 translations in dst frame.
        R_dst_src (torch.Tensor): 3, 3 rotation from src to dst
        t_dst_src (torch.Tensor): 3, 1 translation from src to dst
    Output: 
        R_src_frames (torch.Tensor): N, 3, 3 rotations in src frame.
        t_src_frames (torch.Tensor): N, 3, 1 translations in src frame.
    """
    assert R_dst_src.shape == (3,3)
    assert t_dst_src.shape == (3,1)
    R_src_dst = R_dst_src.T
    t_src_dst = - R_dst_src.T @ t_dst_src
    return transform_poses_to(R_dst_frames, t_dst_frames, R_src_dst, t_src_dst)

def aabb_torch(points: torch.Tensor, buffer: float = 0.0):
    """
    Compute the axis-aligned bounding box of a set of points.
    Args:
        points (torch.Tensor): N, 3 tensor of points
    Returns:
        torch.Tensor: 3, 2 tensor where rows specify min, max bound for the x, y, z dimensions.
    """
    min_bound = points.min(dim=0)[0]
    max_bound = points.max(dim=0)[0]
    return torch.stack([min_bound-buffer, max_bound+buffer], dim=0).T

def voxel_down_sample_torch(points: torch.tensor, voxel_size: float):
    """
        voxel based downsampling. Returns the indices of the points which are closest to the voxel centers.
    Args:
        points (torch.Tensor): [N,3] point coordinates
        voxel_size (float): grid resolution

    Returns:
        indices (torch.Tensor): [M] indices of the original point cloud, downsampled point cloud would be `points[indices]`

    Reference: Louis Wiesmann
    """
    _quantization = 1000  # if change to 1, then it would take the first (smallest) index lie in the voxel

    offset = torch.floor(points.min(dim=0)[0] / voxel_size).long()
    grid = torch.floor(points / voxel_size)
    center = (grid + 0.5) * voxel_size
    dist = ((points - center) ** 2).sum(dim=1) ** 0.5
    dist = (
        dist / dist.max() * (_quantization - 1)
    ).long()  # for speed up # [0-_quantization]

    grid = grid.long() - offset
    v_size = grid.max().float().ceil()
    grid_idx = grid[:, 0] + grid[:, 1] * v_size + grid[:, 2] * v_size * v_size

    unique, inverse = torch.unique(grid_idx, return_inverse=True)
    idx_d = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)

    offset = 10 ** len(str(idx_d.max().item()))

    idx_d = idx_d + dist.long() * offset

    idx = torch.empty(
        unique.shape, dtype=inverse.dtype, device=inverse.device
    ).scatter_reduce_(
        dim=0, index=inverse, src=idx_d, reduce="amin", include_self=False
    )
    # https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html
    # This operation may behave nondeterministically when given tensors on
    # a CUDA device. consider to change a more stable implementation

    idx = idx % offset
    return idx

def crop_points(
    points: torch.tensor,
    ts: torch.tensor,
    min_z_th=-3.0,
    max_z_th=100.0,
    min_range=2.75,
    max_range=100.0,
):
    # print(f"Cropping with min_z_th: {min_z_th}, max_z_th: {max_z_th}, min_range: {min_range}, max_range: {max_range}")
    # print(f"Before cropping, the point count is {points.shape[0]}")
    dist = torch.norm(points, dim=1)
    filtered_idx = (
        (dist > min_range)
        & (dist < max_range)
        & (points[:, 2] > min_z_th)
        & (points[:, 2] < max_z_th)
    )
    points = points[filtered_idx]
    # print(f"After cropping, the point count is {points.shape[0]}")
    if ts is not None:
        ts = ts[filtered_idx]
    return points, ts

def check_numpy_pose_matrix(T:np.ndarray):
    """
    Check if the input matrix is a valid pose matrix
    Args:
        T (np.ndarray): 4x4 pose matrix
    Returns:
        bool: True if valid, False otherwise
    """
    # check infinity
    if np.isinf(T).any():
        logger.error("Pose matrix contains infinity values")
        return False
    # check NaN
    if np.isnan(T).any():
        logger.error("Pose matrix contains NaN values")
        return False
    if T.shape != (4, 4):
        logger.error(f"Invalid pose matrix shape: {T.shape}")
        return False
    if not np.allclose(T[3, :], [0, 0, 0, 1]):
        logger.error(f"Invalid pose matrix last row: {T[3, :]}")
        return False
    R = T[:3, :3]
    if not np.allclose(np.linalg.det(R), 1):
        logger.error(f"Invalid rotation matrix determinant: {np.linalg.det(R)}")
        return False
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5):
        logger.error(f"Invalid rotation matrix transpose: {R.T @ R}")
        return False
    return True

def read_kitti_format_poses(filename: str) -> List[np.ndarray]:
    """
    read pose file (with the kitti format)
    returns -> list, transformation before calibration transformation
    if the format is incorrect, return None
    """
    poses = []
    with open(filename, 'r') as file:            
        for line in file:
            values = line.strip().split()
            if len(values) < 12: # FIXME: > 12 means maybe it's a 4x4 matrix
                print('Not a kitti format pose file')
                return None

            values = [float(value) for value in values]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            poses.append(pose)
    
    return poses

def write_kitti_format_poses(filename: str, poses_np: np.ndarray, direct_use_filename = False):
    poses_out = poses_np[:, :3, :]
    poses_out_kitti = poses_out.reshape(poses_out.shape[0], -1)

    if direct_use_filename:
        fname = filename
    else:
        fname = f"{filename}_kitti.txt"
    
    np.savetxt(fname=fname, X=poses_out_kitti)