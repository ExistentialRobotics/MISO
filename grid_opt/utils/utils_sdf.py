import numpy as np
import torch
import trimesh
import open3d as o3d
import mcubes
import packaging
import os
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.ticker as ticker
import grid_opt.utils.utils as utils

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def sign_mask_from_gt_sdf(gt_sdf:torch.Tensor, trunc_dist=0.15) -> torch.Tensor:
    """Get sign labels from ground truth SDF labels

    Args:
        gt_sdf (torch.Tensor): (N,1) tensor of SDF values
        trunc_dist (float, optional): Defaults to 0.15.

    Returns:
        torch.Tensor: (N,1) tensor of SDF signs where the sign is +1 for SDF with 
        value > trunc_dist, and zero otherwise.
    """
    num_total = gt_sdf.shape[0]
    # Get free space indices
    free_space_indices = torch.nonzero(gt_sdf > trunc_dist, as_tuple=False)[:, 0]
    gt_sdf_sign = torch.zeros_like(gt_sdf)
    gt_sdf_sign[free_space_indices, :] = 1
    num_free = torch.count_nonzero(gt_sdf_sign)
    logger.debug(f"{num_free}/{num_total} coords in free space.")
    return gt_sdf_sign
    

def valid_mask_from_gt_sdf(gt_sdf:torch.Tensor, trunc_dist=0.15) -> torch.Tensor:
    """Get valid labels from ground truth SDF labels

    Args:
        gt_sdf (torch.Tensor): (N,1) tensor of SDF values
        trunc_dist (float, optional): Defaults to 0.15.

    Returns:
        torch.Tensor: (N,1) tensor of SDF signs where valid is True for SDF with 
        abs value < trunc_dist.
    """
    num_total = gt_sdf.shape[0]
    gt_dist = torch.abs(gt_sdf)
    gt_sdf_valid = torch.zeros_like(gt_sdf)
    valid_indices = torch.nonzero(gt_dist < trunc_dist, as_tuple=False)[:, 0]
    gt_sdf_valid[valid_indices, :] = 1
    num_valid = torch.count_nonzero(gt_sdf_valid)
    logger.debug(f"{num_valid}/{num_total} coords with valid sdf.")
    return gt_sdf_valid


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if packaging.version.parse(torch.__version__) < packaging.version.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def extract_fields(bound_min: torch.Tensor, bound_max: torch.Tensor, resolution, query_func):
    bound_min = bound_min.detach().cpu().numpy()
    bound_max = bound_max.detach().cpu().numpy()
    N = 16
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3]
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [N, 1] --> [x, y, z]
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min: torch.Tensor, bound_max: torch.Tensor, resolution, threshold, query_func):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def save_mesh(
        model, 
        bounds: torch.Tensor, 
        save_path=None, 
        resolution=256, 
        device='cuda:0', 
        flip_face=True, 
        transform:torch.Tensor=None   # [4,4] matrix
    ):

    if save_path is not None:
        logger.info(f"Saving mesh to {save_path}...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def query_func(pts):
        pts = pts.to(device)
        with torch.no_grad():
            sdfs = model(pts)
        return sdfs

    vertices, triangles = extract_geometry(
        bounds[:,0], bounds[:,1], resolution=resolution, threshold=0, query_func=query_func)
    
    if flip_face:
        triangles = triangles[:, [2,1,0]]

    mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
    if transform is not None:
        mesh.apply_transform(transform.detach().cpu().numpy())
    if save_path is not None:
        mesh.export(save_path, file_type='ply')
    # Return open3d mesh
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d


def visualize_sdf_plane(model, bounds: torch.Tensor, resolution=512, axis='y', fig_path=None, device='cuda:0', 
                        show_colorbar=True, show_title=True, hide_axis=False, title=None):
    def query_func(pts):
        pts = pts.to(device)
        with torch.no_grad():
            sdfs = model(pts)
        return sdfs
    sdf_values = extract_fields(bounds[:,0], bounds[:,1], resolution=resolution, query_func=query_func)
    
    _idx = resolution // 4  
    if axis == 'y':
        sdf_plane = sdf_values[:, _idx, :]
    elif axis == 'z':
        sdf_plane = sdf_values[:, :, _idx]
    elif axis == 'x':
        sdf_plane = sdf_values[_idx, :, :]
    else:
        raise ValueError(f"Invalid axis {axis}!")
    mean_sdf = np.mean(sdf_plane)
    # logger.info(f"Mean SDF value={mean_sdf}.")
    sdf_max = np.max(sdf_plane)
    sdf_min = np.min(sdf_plane)
    cmap = plt.get_cmap('seismic')  # or use 'seismic', 'bwr', 'PiYG', etc.
    try:
        norm = colors.TwoSlopeNorm(vmin=sdf_min, vcenter=0, vmax=sdf_max)
    except:
        logger.warning(f"Failed to setup colormap! sdf_min={sdf_min}, sdf_max={sdf_max}")
        norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    
    fig, ax = plt.subplots(figsize=(10, 10)) 
    im=ax.imshow(sdf_plane, cmap=cmap, norm=norm)
    ax.invert_yaxis()
    ax.invert_xaxis()
    if show_title:
        title_text = title if title is not None else f"SDF plane ({axis} axis): N={resolution}, avg={mean_sdf:.1e}"
        ax.set_title(title_text, fontsize=50)
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=25)  # Set colorbar font size
        # cbar.set_ticks([np.min(sdf_plane), np.max(sdf_plane)]) # Set ticks to only show min and max
        # Set a fixed number of ticks (e.g., 5)
        cbar.locator = ticker.MaxNLocator(nbins=5)
        cbar.update_ticks()
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))  # Force single-digit precision
    if hide_axis:
        plt.axis('off')
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path)
        logger.info(f"Saved SDF plane figure to {fig_path}.")
    else:
        plt.show()


def sphere_tracing(
    query_func,
    origins, # Nx3
    directions, # Nx3
    min_dist=1e-3,
    max_dist=5e1,
    max_iters=100, 
    epsilon=1e-5
):
    '''
    Input:
        implicit_fn: a module that computes a SDF at a query point
        origins: N_rays X 3
        directions: N_rays X 3
    Output:
        points: N_rays X 3 points indicating ray-surface intersections. For rays that do not intersect the surface,
                the point can be arbitrary.
        mask: N_rays X 1 (boolean tensor) denoting which of the input rays intersect the surface.
    '''
    n = origins.shape[0]
    assert origins.ndim == 2 and directions.ndim == 2, f"Wrong shape: origins {origins.shape}, directions {directions.shape}"
    assert directions.shape[0] == n
    directions = utils.normalize_last_dim(directions)
    points = origins + min_dist * directions
    for i in range(max_iters):
        dists = torch.norm(points - origins, dim=1, keepdim=True)
        sdfs = query_func(points) 
        mask_negative = sdfs < -0.1
        mask_converge = sdfs < epsilon  # converged rays
        mask_far = dists > max_dist    
        mask_stop = torch.logical_or(mask_converge, mask_far)
        logger.debug(f"iter {i}: converged {torch.sum(mask_converge)}, far {torch.sum(mask_far)}, negative {torch.sum(mask_negative)}")
        # check if all rays finish
        if torch.sum(mask_stop) == n:
            break
        # update points
        points = mask_stop * points + ~mask_stop * (points + sdfs * directions)

    # print(f"Sphere tracing iterations: {i}.")
    return points, mask_converge.reshape(-1,1)


###############################
#####       2D SDF       ######
###############################

class SampleMode(Enum):
    UNIFORM = 1
    NEAR_ZERO = 2


def sample2d(array, N, mode=SampleMode.UNIFORM, near_zero_thresh=0.1):
    assert array.ndim == 2
    if mode == SampleMode.UNIFORM:
        total_pixels = array.size
        random_indices = np.random.choice(total_pixels, N, replace=False)
        sampled_locations = np.column_stack(np.unravel_index(random_indices, array.shape))
        
    elif mode == SampleMode.NEAR_ZERO:
        surface_indices = np.argwhere(np.abs(array) <= near_zero_thresh)
        if surface_indices.shape[0] < N:
            logger.warning(f"Warning: Only {surface_indices.shape[0]} surface points found. Returning all.")
            sampled_locations = surface_indices
        else:
            sampled_locations = surface_indices[np.random.choice(surface_indices.shape[0], N, replace=False)]
    
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")

    sampled_values = array[sampled_locations[:, 0], sampled_locations[:, 1]]
    
    return sampled_locations, np.reshape(sampled_values, (N, 1))


def extract_sdf_2d(model, dataset, device='cuda:0'):
    sdf_plane = torch.zeros_like(dataset.full_sdfs).to(device)
    full_coords = dataset.full_coords.to(device)
    num_rows = full_coords.shape[0]
    for row in range(num_rows):
        inputs = full_coords[row,:,:]
        sdf_plane[row, :] = model(inputs)
    return sdf_plane


def visualize_sdf_2d(model, dataset, fig_path=None, device='cuda:0'):
    sdf_plane = extract_sdf_2d(model, dataset, device)
    sdf_plane = sdf_plane.squeeze().detach().cpu().numpy()
    utils.visualize_grid_scalar(sdf_plane, fig_path, cmap='seismic')


def visualize_sdf_residuals_2d(model, dataset, fig_path=None, device='cuda:0', magnitude_only=False):
    sdf_plane = extract_sdf_2d(model, dataset, device)
    sdf_pred = sdf_plane.squeeze().detach().cpu().numpy()
    sdf_gt = dataset.sdf
    sdf_residuals = sdf_gt - sdf_pred
    utils.visualize_grid_scalar(sdf_residuals, fig_path, cmap='seismic')
