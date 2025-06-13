import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.ndimage import distance_transform_edt
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import colors
import grid_opt.utils.utils as utils
from grid_opt.utils.utils_sdf import *
from grid_opt.diff import *

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Sdf2D(Dataset):
    
    def __init__(self, mapfile, pix_per_meter=100.0, samples_near=500, samples_unif=500, near_zero_thresh=0.1, batch_size=100, random_batch=False):
        super().__init__()
        occupancy_map = mpimg.imread(mapfile)
        self.pix_per_meter = pix_per_meter
        self.sdf = compute_signed_distance_field(occupancy_map, pix_per_meter=pix_per_meter)
        self.min_x = 0.
        self.max_x = self.sdf.shape[0] / pix_per_meter
        self.min_y = 0
        self.max_y = self.sdf.shape[1] / pix_per_meter
        loc_near, val_near = sample2d(self.sdf, samples_near, mode=SampleMode.NEAR_ZERO, near_zero_thresh=near_zero_thresh)
        loc_rand, val_rand = sample2d(self.sdf, samples_unif, mode=SampleMode.UNIFORM)
        sample_locs = np.concatenate((loc_near, loc_rand), axis=0)
        sample_vals = np.concatenate((val_near, val_rand), axis=0)
        assert sample_locs.shape[1] == 2
        assert sample_vals.shape[1] == 1
        self.num_samples = sample_locs.shape[0]
        self.batch_size = batch_size
        self.coords = torch.from_numpy(sample_locs / pix_per_meter).float()
        self.sdfs = torch.from_numpy(sample_vals).float()

        Hout, Wout = occupancy_map.shape
        x = np.linspace(self.min_x, self.max_x, Hout)
        y = np.linspace(self.min_y, self.max_y, Wout)
        grid_x, grid_y = np.meshgrid(x, y)
        grid_np = np.stack((grid_x, grid_y), axis=-1)  
        self.full_coords = torch.from_numpy(grid_np).float().transpose(0, 1)
        self.full_sdfs = torch.from_numpy(self.sdf).float().unsqueeze(-1)
        self.random_batch = random_batch
        self.print_bound()

    
    def __len__(self):
        len = self.num_samples // self.batch_size
        if self.num_samples % self.batch_size != 0:
            len += 1
        return len
    

    def __getitem__(self, idx):
        if self.random_batch:
            raise ValueError("This option is deprecated. ")
            selected_indices = np.random.choice(self.num_samples, size=self.batch_size)
        else:
            index_start = idx * self.batch_size
            index_end = min(index_start + self.batch_size, self.num_samples)
            selected_indices = range(index_start, index_end)
        input_dict = {
            'coords': self.coords[selected_indices, :]
        }
        gt_dict = {
            'sdf': self.sdfs[selected_indices, :]
        }
        return input_dict, gt_dict
    

    def visualize(self, fig_path=None, show_samples=True):
        # utils.visualize_grid_scalar(self.sdf, fig_path, cmap='seismic')
        grid = self.sdf
        fig, ax = plt.subplots()
        cmap = 'seismic'
        rmax = np.max(grid)
        rmin = np.min(grid)
        cmap = plt.get_cmap('seismic')  # or use 'seismic', 'bwr', 'PiYG', etc.
        try:
            norm = colors.TwoSlopeNorm(vmin=rmin, vcenter=0, vmax=rmax)
        except:
            logger.warning(f"Failed to setup colormap! sdf_min={rmin}, sdf_max={rmax}")
            norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        im = ax.imshow(grid, cmap=cmap, norm=norm)
        plt.colorbar(im)
        if show_samples:
            pix_sdf = self.pix_per_meter * self.coords.detach().cpu().numpy()
            ax.scatter(pix_sdf[:, 1], pix_sdf[:, 0], color='yellow', s=10, zorder=3, label='SDF samples')
        ax.axis('off')
        ax.invert_yaxis()
        ax.set_title("Grid values")
        if fig_path is not None:
            plt.savefig(fig_path)
            plt.close(fig)
        else:
            plt.show()
        # TODO: plot gradient samples when incorporated
        # See visualize_2d_sdf_problem below for legacy impl.
    

    def print_bound(self):
        logger.info(f"SDF 2D bound: x: {self.min_x}, {self.max_x}. y: {self.min_y}, {self.max_y}.")

    def get_bound(self):
        return [[self.min_x, self.max_x], [self.min_y, self.max_y]]



def create_2d_sdf_problem(args):
    assert False, "Legacy code."
    cfg = load_config(args.config, args.default_config)
    device = cfg['device']
    dtype = torch.float32
    occupancy_map = mpimg.imread(args.map_file)
    sdf = compute_signed_distance_field(occupancy_map, pix_per_meter=args.pix_per_meter)
    min_x = 0.
    max_x = sdf.shape[0] / args.pix_per_meter
    min_y = 0
    max_y = sdf.shape[1] / args.pix_per_meter
    cfg['model']['grid']['bound'] = [[min_x, max_x], [min_y, max_y]]
    logger.warning(f"Changed model grid bound to match input 2d map: \n, {cfg['model']['grid']['bound']}")

    # Training data:
    loc_near, val_near = sample2d(sdf, args.N_near, mode=SampleMode.NEAR_ZERO, near_zero_thresh=args.near_zero_thresh)
    loc_rand, val_rand = sample2d(sdf, args.N_unif, mode=SampleMode.UNIFORM)
    sample_locs = np.concatenate((loc_near, loc_rand), axis=0)
    sample_vals = np.concatenate((val_near, val_rand), axis=0)
    assert sample_locs.shape[1] == 2
    assert sample_vals.shape[1] == 1
    # Sample SDF values
    train_coords= torch.tensor(sample_locs / args.pix_per_meter, device=device, dtype=dtype)
    train_vals = torch.tensor(sample_vals, device=device, dtype=dtype)
    # Sample SDF gradients near surface
    grad_coords = torch.tensor(loc_near / args.pix_per_meter, dtype=dtype, device=device)
    grad_coords, grad_val = compute_sdf_gradient_torch(sdf, grad_coords, args.pix_per_meter, finite_diff_eps=1e-1)
    # grad_val = torch.tensor(sdf_grad[loc_near[:,0], loc_near[:,1]], device=device, dtype=dtype)
    
    # Assemble into dict
    train_data = {
        'sdf': (train_coords, train_vals), 
        'sdf_grad': (grad_coords, grad_val)
    }
    logger.debug(f"SDF coords {train_coords.shape}, vals {train_vals.shape}")
    logger.debug(f"Grad coords {grad_coords.shape}, vals {grad_val.shape}")

    # Validation data: use entire SDF
    validation_vals = torch.tensor(sdf, dtype=dtype, device=device).unsqueeze(-1)
    Hout, Wout = occupancy_map.shape
    x = np.linspace(min_x, max_x, Hout)
    y = np.linspace(min_y, max_y, Wout)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_np = np.stack((grid_x, grid_y), axis=-1)  
    validation_coords = torch.tensor(grid_np, dtype=dtype, device=device).transpose(0, 1)
    val_data = (validation_coords, validation_vals)

    return cfg, sdf, train_data, val_data


def visualize_2d_sdf_problem(ax, args, sdf, train_data, val_data):
    """
    Visualize points, gradients, and the signed distance field (SDF) on a given matplotlib axes object.
    
    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes object to plot on.
        sdf (np.ndarray): 2D numpy array representing the signed distance field.
        points (np.ndarray): (N, 2) array of points (x, y) on the grid.
        gradients (np.ndarray): (N, 2) array of gradient vectors at the points.
    """
    assert False, "Legacy code."
    # fig, ax = plt.subplots(figsize=(10, 10))
    sdf_max = np.max(sdf)
    sdf_min = np.min(sdf)
    cmap = plt.get_cmap('seismic')  # or use 'seismic', 'bwr', 'PiYG', etc.
    norm = colors.TwoSlopeNorm(vmin=sdf_min, vcenter=0, vmax=sdf_max)
    im = ax.imshow(sdf, cmap=cmap, norm=norm)
    
    # Plot SDF samples
    # Note that we need to flip the point dimension, because in plt 
    # horizontal is the X axis, and vertical is the Y axis
    coords_sdf, _ = train_data['sdf']
    pix_sdf = args.pix_per_meter * coords_sdf.detach().cpu().numpy()
    ax.scatter(pix_sdf[:, 1], pix_sdf[:, 0], color='yellow', s=10, zorder=3, label='SDF samples')
    
    # Plot gradient samples
    coords_grad, vec_grad = train_data['sdf_grad']
    pix_grad = args.pix_per_meter * coords_grad.detach().cpu().numpy()
    vec_grad = vec_grad.detach().cpu().numpy()
    vec_grad_norm = np.linalg.norm(vec_grad, axis=-1)
    if not np.allclose(vec_grad_norm, 1.0, rtol=1e-3, atol=1e-3):
        logging.warning('Norm of training gradient not close to one!')
        logging.warning(f"Max = {np.max(vec_grad_norm)}, min = {np.min(vec_grad_norm)} ")

    ax.quiver(pix_grad[:, 1], pix_grad[:, 0], vec_grad[:, 1], vec_grad[:, 0], 
              color='black', scale=0.02, scale_units='x', width=0.005, zorder=3, label='Gradient samples')
    
    ax.invert_yaxis()
    ax.legend()
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('2D SDF Problem')
    return ax


def compute_signed_distance_field(occupancy_map, pix_per_meter=10.0):
    """
    Compute the signed distance field from an occupancy map.

    Args:
        occupancy_map (np.ndarray): A 2D grayscale image representing the occupancy of the map.
                                    Occupied regions should be 0, free space should be non-zero.
        
        pix_per_meter (float): scale parameter that is used to convert pixels to meter.

    Returns:
        np.ndarray: A 2D signed distance field.
    """
    # Compute the distance transform for free space (positive distances)
    dist_free_space = distance_transform_edt(occupancy_map != 0)

    # Compute the distance transform for occupied space (negative distances)
    dist_occupied_space = distance_transform_edt(occupancy_map == 0)

    # Combine the distance transforms to get the signed distance field
    sdf = dist_free_space - dist_occupied_space

    # sdf = dist_free_space

    return sdf / pix_per_meter


def compute_sdf_gradient_torch(sdf, coords, pix_per_meter=100.0, finite_diff_eps=0.1):
    N = coords.shape[0]
    assert coords.shape[-1] == 2
    H, W = sdf.shape
    bounds = np.asarray([[0., H/pix_per_meter], [0., W/pix_per_meter]])
    bounds = torch.tensor(bounds, device=coords.device, dtype=coords.dtype)
    sdf_tensor = torch.tensor(sdf.T, device=coords.device, dtype=coords.dtype)
    sdf_tensor = sdf_tensor.unsqueeze(0)
    sdf_tensor = sdf_tensor.unsqueeze(0)
    assert sdf_tensor.shape == (1, 1, W, H)

    def f(x):
        # Define a method f that takes input spatial coordinate x
        # and return the GT SDF values
        x_nrm = normalize_coordinates(x, bounds)
        vals = F.grid_sample(
            sdf_tensor,
            x_nrm.reshape(1, N, 1, 2), 
            align_corners=True,
            mode='nearest',
            padding_mode='zeros'
        )[0, :, :, 0].transpose(0, 1)
        return vals

    gradient = gradient2d(coords, f, method='finitediff', finite_diff_eps=finite_diff_eps)

    # Filter out based on gradient norm (should satisfy eikonal)
    gradient_norms = torch.linalg.vector_norm(gradient, dim=1)
    valid_indices = torch.argwhere(torch.abs(gradient_norms - 1.0) < 0.1).flatten()
    # logger.info(f"gradient_norms: \n {gradient_norms.shape}")
    # logger.info(f"valid_indices: \n {valid_indices.shape}")
    gradient = gradient[valid_indices, :] 
    gradient = gradient / torch.linalg.vector_norm(gradient, axis=-1, keepdim=True)
    coords = coords[valid_indices, :]

    return coords, gradient


def evaluate_full_sdf(coords_full, model, batch_size=16):
    """Evaluate the SDF row by row, but in smaller batches to avoid memory issues.

    Args:
        coords_full (torch.Tensor): Input coordinates of shape (H, W, 2).
        model (torch.nn.Module): The model used to evaluate the SDF.
        batch_size (int, optional): Number of rows to process at once. Defaults to 64.

    Returns:
        torch.Tensor: The evaluated output of shape (H, W, 1).
    """
    torch.cuda.empty_cache()
    H, W = coords_full.shape[0], coords_full.shape[1]
    output = torch.zeros((H, W, 1), device=coords_full.device)

    with torch.no_grad():
        # Process input in batches
        for start_row in range(0, H, batch_size):
            end_row = min(start_row + batch_size, H)
            
            # Extract the batch
            batch_coords = coords_full[start_row:end_row, :, :]
            
            # Evaluate the model on the batch
            output[start_row:end_row, :, :] = model(batch_coords)
    
    return output.squeeze(-1)

def evaluate_full_gradient_2d(coords_full, model, batch_size=16):
    torch.cuda.empty_cache()
    H, W = coords_full.shape[0], coords_full.shape[1]
    output = torch.zeros((H, W, 2), device=coords_full.device)
    with torch.enable_grad():
        # Process input in batches
        for start_row in range(0, H, batch_size):
            end_row = min(start_row + batch_size, H)
            
            # Extract the batch
            batch_coords = coords_full[start_row:end_row, :, :]
            
            # Evaluate the model on the batch
            output[start_row:end_row, :, :] = gradient2d(batch_coords, model, method='autograd', create_graph=False)
    return output