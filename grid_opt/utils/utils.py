import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import colors
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import sys
sys.path.append("./third_party/cuda_gridsample_grad2")

def normalize_last_dim(A:torch.Tensor, epsilon=1e-8):
    norms = torch.norm(A, dim=-1, keepdim=True)
    norms = norms + epsilon
    return A / norms


def normalize_coordinates(queries: torch.tensor, bounds: torch.tensor):
    """
    Normalize coordinates to be between [-1, 1] based on given bounds.

    Args:
        queries (torch.Tensor): Tensor of shape (N, d) or (H, W, d) representing coordinates in d dimensions.
        bounds (torch.Tensor): Tensor of shape (d, 2) where each row represents the [min, max] bounds for each dimension.

    Returns:
        torch.Tensor: Normalized coordinates with values between [-1, 1]. The shape of the output will match the shape of the input queries.
    """
    d = bounds.shape[0]
    assert queries.shape[-1] == d

    # Determine the number of dimensions in the queries
    if queries.dim() == 2:
        # 2D case: shape (N, d)
        bounds_min = bounds[:, 0].view(1, -1)  # Shape (1, d)
        bounds_max = bounds[:, 1].view(1, -1)  # Shape (1, d)
    elif queries.dim() == 3:
        # 3D case: shape (H, W, d)
        bounds_min = bounds[:, 0].view(1, 1, -1)  # Shape (1, 1, d)
        bounds_max = bounds[:, 1].view(1, 1, -1)  # Shape (1, 1, d)
    else:
        raise ValueError("queries tensor must be either 2D or 3D")

    # Normalize the queries
    normalized_queries = 2 * (queries - bounds_min) / (bounds_max - bounds_min) - 1
    
    return normalized_queries

def denormalize_coordinates(normalized_queries: torch.Tensor, bounds: torch.Tensor):
    """
    Denormalize coordinates from the range [-1, 1] to the original coordinate range based on given bounds.

    Args:
        normalized_queries (torch.Tensor): Tensor of shape (N, d) or (H, W, d) with normalized coordinates in the range [-1, 1].
        bounds (torch.Tensor): Tensor of shape (d, 2) where each row represents the [min, max] bounds for each dimension.

    Returns:
        torch.Tensor: Denormalized coordinates in the original coordinate range. The shape of the output matches the input normalized_queries.
    """
    d = bounds.shape[0]
    assert normalized_queries.shape[-1] == d

    # Determine the number of dimensions in the normalized_queries
    if normalized_queries.dim() == 2:
        # 2D case: shape (N, d)
        bounds_min = bounds[:, 0].view(1, -1)  # Shape (1, d)
        bounds_max = bounds[:, 1].view(1, -1)  # Shape (1, d)
    elif normalized_queries.dim() == 3:
        # 3D case: shape (H, W, d)
        bounds_min = bounds[:, 0].view(1, 1, -1)  # Shape (1, 1, d)
        bounds_max = bounds[:, 1].view(1, 1, -1)  # Shape (1, 1, d)
    else:
        raise ValueError("normalized_queries tensor must be either 2D or 3D")

    # Denormalize the queries
    original_queries = (normalized_queries + 1) / 2 * (bounds_max - bounds_min) + bounds_min

    return original_queries

###############################
#####    Grid Utils      ######
###############################


def interp_2d(features, x, ignore_level=None):
    # Input query has [batch_size, 2]
    # i.e., a list of 2D coordinates
    # Output should be [batch_size, Fdim]
    num_levels = len(features)
    if ignore_level is None:
        ignore_level = np.zeros(num_levels).astype(bool)
    N = x.shape[0]
    sample_coords = x.reshape(1, N, 1, 2)
    level_feats = []
    for level in range(num_levels):
        feats = F.grid_sample(
            features[level],
            sample_coords, 
            align_corners=False,
            mode='bilinear',
            padding_mode='zeros'
        )[0, :, :, 0].transpose(0, 1)
        if not ignore_level[level]:
            level_feats.append(feats)
        else:
            level_feats.append(torch.zeros_like(feats))
    return torch.cat(level_feats, dim=1) 


def interp_3d(features, x, ignore_level=None, second_order_grid_sample=False):
    if second_order_grid_sample:
        import cuda_gridsample as cu
        grid_sample_func = cu.grid_sample_3d
    else:
        grid_sample_func = F.grid_sample
    num_levels = len(features)
    if ignore_level is None:
        ignore_level = np.zeros(num_levels).astype(bool)
    N = x.shape[0]
    sample_coords = x.reshape(1, N, 1, 1, 3)
    level_feats = []
    for level in range(num_levels):
        inputs = features[level]
        if second_order_grid_sample and not inputs.requires_grad:
            inputs = inputs.requires_grad_(True)
        feats = grid_sample_func(
            inputs,
            sample_coords,
            align_corners=False,
            padding_mode='zeros'
        )[0, :, :, 0, 0].transpose(0, 1)
        if not ignore_level[level]:
            level_feats.append(feats)
        else:
            level_feats.append(torch.zeros_like(feats))
    return torch.cat(level_feats, dim=1)


def grid_interp_regular(reg_grids, x, ignore_level=None):
    """Interpolate from a list of regular feature grids.

    Args:
        reg_grids: Each element is an instance of FeatureGrid.
        x: Unnormalized query coordinates
        ignore_level: list of bools indicating which level to ignore. Defaults to None.

    Returns:
        Interpolated features, concatenated across the levels.
    """
    num_levels = len(reg_grids)
    if ignore_level is None:
        ignore_level = np.zeros(num_levels).astype(bool)
    level_feats = []
    for level in range(num_levels):
        feats = reg_grids[level].interpolate(x)
        if not ignore_level[level]:
            level_feats.append(feats)
        else:
            level_feats.append(torch.zeros_like(feats))
    return torch.cat(level_feats, dim=1)


def grid_interp_VM(vm_grids, vm_bases, x, ignore_level=None):
    """Interpolate from a list of feature grids.

    Args:
        vm_grids: Each element is an instance of FeatureGridVM.
        vm_bases: Each element is an instance of VMBasis.
        x: Unnormalized query coordinates
        ignore_level: list of bools indicating which level to ignore. Defaults to None.

    Returns:
        Interpolated features, concatenated across the levels.
    """
    num_levels = len(vm_grids)
    assert len(vm_bases) == num_levels
    if ignore_level is None:
        ignore_level = np.zeros(num_levels).astype(bool)
    level_feats = []
    for level in range(num_levels):
        coeffs = vm_grids[level].interpolate(x)
        feats = vm_bases[level](coeffs)
        if not ignore_level[level]:
            level_feats.append(feats)
        else:
            level_feats.append(torch.zeros_like(feats))
    return torch.cat(level_feats, dim=1)


def grid_decode(feats, x, decoder=None, pos_invariant=True):
    assert feats.ndim == 2
    
    # Pass through decoder
    if decoder is not None:
        if pos_invariant:
            inputs = feats
        else:
            inputs = torch.cat((feats, x), dim=1)
        preds = decoder(inputs)
    else:
        # In this case, the grid directly gives prediction
        preds = feats

    return preds


def grid_pool_2d_avg(coords, features, grid_bound, cell_size):
    """Given a set of 2D points each with a latent feature, 
    pool the features onto a 2D regular grid, where each grid cell stores
    the average of the features it contains.
    """
    P = coords
    F = features
    assert P.ndim == 2 and F.ndim == 2, f"Invalid shape: coords {coords.shape}, features {features.shape}."
    N, d = F.shape
    assert P.shape == (N, 2)
    grid_len = (grid_bound[:,1] - grid_bound[:,0]).cpu().numpy()
    H, W = np.ceil(grid_len / cell_size).astype(int) 
    logger.debug(f"Grid size: {H} x {W}.")
    col_indices = ((P[:, 1] - grid_bound[1, 0]) / cell_size).long().clamp(0, W - 1)
    row_indices = ((P[:, 0] - grid_bound[0, 0]) / cell_size).long().clamp(0, H - 1)
    G = torch.zeros(H, W, d, device=F.device)
    counts = torch.zeros(H, W, dtype=torch.int32, device=F.device)
    G_flat = G.view(-1, d)
    counts_flat = counts.view(-1)
    linear_indices = (row_indices * W + col_indices)
    G_flat.scatter_add_(0, linear_indices.unsqueeze(-1).expand(-1, d), F)
    counts_flat.scatter_add_(0, linear_indices, torch.ones(N, dtype=torch.int32, device=F.device))
    counts = counts.view(H, W)
    G = G.view(H, W, d)
    G /= counts.clamp(min=1).unsqueeze(-1)  # Normalize to get the average feature per grid cell
    return G


def grid_pool_3d_avg(coords, features, grid_bound, cell_size):
    """
    Given a set of 3D points each with a latent feature, 
    pool the features onto a 3D regular grid, where each grid cell stores
    the average of the features it contains.
    
    Args:
        coords (torch.Tensor): (N, 3) tensor of 3D point coordinates.
        features (torch.Tensor): (N, d) tensor of features for each point.
        grid_bound (torch.Tensor): (3, 2) tensor specifying min and max bounds for x, y, z.
        cell_size (float): The size of each grid cell along x, y, and z dimensions.
    
    Returns:
        torch.Tensor: A grid of shape (D, H, W, d) where each cell contains the average feature.
    """
    P = coords
    F = features
    assert P.ndim == 2 and F.ndim == 2, f"Invalid shape: coords {coords.shape}, features {features.shape}."
    N, d = F.shape
    assert P.shape == (N, 3)
    
    # Calculate the size of the grid in each dimension
    grid_len = (grid_bound[:, 1] - grid_bound[:, 0]).cpu().numpy()
    H, W, D = np.ceil(grid_len / cell_size).astype(int) 
    logger.debug(f"Grid size: {H} x {W} x {D}")
    
    # Calculate indices for each point in the grid
    z_indices = ((P[:, 2] - grid_bound[2, 0]) / cell_size).long().clamp(0, D - 1)
    y_indices = ((P[:, 1] - grid_bound[1, 0]) / cell_size).long().clamp(0, W - 1)
    x_indices = ((P[:, 0] - grid_bound[0, 0]) / cell_size).long().clamp(0, H - 1)
    
    # Initialize accumulation grid and count grid
    G = torch.zeros(H, W, D, d, device=F.device)
    counts = torch.zeros(H, W, D, dtype=torch.int32, device=F.device)
    
    # Flatten grids for scatter operations
    G_flat = G.view(-1, d)
    counts_flat = counts.view(-1)
    
    # Calculate linear indices for 3D grid cells
    # linear_indices = (z_indices * H * W) + (y_indices * W) + x_indices
    linear_indices = (x_indices * W * D) + (y_indices * D) + z_indices
    
    # Scatter-add features and counts
    G_flat.scatter_add_(0, linear_indices.unsqueeze(-1).expand(-1, d), F)
    counts_flat.scatter_add_(0, linear_indices, torch.ones(N, dtype=torch.int32, device=F.device))
    
    # Reshape grids back to 3D and normalize by counts
    counts = counts.view(H, W, D)
    G = G.view(H, W, D, d)
    G /= counts.clamp(min=1).unsqueeze(-1)  # Normalize to get the average feature per grid cell
    
    return G


def all_grid_positions(features):
    """Return the 3D coordinates of the centers of a 3D regular grid.
    Return shape is (1, Z, Y, X, 3)
    """
    B, C, D, H, W = features.shape
    half_dx = 0.5 * 1/D
    half_dy = 0.5 * 1/H
    half_dz = 0.5 * 1/W
    xs = 2 * torch.linspace(half_dx, 1-half_dx, D) - 1.
    ys = 2 * torch.linspace(half_dy, 1-half_dy, H) - 1.
    zs = 2 * torch.linspace(half_dz, 1-half_dz, W) - 1.
    xv, yv, zv = torch.meshgrid([xs, ys, zs])
    grid = torch.stack((zv, yv, xv), axis=-1)  
    return grid.unsqueeze(0)


def meshgrid_full(bounds, resolution):
    """
    Generate a full 3D grid within the given bounds.

    Parameters:
        bounds (np.ndarray): Shape (3, 2), where each row is [min, max] for x, y, z.
        resolution (float): Grid spacing.

    Returns:
        np.ndarray: Shape (N, 3), where each row is a [x, y, z] point.
    """
    # Generate ranges for x, y, z
    ranges = [np.arange(bounds[i, 0], bounds[i, 1] + resolution, resolution) for i in range(3)]
    
    # Create 3D meshgrid
    xx, yy, zz = np.meshgrid(*ranges, indexing='ij')

    # Flatten and stack into (N, 3)
    grid = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    return grid


def meshgrid_with_fixed_dim(bounds, resolution, fixed_dim, fixed_value):
    """
    Generate a (N, 3) meshgrid by fixing one dimension and discretizing the other two.

    Parameters:
        bounds (np.ndarray): Shape (3, 2), each row is [min, max] for x, y, z.
        resolution (float): Grid spacing.
        fixed_dim (int or str): Dimension to fix: 0/'x', 1/'y', or 2/'z'.
        fixed_value (float): Value to fix for the specified dimension.

    Returns:
        np.ndarray: Shape (H, W, 3), each row is a 3D point.
    """
    # Map string names to indices if needed
    if isinstance(fixed_dim, str):
        dim_map = {'x': 0, 'y': 1, 'z': 2}
        fixed_dim = dim_map[fixed_dim.lower()]

    # Indices of the two dimensions to discretize
    free_dims = [i for i in range(3) if i != fixed_dim]

    # Create ranges for the free dimensions
    ranges = [np.arange(bounds[d, 0], bounds[d, 1] + resolution, resolution) for d in free_dims]
    grid_1, grid_2 = np.meshgrid(*ranges)

    # Get output shape (H, W)
    shape = grid_1.shape
    H, W = shape

    # Initialize output grid (H, W, 3)
    grid = np.zeros((H, W, 3))
    grid[..., free_dims[0]] = grid_1
    grid[..., free_dims[1]] = grid_2
    grid[..., fixed_dim] = fixed_value

    return grid


def normalize_feature_grid(grid):
    assert grid.ndim == 3
    H, W, F = grid.shape
    x = np.reshape(grid, (H * W, F))
    row_norms = np.linalg.norm(x, axis=1, keepdims=True)    
    row_norms = np.where(row_norms < 1e-12, 1e-8, row_norms)    
    x_nrm = x / row_norms
    return np.reshape(x_nrm, (H, W, F))


def feature_grid_to_rgb(grid, normalize=False):
    assert grid.ndim == 3
    # TODO: generalize to 3D
    H, W, F = grid.shape
    if normalize:
        grid = normalize_feature_grid(grid)
    x = np.reshape(grid, (H * W, F))
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(x)
    scaler = MinMaxScaler()
    normalized_vectors = scaler.fit_transform(reduced_vectors)
    return np.reshape(normalized_vectors, (H, W, 3))


def visualize_feature_grid(grid, mode='pca', normalize=False, fig_path=None):
    fig, ax = plt.subplots(layout='compressed')
    if normalize:
        grid = normalize_feature_grid(grid)
    if mode == 'pca':
        grid_rgb = feature_grid_to_rgb(grid, normalize=False)
    elif mode == 'simple':
        grid_rgb = grid[:, :, :3]
    else:
        raise ValueError(f"Invalid mode: {mode}!")
    ax.imshow(grid_rgb)
    ax.axis('off')
    # ax.invert_yaxis()
    ax.set_title(f"Feature (mode: {mode}, normalized: {normalize}, shape: {grid.shape})")
    if fig_path is not None:
        plt.savefig(fig_path)
        plt.close(fig)
    else:
        plt.show()


def visualize_grid_magnitude(grid, fig_path=None, log_scale=False):
    assert grid.ndim == 3, "Only 2D grid is supported."
    fig, ax = plt.subplots(layout='compressed')
    grid_norm = np.linalg.norm(grid, axis=2)
    if log_scale:
        grid_norm = np.log10(grid_norm)
        label = "Magnitudes (log10)"
    else:
        label = "Magnitudes"
    im = ax.imshow(grid_norm, cmap='plasma')
    plt.colorbar(im)
    ax.axis('off')
    ax.invert_yaxis()
    ax.set_title(label)
    if fig_path is not None:
        plt.savefig(fig_path)
        plt.close(fig)
    else:
        plt.show()


def visualize_grid_scalar(grid, fig_path=None, cmap='seismic', bounds=None, 
                          show_title=True, show_colorbar=True, hide_axis=False):
    assert grid.ndim == 2, "Only 2D grid is supported."
    fig, ax = plt.subplots(layout='compressed')
    if cmap == 'seismic':
        if bounds is None:
            rmax = np.max(grid)
            rmin = np.min(grid)
        else:
            rmin, rmax = bounds
        cmap = plt.get_cmap('seismic')  # or use 'seismic', 'bwr', 'PiYG', etc.
        try:
            norm = colors.TwoSlopeNorm(vmin=rmin, vcenter=0, vmax=rmax)
        except:
            logger.debug(f"Failed to setup colormap! sdf_min={rmin}, sdf_max={rmax}")
            norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        im = ax.imshow(grid, cmap=cmap, norm=norm)
    else:
        im = ax.imshow(grid, cmap=cmap)
    if show_colorbar:
        plt.colorbar(im)
    # ax.axis('off')
    ax.invert_yaxis()
    if show_title:
        ax.set_title(f"Grid values (shape: {grid.shape})")
    if hide_axis:
        plt.axis('off')
    if fig_path is not None:
        plt.savefig(fig_path)
        plt.close(fig)
    else:
        plt.show()


###############################
#####    Other Utils     ######
###############################

def cond_mkdir(path):
    if not os.path.exists(path):
        logger.info(f"Create directory: {path}")
        os.makedirs(path)

def check_tensor(tensor):
    num_inf = torch.isinf(tensor).sum().item()
    num_nan = torch.isnan(tensor).sum().item()
    if num_nan > 0:
        raise ValueError(f"Tensor has {num_nan} nan values!")
    if num_inf > 0:
        raise ValueError(f"Tensor has {num_inf} inf values!")

def sanitize_tensor_dict(input_dict):
    for key, val in input_dict.items():
        num_nan = torch.isnan(val).sum().item()
        if num_nan > 0:
            logger.warning(f"Tensor {key} has {num_nan}/{val.numel()} nan values!")
            input_dict[key] = torch.nan_to_num(val)
    return input_dict

def get_batch(data_loader, device='cuda:0'):
    for step, (model_input, gt) in enumerate(data_loader):
        model_input, gt = prepare_batch(model_input, gt, device)
        return model_input, gt

def prepare_batch(model_input, gt, device='cuda:0'):
    model_input = {key: value.to(device) for key, value in model_input.items()}
    gt = {key: value.to(device) for key, value in gt.items()}
    model_input = sanitize_tensor_dict(model_input)
    gt = sanitize_tensor_dict(gt)
    return model_input, gt

def relative_param_change(params_curr, params_prev=None):
    if params_prev is None:
        return np.inf
    num_sq = 0
    den_sq = 0
    for idx in range(len(params_curr)):
        num_sq += torch.sum((params_curr[idx] - params_prev[idx])**2)
        den_sq += torch.sum((params_prev[idx])**2)
    relchange = torch.sqrt(num_sq / den_sq).item()
    return relchange

class PerfTimer():
    def __init__(self, activate=False):
        self.prev_time = time.process_time()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.prev_time_gpu = self.start.record()
        self.counter = 0
        self.activate = activate

    def reset(self):
        self.counter = 0
        self.prev_time = time.process_time()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.prev_time_gpu = self.start.record()

    def check(self, name=None):
        if self.activate:
            cpu_time = time.process_time() - self.prev_time
          
            self.end.record()
            torch.cuda.synchronize()

            gpu_time = self.start.elapsed_time(self.end) / 1e3
            # if name:
            #     print("CPU Checkpoint {}: {} s".format(name, cpu_time))
            #     print("GPU Checkpoint {}: {} s".format(name, gpu_time))
            # else:
            #     print("CPU Checkpoint {}: {} s".format(self.counter, cpu_time))
            #     print("GPU Checkpoint {}: {} s".format(self.counter, gpu_time))

            self.prev_time = time.process_time()
            self.prev_time_gpu = self.start.record()
            self.counter += 1
            return cpu_time, gpu_time


class InfoNCE(torch.nn.Module):
    def __init__(self, temperature=0.07, reduction='mean'):
        """
        InfoNCE Loss module.

        Args:
            temperature (float): Temperature parameter for scaling the similarity.
            reduction (str): Specifies the reduction to apply to the output: 'mean' or 'sum'.
        """
        super(InfoNCE, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.criterion = torch.nn.CrossEntropyLoss(reduction=self.reduction)
    
    def forward(self, query, key):
        """
        Args:
            query (torch.Tensor): Query embeddings of shape (batch_size, feature_dim)
            key (torch.Tensor):   Key embeddings of shape (batch_size, feature_dim)
        
        Returns:
            torch.Tensor: Scalar InfoNCE loss.
        """
        # Compute similarity (dot product) between query and all keys: shape (batch_size, batch_size)
        logits = torch.matmul(query, key.T)
        
        # Scale the similarities by temperature
        logits = logits / self.temperature
        
        batch_size = query.shape[0]
        labels = torch.arange(batch_size, device=query.device)
        
        # Compute cross-entropy loss
        loss = self.criterion(logits, labels)
        
        return loss