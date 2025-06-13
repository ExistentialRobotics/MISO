import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import trimesh
import plotly.graph_objects as go
from pytorch3d.ops import knn_points
from typing import Tuple, Dict

from ..base_net import BaseNet  


class FourierPositionalEncoding(nn.Module):
    def __init__(self, num_freqs: int = 10, max_freq: float = 10.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.max_freq = max_freq
        self.freq_bands = 2.0 ** torch.linspace(0, np.log2(self.max_freq), self.num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for freq in self.freq_bands.to(x.device):
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        layers = []
        # Input layer
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        # Hidden layers
        for _ in range(self.num_layers - 2):
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        # Output layer
        layers.append(nn.LayerNorm(self.hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PointSDF(BaseNet):
    def __init__(
        self, 
        cfg: Dict, 
        meshfile: str = '/home/sunghwan/data/Replica/room_0/mesh.ply', 
        device: str = 'cuda:0', 
        dtype: torch.dtype = torch.float32
    ):
        super(PointSDF, self).__init__(cfg, device, dtype)

        np.random.seed(42)
        torch.manual_seed(42)

        # Configuration
        self.meshfile = meshfile
        self.total_samples = cfg['point']['total_samples']
        self.noise_threshold = cfg['point']['noise_threshold']
        self.sample_ratio_surface = cfg['point']['sample_ratio_surface']
        self.sample_ratio_random = cfg['point']['sample_ratio_random']
        self.feature_dim = cfg['point']['feature_dim']
        self.k_neighbors = cfg['point']['k_neighbors']
        self.resolution = cfg['point']['resolution']
        self.hash_table_size = cfg['point']['hash_table_size']
        self.hash_voxel_on = cfg['point']['hash_voxel_on']
        self.sinusoidal_pe = cfg['decoder']['sinusoidal_pe']
        self.hidden_dim = cfg['decoder']['hidden_dim']
        self.num_layers = cfg['decoder']['num_layers']
        self.output_dim = cfg['decoder']['output_dim']
        self.num_nei_cells = cfg['point']['num_nei_cells']
        self.search_alpha = cfg['point']['search_alpha']
        self.bound_np = np.array(cfg['point']['bound'])

        self.init_poses(cfg)
        # Load mesh
        print(f"Constructing PointSDF from mesh {self.meshfile}.")
        self.mesh = trimesh.load(self.meshfile)

        # Sample surface points and features
        self.points = torch.tensor(
            self.sample_surface_with_noise(), 
            device=device, dtype=dtype
        )
        self.features = nn.Parameter(
            torch.randn(self.total_samples, self.feature_dim, device=device, dtype=dtype) * 0.01
        )

        # Positional encoding
        if self.sinusoidal_pe:
            self.pe = FourierPositionalEncoding().to(device=device, dtype=dtype)
            dummy_input = torch.zeros(1, 3, device=device, dtype=dtype)
            encoded_dim = self.pe(dummy_input).shape[-1]
        else:
            encoded_dim = 3

        # Decoder MLP
        self.decoder = MLP(self.feature_dim + encoded_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim, num_layers=self.num_layers)

        # Hash grid setup
        if self.hash_voxel_on:
            self.primes = torch.tensor([73856093, 19349669, 83492791], device=device, dtype=torch.int64)
            self.set_search_neighborhood(num_nei_cells=self.num_nei_cells, search_alpha=self.search_alpha )
            self.build_hash_grid()

    def init_poses(self, cfg):
        self.num_frames = cfg['pose']['num_frames']
        self.optimize_pose = cfg['pose']['optimize']
        self.rotation_corrections = torch.nn.Parameter(
            torch.zeros(self.num_frames, 3).float().to(self.device),
            requires_grad=self.optimize_pose
        )
        self.translation_corrections = torch.nn.Parameter(
            torch.zeros(self.num_frames, 3, 1).float().to(self.device),
            requires_grad=self.optimize_pose
        )
        # logger.info(f"Initialized {self.num_frames} pose variables, optimize={self.optimize_pose}.")
    

    def set_search_neighborhood(self, num_nei_cells: int = 2, search_alpha: float = 1.0):
        device = self.points.device
        dx_range = torch.arange(-num_nei_cells, num_nei_cells + 1, device=device, dtype=torch.int64)
        coords = torch.stack(torch.meshgrid(dx_range, dx_range, dx_range, indexing="ij"), dim=-1)  
        dx2 = torch.sum(coords**2, dim=-1)
        self.neighbor_dx = coords[dx2 < (num_nei_cells + search_alpha)**2]
        self.max_valid_dist2 = 3 * ((num_nei_cells + 1) * self.resolution)**2

    def build_hash_grid(self):
        device = self.points.device
        self.buffer_pt_index = torch.full((self.hash_table_size,), -1, dtype=torch.long, device=device)
        grid_coords = torch.floor(self.points / self.resolution).to(torch.int64)
        hash_vals = torch.fmod((grid_coords * self.primes).sum(-1), self.hash_table_size)

        collisions_mask = (self.buffer_pt_index[hash_vals] != -1)
        collisions_count = collisions_mask.sum().item()
        
        if collisions_count > 0:
            print(f"[build_hash_grid] Found {collisions_count} collisions out of {len(hash_vals)} total points.")

        unique_mask = (self.buffer_pt_index[hash_vals] == -1)
        self.buffer_pt_index[hash_vals[unique_mask]] = torch.nonzero(unique_mask).squeeze(-1)
        self.grid_coords = grid_coords

    def query_neighbors_from_hash(self, query_points):
        # query_points: [B, 3]
        device = query_points.device
        B = query_points.shape[0]

        query_grid = torch.floor(query_points / self.resolution).to(torch.int64) # [B,3]

        neighbor_cells = query_grid.unsqueeze(1) + self.neighbor_dx.unsqueeze(0)
        hash_vals = torch.fmod((neighbor_cells * self.primes).sum(-1), self.hash_table_size)

        neighb_idx = self.buffer_pt_index[hash_vals]

        invalid_mask = (neighb_idx == -1)

        neighb_points = torch.empty(B, self.neighbor_dx.shape[0], 3, device=device, dtype=self.points.dtype)
        neighb_points[~invalid_mask] = self.points[neighb_idx[~invalid_mask]]
        neighb_points[invalid_mask] = 1e12 

        diff = query_points.unsqueeze(1) - neighb_points
        dist2 = torch.sum(diff * diff, dim=-1)

        sorted_dist2, sorted_idx = torch.sort(dist2, dim=1)
        sorted_dist2 = sorted_dist2[:, :self.k_neighbors]
        sorted_idx = sorted_idx[:, :self.k_neighbors]

        nn_idx = neighb_idx.gather(1, sorted_idx)

        dist_vals = torch.sqrt(sorted_dist2)

        return dist_vals, nn_idx
    
    def forward(self, x, noise_std=None):
        # x: [B, 3]
        B = x.shape[0]

        # k-NN
        if self.hash_voxel_on:
            dist_vals, nn_idx = self.query_neighbors_from_hash(x)
        else:
            point_cloud = self.points.unsqueeze(0)  # [1, N, 3]
            query_points = x.unsqueeze(0)           # [1, B, 3]
            knn = knn_points(query_points, point_cloud, K=self.k_neighbors)
            nn_idx = knn.idx[0]  # [B, k]
            dist_vals = knn.dists[0].sqrt()  # [B, k]

        neighbor_points = self.points[nn_idx] # [B, k, 3]
        neighbor_features = self.features[nn_idx] # [B, k, feature_dim]

        diff = x[:, None, :] - neighbor_points # [B, k, 3]

        if self.sinusoidal_pe:
            diff_encoding = self.pe(diff.view(-1, 3)).view(B, self.k_neighbors, -1) 
        else:
            diff_encoding = diff

        concatenated_inputs = torch.cat([neighbor_features, diff_encoding], dim=-1) # [B, k, feature_dim + encoded_dim]

        # decoder
        sdf_values = self.decoder(concatenated_inputs.view(B*self.k_neighbors, -1))
        sdf_values = sdf_values.view(B, self.k_neighbors, 1)

        weights = 1.0 / (dist_vals + 1e-8)  # [B, k]
        weights = weights / torch.sum(weights, dim=1, keepdim=True)
        interpolated_sdf_values = torch.sum(sdf_values * weights.unsqueeze(-1), dim=1) # [B, 1]

        if noise_std is not None:
            noise = torch.randn(interpolated_sdf_values.shape, device=x.device) * noise_std
            interpolated_sdf_values = interpolated_sdf_values + noise

        return interpolated_sdf_values

    def sample_surface_with_noise(self):
        surface_count = int(self.total_samples * self.sample_ratio_surface)
        near_surface_count = int(self.total_samples * self.sample_ratio_surface)
        random_count = int(self.total_samples * self.sample_ratio_random)

        surface_points = self.mesh.sample(surface_count)
        surface_near_points = self.mesh.sample(near_surface_count)

        noise = np.random.normal(
            loc=0.0,
            scale=self.noise_threshold,
            size=surface_near_points.shape
        )
        surface_near_points = surface_near_points + noise

        random_points = np.zeros((random_count, 3))
        for i in range(3):
            random_points[:, i] = np.random.uniform(
                low=self.bound_np[i, 0],
                high=self.bound_np[i, 1],
                size=random_count
            )

        all_points = np.concatenate([surface_points, surface_near_points, random_points], axis=0)
        return all_points