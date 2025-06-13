from os.path import join
import sys
import numpy as np
import torch
import torch.nn.functional as F
import grid_opt.utils.utils as utils
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sys.path.append(join("third_party", "cuda_gridsample_grad2"))

class FeatureGridBase(torch.nn.Module):
    def __init__(self, d, fdim, bound, cell_size, name="grid", dtype=torch.float32):
        super().__init__()
        assert d == 2 or d == 3
        self.d = d
        self.fdim = fdim
        self.bound = bound
        self.cell_size = cell_size
        self.dtype = dtype
        self.name = name
        assert self.bound.shape == (d, 2)
    def interpolate(self, x):
        raise NotImplementedError
    def norm(self):
        raise NotImplementedError
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def lock(self):
        #logger.debug(f"Locked grid: {self.name}.")
        for param in self.parameters():
            param.requires_grad = False
    def unlock(self):
        #logger.debug(f"Unlocked grid: {self.name}.")
        for param in self.parameters():
            param.requires_grad = True
    def zero_features(self):
        raise NotImplementedError


class FeatureGrid(FeatureGridBase):
    """Regular (dense) 2D/3D feature grid, where each voxel stores a latent feature of
    dimension fdim.
    """
    def __init__(self, d, fdim, bound, cell_size, name="grid", dtype=torch.float32, initial_feature=None, init_stddev=0.0, second_order_grid_sample=False):
        super().__init__(d=d, fdim=fdim, bound=bound, cell_size=cell_size, name=name, dtype=dtype)
        grid_len = (self.bound[:,1] - self.bound[:,0]).cpu().numpy()
        grid_size = np.ceil(grid_len / cell_size).astype(int)
        # Initialize 2D or 3D feature grid
        # Notice that we change the order of grid_size due to the behavior of grid_sample
        # See discussion here: https://discuss.pytorch.org/t/surprising-convention-for-grid-sample-coordinates/79997
        # In short, x-axis is along the width of an image and the y-axis is along its height.
        # This is the opposite of the coordinate system used in our implementation.
        if self.d == 2:
            feature_shape = (1, self.fdim, grid_size[1], grid_size[0])
        else:
            feature_shape = (1, self.fdim, grid_size[2], grid_size[1], grid_size[0])
        init_avail = initial_feature is not None
        if initial_feature is None:
            initial_feature = torch.randn(feature_shape, dtype=self.dtype) * init_stddev
        assert initial_feature.shape == feature_shape
        self.feature = torch.nn.Parameter(initial_feature)
        if second_order_grid_sample:
            # Use grid_sample that supports 2nd order derivatives
            import cuda_gridsample as cu
            self.grid_sample_func = cu.grid_sample_3d if self.d == 3 else cu.grid_sample_2d
        else:
            # Use standard pytorch implementation
            self.grid_sample_func = F.grid_sample
        logger.info(f"Regular {self.name}: cell_size={self.cell_size:.2f}, feature shape={feature_shape}, init_avail={init_avail}, 2nd_order_sample={second_order_grid_sample}, init_norm={initial_feature.norm():.1e}.")

    def interpolate(self, x):
        # Normalize query coordinates (required by F.grid_sample)
        x = utils.normalize_coordinates(x, self.bound)
        if self.d == 2:
            N = x.shape[0]
            sample_coords = x.reshape(1, N, 1, 2)
            feats = self.grid_sample_func(
                self.feature,
                sample_coords, 
                align_corners=False,
                padding_mode='zeros'
            )[0, :, :, 0].transpose(0, 1)
        else:
            N = x.shape[0]
            sample_coords = x.reshape(1, N, 1, 1, 3)
            # print('feature device:', self.feature.device)
            # print('sample coords device:', sample_coords.device)
            feats = self.grid_sample_func(
                self.feature,
                sample_coords,
                align_corners=False,
                padding_mode='zeros'
            )[0, :, :, 0, 0].transpose(0, 1)
        return feats

    def norm(self):
        return self.feature.norm()
    
    def zero_features(self):
        logger.info(f"Set grid features to zero: {self.name}.")
        with torch.no_grad():
            self.feature.copy_(torch.zeros_like(self.feature))
    
    def randn_features(self, std):
        logger.info(f"Set grid features to randn: {self.name}, std={std:.2e}.")
        with torch.no_grad():
            new_feat = torch.randn(self.feature.shape, dtype=self.dtype) * std
            self.feature.copy_(new_feat.to(self.feature))

    def vertex_positions(self, denormalize=True) -> torch.Tensor:
        """
        Args:
            denormalize (bool, optional): Whether to denormalize positions. Defaults to True.
        Returns:
            torch.Tensor: The 3D positions of grid vertices, unnormalized by default. 
        """
        pos_nrm = utils.all_grid_positions(self.feature)
        pos_nrm = torch.flatten(pos_nrm.squeeze(), start_dim=0, end_dim=-2)  # N,3
        if denormalize:
            return utils.denormalize_coordinates(pos_nrm, self.bound.to(pos_nrm))
        else:
            return pos_nrm

class FeatureGridVM(FeatureGridBase):
    """3D feature grid store using low-rank VM factorization.
    Ref:
    Chen et al., TensoRF: Tensorial Radiance Fields
    """
    def __init__(self, d, fdim, bound, cell_size, name="grid", dtype=torch.float32, rank=10, init_stddev=0.0):
        super().__init__(d=d, fdim=fdim, bound=bound, cell_size=cell_size, name=name, dtype=dtype)
        assert self.d == 3, "VM factorization only supports 3D grid!"
        self.rank = rank
        grid_len = (self.bound[:,1] - self.bound[:,0]).cpu().numpy()
        grid_size = np.ceil(grid_len / cell_size).astype(int)
        xsize, ysize, zsize = grid_size
        # XY + Z decomp
        self.feats_XY = torch.nn.Parameter(torch.randn((1, self.rank, ysize, xsize), dtype=self.dtype) * init_stddev)
        self.feats_Z = torch.nn.Parameter(torch.randn((1, self.rank, zsize, 1), dtype=self.dtype) * init_stddev)
        # XZ + Y decomp
        self.feats_XZ = torch.nn.Parameter(torch.randn((1, self.rank, zsize, xsize), dtype=self.dtype) * init_stddev)
        self.feats_Y = torch.nn.Parameter(torch.randn((1, self.rank, ysize, 1), dtype=self.dtype) * init_stddev)
        # YZ + X decomp
        self.feats_YZ = torch.nn.Parameter(torch.randn((1, self.rank, zsize, ysize), dtype=self.dtype) * init_stddev)
        self.feats_X = torch.nn.Parameter(torch.randn((1, self.rank, xsize, 1), dtype=self.dtype) * init_stddev)
        logger.info(f"VM-{self.name}: cell_size={self.cell_size}, rank={self.rank}, num_param={self.num_params()}.")
    

    def _interpolate_helper(self, feats, x_nrm):
        N, d = x_nrm.shape
        assert d == 1 or d == 2
        assert feats.ndim == 4
        if d == 1:
            # Pad query coordinates with a zero column
            # to lift to 2d coordinates
            x_nrm = torch.concat((torch.zeros_like(x_nrm), x_nrm), dim=1)
            assert feats.shape[-1] == 1
        assert x_nrm.shape == (N, 2)
        sample_coords = x_nrm.reshape(1, N, 1, 2)
        feats = F.grid_sample(
            feats,
            sample_coords, 
            align_corners=False,
            mode='bilinear',
            padding_mode='zeros'
        )[0, :, :, 0].transpose(0, 1)
        return feats


    def interpolate(self, x):
        # Normalize query coordinates (required by F.grid_sample)
        x = utils.normalize_coordinates(x, self.bound)

        # Interpolate on XY+Z factors
        coeffs_xy = self._interpolate_helper(self.feats_XY, x[:, [0,1]])
        coeffs_z = self._interpolate_helper(self.feats_Z, x[:, [2]])
        coeffs_xy_z = coeffs_xy * coeffs_z  # (N, R)
        
        # Interpolate on XZ+Y factors
        coeffs_xz = self._interpolate_helper(self.feats_XZ, x[:, [0,2]])
        coeffs_y = self._interpolate_helper(self.feats_Y, x[:, [1]])
        coeffs_xz_y = coeffs_xz * coeffs_y

        # Interpolate on YZ+X factors
        coeffs_yz = self._interpolate_helper(self.feats_YZ, x[:, [1,2]])
        coeffs_x = self._interpolate_helper(self.feats_X, x[:, [0]])
        coeffs_yz_x = coeffs_yz * coeffs_x

        coeffs_dict = {
            'xy_z': coeffs_xy_z,
            'xz_y': coeffs_xz_y,
            'yz_x': coeffs_yz_x
        }

        return coeffs_dict
    
    
    def norm(self):
        # TODO:
        logger.warning("norm for FeatureGridVM is not implemented!")
        return -1
    

    def visualize_planes(self, prefix_path):
        feats_XY = self.feats_XY.squeeze().permute([2, 1, 0])
        grid = feats_XY[:, :, 0].detach().cpu().numpy()
        utils.visualize_grid_scalar(grid, fig_path=prefix_path+"vm_factor_xy.png")

        feats_XZ = self.feats_XZ.squeeze().permute([2, 1, 0])
        grid = feats_XZ[:, :, 0].detach().cpu().numpy()
        utils.visualize_grid_scalar(grid, fig_path=prefix_path+"vm_factor_xz.png")

        feats_YZ = self.feats_YZ.squeeze().permute([2, 1, 0])
        grid = feats_YZ[:, :, 0].detach().cpu().numpy()
        utils.visualize_grid_scalar(grid, fig_path=prefix_path+"vm_factor_yz.png")


class BasisVM(torch.nn.Module):
    def __init__(self, fdim, name="basis", dtype=torch.float32, rank=10, init_stddev=0.0, pretrained_path=None, no_optimize=False):
        super().__init__()
        self.rank = rank
        self.fdim = fdim
        self.name = name
        self.B_XY_Z = torch.nn.Parameter(torch.randn((self.fdim, self.rank), dtype=dtype) * init_stddev)
        self.B_XZ_Y = torch.nn.Parameter(torch.randn((self.fdim, self.rank), dtype=dtype) * init_stddev)
        self.B_YZ_X = torch.nn.Parameter(torch.randn((self.fdim, self.rank), dtype=dtype) * init_stddev)
        logger.info(f"VM-Basis-{self.name}: rank={self.rank}.")
        if pretrained_path is not None:
            self.load(pretrained_path)
        if no_optimize:
            logger.info(f"Fixing BasisVM weight.")
            for param in self.parameters():
                param.requires_grad = False
        else:
            logger.info(f"BasisVM is optimizable.")

    def forward(self, coeffs_dict):
        # coeffs: (N, R)
        feats_xy_z = coeffs_dict['xy_z'] @ self.B_XY_Z.T  # (N, fdim)
        feats_xz_y = coeffs_dict['xz_y'] @ self.B_XZ_Y.T 
        feats_yz_x = coeffs_dict['yz_x'] @ self.B_YZ_X.T
        return feats_xy_z + feats_xz_y + feats_yz_x
    
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        ckpt = torch.load(filepath)
        self.load_state_dict(ckpt)
        logger.info(f"Loaded pretrained BasisVM from {filepath}.")

