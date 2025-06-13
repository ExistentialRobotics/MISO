import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import grid_opt.utils.utils as utils
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MLPNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, hidden_layers=1, bias=False, acti_func=nn.ReLU, pretrained_path=None, no_optimize=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = [nn.Linear(input_dim, hidden_dim, bias=bias), acti_func()]
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            self.layers.append(acti_func())
        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))
        self.network = nn.Sequential(*self.layers)
        if pretrained_path is not None:
            self.load(pretrained_path)
        if no_optimize:
            logger.debug(f"Fixing MLP weight.")
            for param in self.parameters():
                param.requires_grad = False
        else:
            logger.debug(f"MLP is optimizable.")

    def forward(self, x):
        return self.network(x)
    
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        ckpt = torch.load(filepath)
        self.load_state_dict(ckpt)
        logger.info(f"Loaded pretrained decoder from {filepath}.")


class LinearNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_layer = nn.Linear(input_dim, output_dim, bias=False)
        with torch.no_grad():
            self.linear_layer.weight.copy_(
                torch.tensor(np.ones((output_dim, input_dim)), dtype=torch.float32)
            )


    def forward(self, x):
        return self.linear_layer(x)
    

class FeatureUpsampler(torch.nn.Module):
    """Implement upsampler of 2D/3D feature grid.
    TODO: Allow more flexible configurations.
    """
    def __init__(self, dim, in_channels, out_channels):
        super().__init__()
        self.d = dim
        if dim == 2:
            UpSampleModule = torch.nn.ConvTranspose2d
        elif dim == 3:
            UpSampleModule = torch.nn.ConvTranspose3d
        else:
            raise ValueError(f"Invalid dimension: {dim}!")
        # For now, upsample by a fixed factor of 2
        self.upsampler = UpSampleModule(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.refiner = MLPNet(input_dim=out_channels, output_dim=out_channels, hidden_dim=2*out_channels, hidden_layers=1, bias=True)
    
    def forward(self, x):
        """

        Args:
            x (_type_): input feature grid.
            For 2D: has shape (1, fdim, H, W)
            For 3D: has shape (1, fdim, H, W, D)

        Returns:
            Upsampled feature grid. E.g., when factor is 2:
            For 2D: (1, outdim, 2H, 2W)
            For 3D: (1, outdim, 2H, 2D, 2D)
        """
        x = self.upsampler(x)
        x = torch.nn.functional.relu(x) 
        if self.d == 2:
            B, fdim, H, W = x.shape
            assert B == 1
            x = x.permute([0, 2, 3, 1]).reshape(-1, fdim)
            x = self.refiner(x)
            x = x.view(B, H, W, fdim).permute([0, 3, 1, 2])
        else:
            B, fdim, H, W, D = x.shape
            assert B == 1
            x = x.permute([0, 2, 3, 4, 1]).reshape(-1, fdim)
            x = self.refiner(x)
            x = x.view(B, H, W, D, fdim).permute([0, 4, 1, 2, 3])

        return x


class ConvInterp(torch.nn.Module):
    """Implements multiple layers of 2D/3D convolution.
    Then, the final result is upsampled/downsampled to a desired output spatial size 
    using interpolation.
    """
    def __init__(self, 
                 dim, 
                 in_channels, 
                 base_channels=4, 
                 hidden_layers=2, 
                 kernel_size=3, 
                 padding=1, 
                 reduction_factor=2, 
                 dtype=torch.float32, 
                 device='cuda:0', 
                 name='ConvInterp'):
        super().__init__()
        self.name = name
        self.d = dim
        if self.d == 2:
            PoolModule = torch.nn.MaxPool2d
            ConvModule = torch.nn.Conv2d
        elif self.d == 3:
            PoolModule = torch.nn.MaxPool3d
            ConvModule = torch.nn.Conv3d
        else:
            raise ValueError(f"Invalid dimension: {self.d}!")
        if reduction_factor > 1:
            self.pool = PoolModule(kernel_size=reduction_factor, stride=reduction_factor)
        else:
            self.pool = None
        self.conv_layers = nn.ModuleList()
        for layer_index in range(hidden_layers):
            in_ch = in_channels if layer_index == 0 else base_channels * (2**(layer_index-1))
            out_ch = base_channels * (2**layer_index)
            self.conv_layers.append(
                ConvModule(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, dtype=dtype, device=device)
            )
        self.output_channels = out_ch
        

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = torch.nn.functional.relu(x)
            if self.pool is not None:
                x  = self.pool(x)
        return x


    def forward_and_interpolate(self, x, output_spatial_size):
        x = self.forward(x)
        # Interpolate from (B, C, H', W') 
        # to final desired spatial size (H, W)
        input_spatial_size = x.shape[2:]
        in_size = torch.tensor(input_spatial_size, dtype=torch.float32)
        out_size = torch.tensor(output_spatial_size, dtype=torch.float32)
        if torch.all(in_size <= out_size):
            # Upsampling
            interp_mode = 'bilinear' if self.d == 2 else 'trilinear'
            align_corners = False
        elif torch.all(in_size > out_size):
            # Downsampling
            interp_mode = 'area'
            align_corners=None
        else:
            raise ValueError(f"Invalid input and output size! input_size={input_spatial_size}, output_size={output_spatial_size}. ")
        logger.debug(f"{self.name}: input_size={input_spatial_size}, output_size={output_spatial_size}, interp={interp_mode}, align_corners={align_corners}.")
        x = torch.nn.functional.interpolate(
            input=x,
            size=output_spatial_size,
            mode=interp_mode,
            align_corners=align_corners
        )
        return x
    

class FeatureReduction3D(nn.Module):
    """Given a 3D volume of features of shape (1, C, H, W, D) 
    where 1 is batch size, C is the feature dimension, and H, W, D are spatial dimensions
    reduce 1 or 2 of the spatial dimensions using a chosen reduction operation
    followed by a MLP transformation.
    
    E.g. if reduce_dims = [2,3], the output dim will be (1, C', D) where C' is the output dim of the MLP.
    """
    def __init__(self, 
        reduce_dims,
        reduce_op='max',
        input_dim=8,
        output_dim=8,
        mlp_hidden_layers=1,
        mlp_hidden_dim=8, 
        name='FeatureReduction3D'
    ):
        super().__init__()
        self.name = name
        for dim in reduce_dims:
            assert dim >= 2, f"Invalid reduction dim: {dim}!"
        self.reduce_dims = reduce_dims
        self.reduce_op = reduce_op
        self.mlp = MLPNet(
            input_dim=input_dim,
            output_dim=output_dim, 
            hidden_dim=mlp_hidden_dim, 
            hidden_layers=mlp_hidden_layers, 
            bias=True
        )
    
    def forward(self, x):
        assert x.ndim == 5
        assert x.shape[0] == 1
        assert x.shape[1] == self.mlp.input_dim
        if self.reduce_op == 'max':
            x = torch.amax(x, dim=self.reduce_dims, keepdim=True)
        elif self.reduce_op == 'mean':
            x = torch.mean(x, dim=self.reduce_dims, keepdim=True)
        else:
            raise ValueError(f"Invalid reduce op: {self.reduce_op}!")
        # e.g., x now (1, C, H, 1, D)
        x = x.squeeze(dim=0).permute([1,2,3,0])  # (H, 1, D, C)
        desired_dim = x.shape[:-1] + (self.mlp.output_dim, )
        x = x.reshape(-1, self.mlp.input_dim)
        x = self.mlp(x).reshape(*desired_dim)  # (H, 1, D, C')
        x = x.permute([3,0,1,2]).squeeze()  # (C', H, D)
        return x.unsqueeze(0)   # (1, C', H, D)
    


class FeaturePrediction(nn.Module):
    def __init__(self, d, fdim, rdim=1, feature_processor=True, residual_processor=True, normalize_output=False, device='cuda:0', initial_param_std=None):
        super().__init__()
        self.d = d
        mlp_input_dim = 0
        if feature_processor:
            self.feature_processor = ConvInterp(
                dim=d,
                in_channels=fdim, 
                reduction_factor=1,
                hidden_layers=2,
                name=f'feature_proc_{d}D'
            ).to(device)
            mlp_input_dim += self.feature_processor.output_channels
        else:
            self.feature_processor = None
        
        if residual_processor: 
            self.residual_processor = ConvInterp(
                dim=d,
                in_channels=rdim, 
                reduction_factor=1,
                hidden_layers=2,
                name=f'residual_proc_{d}D'
            ).to(device)
            mlp_input_dim += self.residual_processor.output_channels
        else:
            self.residual_processor = None

        self.mlp = MLPNet(
            input_dim=mlp_input_dim, 
            output_dim=fdim, 
            hidden_dim=16, 
            hidden_layers=2, 
            bias=True
        ).to(device)

        self.normalize_output = normalize_output
        
        if initial_param_std is not None:
            logger.info(f"Initialize feature prediction params with std {initial_param_std}.")
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.normal_(m.weight, std=initial_param_std)
                    torch.nn.init.normal_(m.bias, std=initial_param_std)
            self.apply(init_weights)
    
    def predict(self, coarse_features: torch.Tensor, coarse_residuals: torch.Tensor, output_spatial_size):
        if self.d == 2:
            return self._predict_2d(coarse_features, coarse_residuals, output_spatial_size)
        elif self.d == 3:
            return self._predict_3d(coarse_features, coarse_residuals, output_spatial_size)
        else:
            raise ValueError(f"Invalid dimension: {self.d}!")

    
    def _predict_3d(self, coarse_features: torch.Tensor, coarse_residuals: torch.Tensor, output_spatial_size):
        """Predict feature from coarse-level features and residuals

        Args:
            coarse_features (torch.Tensor): (1, fdim, H1, W1, D1)
            coarse_residuals (torch.Tensor): (1, 1, H2, W2, D2)
        Output:
            predicted fine features (1, fdim, H, W, D) where (H, W, D) is specified by target_size arg.
        """
        feats_to_concat = []
        if self.feature_processor is not None:
            output_from_features = self.feature_processor.forward_and_interpolate(coarse_features, output_spatial_size)
            output_from_features = output_from_features.squeeze().permute([1,2,3,0]) # H, W, D, F  
            H, W, D, C_feat = output_from_features.shape
            m_from_feats = output_from_features.reshape(-1, C_feat)
            feats_to_concat.append(m_from_feats)
        if self.residual_processor is not None:
            output_from_residual = self.residual_processor.forward_and_interpolate(coarse_residuals, output_spatial_size)
            output_from_residual = output_from_residual.squeeze().permute([1,2,3,0])
            H, W, D, C_res = output_from_residual.shape
            m_from_residuals = output_from_residual.reshape(-1, C_res)
            feats_to_concat.append(m_from_residuals)
        assert len(feats_to_concat) > 0, "Input to MLP is empty! "
        embeddings_flat = self.mlp(torch.concat(feats_to_concat, dim=1))
        if self.normalize_output:
            embeddings_flat = utils.normalize_last_dim(embeddings_flat)
        embeddings = embeddings_flat.reshape(H, W, D, -1)
        embeddings = embeddings.permute([3, 0, 1, 2]).unsqueeze(0)  # (1, C, H, W, D)
        return embeddings
    

    def _predict_2d(self, coarse_features: torch.Tensor, coarse_residuals: torch.Tensor, output_spatial_size):
        """Predict feature from coarse-level features and residuals

        Args:
            coarse_features (torch.Tensor): (1, fdim, H1, W1)
            coarse_residuals (torch.Tensor): (1, 1, H2, W2)
        Output:
            predicted fine features (1, fdim, H, W) where (H, W) is specified by target_size arg.
        """
        feats_to_concat = []
        if self.feature_processor is not None:
            output_from_features = self.feature_processor.forward_and_interpolate(coarse_features, output_spatial_size)
            output_from_features = output_from_features.squeeze().permute([1,2,0]) # H, W, F  
            H, W, C_feat = output_from_features.shape
            m_from_feats = output_from_features.reshape(-1, C_feat)
            feats_to_concat.append(m_from_feats)
        if self.residual_processor is not None:
            output_from_residual = self.residual_processor.forward_and_interpolate(coarse_residuals, output_spatial_size)
            output_from_residual = output_from_residual.squeeze().permute([1,2,0])
            H, W, C_res = output_from_residual.shape
            m_from_residuals = output_from_residual.reshape(-1, C_res)
            feats_to_concat.append(m_from_residuals)
        assert len(feats_to_concat) > 0, "Input to MLP is empty! "
        embeddings_flat = self.mlp(torch.concat(feats_to_concat, dim=1))
        if self.normalize_output:
            embeddings_flat = utils.normalize_last_dim(embeddings_flat)
        embeddings = embeddings_flat.reshape(H, W, -1)
        embeddings = embeddings.permute([2, 0, 1]).unsqueeze(0)  # (1, C, H, W)
        return embeddings
    

class FeaturePredictionVM(nn.Module):
    def __init__(self, d, rank, feature_processor=True, residual_processor=True, device='cuda:0'):
        super().__init__()
        assert d == 3, "VM decomposition only supports 3D!"
        self.d = d
        mlp_input_dim = 0
        if feature_processor:
            raise NotImplementedError
        else:
            self.feature_processor = None
        
        if residual_processor: 
            self.residual_processor = ConvInterp(
                dim=d,
                in_channels=1, 
                reduction_factor=1,
                hidden_layers=2,
                name=f'residual_proc_{d}D'
            ).to(device)
            mlp_input_dim += self.residual_processor.output_channels
        else:
            self.residual_processor = None

        # Initialize VM feature MLPs
        self.mlp_dict = nn.ModuleDict()
        zindex, yindex, xindex = 2, 3, 4
        self.mlp_dict['xy'] = FeatureReduction3D(reduce_dims=zindex, name='vm-xy', input_dim=mlp_input_dim, output_dim=rank)
        self.mlp_dict['xz'] = FeatureReduction3D(reduce_dims=yindex, name='vm-xz', input_dim=mlp_input_dim, output_dim=rank)
        self.mlp_dict['yz'] = FeatureReduction3D(reduce_dims=xindex, name='vm-yz', input_dim=mlp_input_dim, output_dim=rank)
        self.mlp_dict['z'] = FeatureReduction3D(reduce_dims=(xindex,yindex), name='vm-z', input_dim=mlp_input_dim, output_dim=rank)
        self.mlp_dict['y'] = FeatureReduction3D(reduce_dims=(xindex,zindex), name='vm-y', input_dim=mlp_input_dim, output_dim=rank)
        self.mlp_dict['x'] = FeatureReduction3D(reduce_dims=(yindex,zindex), name='vm-x', input_dim=mlp_input_dim, output_dim=rank)
