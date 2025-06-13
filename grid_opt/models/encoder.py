from os.path import join
import numpy as np
import torch
from torch import Tensor
from dataclasses import dataclass
from grid_opt.models.grid_net import GridNet
from grid_opt.loss import BaseLoss
import grid_opt.utils.utils_geometry as utils_geometry
import grid_opt.utils.utils_sdf as utils_sdf 
from .modules import *
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class EncoderObservation:
    """A dataclass that contains all raw SDF input data needed by the Encoder.
    """
    coords_world: Tensor   # (N, 3)
    gt_sdf: Tensor         # (N, 1)  SDF labels for observed positions 
    gt_sdf_sign: Tensor    # (N, 1)  sign labels for observed positions
    gt_sdf_valid: Tensor   # (N, 1)  binary mask on `gt_sdf` indicating valid 

class Encoder(torch.nn.Module):
    def __init__(self, cfg, pretrained_dir=None):
        """A hierarchical encoder module for neural SDF.

        Args:
            cfg (_type_): _description_
            pretrained_dir (_type_, optional): Directory containing pre-trained encoder weights. Defaults to None.
        """
        super().__init__()
        self.num_levels = cfg['model']['grid']['n_levels']
        self.fdim = cfg['model']['grid']['feature_dim']
        self.rdim = 3  # input residual feature dimension is 3
        self.trunc_dist = 0.15    # set in the pre-training and is hardcoded for now
        self.device = cfg['device']
        assert self.num_levels == 2
        # Initialize pretrained feature encoders
        self.feature_encoders = torch.nn.ModuleList()
        for level in range(self.num_levels):
            pretrained_path = join(pretrained_dir, f"feature_encoder_level_{level}.pt") if pretrained_dir is not None else None
            self.feature_encoders.append(self.init_level_encoder(pretrained_path))
        self.grid_nets = torch.nn.ModuleDict()
        self.grid_corrections = torch.nn.ParameterDict()
        self.intermediate_results = {}   # saveing intermediate results for visualization 
        # self.model = construct_grid_model_for_dataset(cfg)
        # self.corrections = construct_grid_corrections(self.model)
        # self.print_trainable_params()
        # self.print_correction_norms()

    def init_level_encoder(self, pretrained_path=None):
        """Initialize core feature prediction module at each level of the feature grid.
        """
        encoder = FeaturePrediction(d=3, fdim=self.fdim, rdim=self.rdim, feature_processor=False, residual_processor=True, device=self.device)
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path)
            encoder.load_state_dict(ckpt)
        # Default to freeze weights if loaded from file
        for param in encoder.parameters():
            param.requires_grad = False
        logger.info(f"Loaded encoder from: {pretrained_path}")
        return encoder
    
    def lock_all_params(self):
        for encoder in self.feature_encoders:
            for param in encoder.parameters():
                param.requires_grad = False
        for key, model in self.grid_nets.items():
            for level in range(model.num_levels):
                model.lock_level(level)
        for key, corr in self.grid_corrections.items():
            corr.requires_grad = False

    def unlock_encoder_at_level(self, level):
        encoder = self.feature_encoders[level]
        for param in encoder.parameters():
            param.requires_grad = True
    
    def grid_key(self, model_id):
        return f"gridnet{model_id}"
    
    def correction_key(self, model_id, level):
        return f"gridnet{model_id}_correction_level{level}"
    
    def get_grid_net(self, model_id) -> GridNet:
        return self.grid_nets[self.grid_key(model_id)]

    def get_grid_correction(self, model_id, level) -> torch.nn.Parameter:
        return self.grid_corrections[self.correction_key(model_id, level)]
    
    def register_grid_model(self, model:GridNet):
        """Register a single GridNet model to this encoder, and initialize the corresponding corrections.
        """
        model_id = len(self.grid_nets)
        model_key = self.grid_key(model_id)
        self.grid_nets[model_key] = model
        for level in range(model.num_levels):
            delta = torch.zeros_like(model.features[level].feature)
            corr_key = self.correction_key(model_id, level)
            self.grid_corrections[corr_key] = torch.nn.Parameter(delta)
        return model_id
    
    def print_trainable_params(self):
        print("=== Summary of trainable params === ")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}")
        print("=== END Summary of trainable params ===")

    def print_correction_norms(self):
        for model_id, model in self.grid_nets.items():
            print(f"Correction norms for gridnet {model_id}:")
            for level in range(model.num_levels):
                feat = model.features[level].feature
                delta = self.get_grid_correction(model_id, level)
                print(f"Level {level}: curr_norm={feat.norm():.2e}, delta_norm={delta.norm():.2e}")
            print(f"Rotation update: {model.rotation_corrections.norm():.2e}")
            print(f"Translation update: {model.translation_corrections.norm():.2e}")
    
    def stored_corrections_until_level(self, model_id: int, stop_level: int):
        """Obtain the currently stored grid corrections, up to and NOT INCLUDING the stop level.
        """
        grid = self.get_grid_net(model_id)
        input_corrections = []
        for level in range(grid.num_levels):
            level_correction = self.get_grid_correction(model_id, level)
            if level < stop_level:
                input_corrections.append(level_correction)
            else:
                input_corrections.append(torch.zeros_like(level_correction))
        return input_corrections
    
    def predict_corrections_until_level(self, model_id: int, stop_level: int, observation: EncoderObservation,
                                        pred_std=0, store_corrections=False):
        """Predict from scratch the stored grid corrections, up to and NOT INCLUDING the stop level.
        """
        grid = self.get_grid_net(model_id)
        current_corrections = [
            torch.zeros_like(self.get_grid_correction(model_id, level)) for level in range(grid.num_levels)
        ]
        for level in range(stop_level):
            input_residuals = self.compute_residuals(
                model_id=model_id, corrections=current_corrections, observation=observation,
                skip_sign=False, skip_eik=True, skip_smooth=True
            )
            encoder_inputs = self.compute_encoder_inputs_from_residuals(
                input_residuals, model_id=model_id, target_level=level, save=True
            )
            encoder_outputs = self.compute_encoder_outputs(
                model_id=model_id, encoder_inputs=encoder_inputs, target_level=level
            )
            # Add random perturbations to the encoder predicted features
            encoder_outputs += torch.normal(mean=0., std=pred_std, size=encoder_outputs.shape).to(encoder_outputs)
            assert current_corrections[level].shape == encoder_outputs.shape
            current_corrections[level] = encoder_outputs
        
        if store_corrections:
            with torch.no_grad():
                for level in range(grid.num_levels):
                    corr_key = self.correction_key(model_id, level)
                    assert self.grid_corrections[corr_key].shape == current_corrections[level].shape
                    self.grid_corrections[corr_key].copy_(current_corrections[level])
        return current_corrections

    def query_sdf(self, model: GridNet, corrections, x):
        bound_torch = model.bound
        x_nrm = utils.normalize_coordinates(x, bound_torch)
        updated_grids = []
        for level in range(model.num_levels):
            updated_grids.append(model.features[level].feature + corrections[level])
        feats = utils.interp_3d(updated_grids, x_nrm, second_order_grid_sample=True)
        sdf_pred = utils.grid_decode(feats, x, model.decoder, pos_invariant=True)
        return sdf_pred
    
    def compute_residuals(self, model_id: int, corrections, observation: EncoderObservation,
                          skip_sign=False, skip_eik=True, skip_smooth=True, smooth_std=0.1):
        """Compute SDF residuals given input model and corrections.

        Args:
            model (GridNet): 
            corrections (_type_): 
            coords_world (Tensor): N,3 set of observed 3D positions
            gt_sdf (Tensor): N,1 SDF labels
            gt_sdf_sign (Tensor): N,1 SDF sign labels
            gt_sdf_valid (Tensor): N,1 SDF valid labels
            skip_sign (bool, optional): Defaults to False.
            skip_eik (bool, optional):  Defaults to True.
            skip_smooth (bool, optional): Defaults to True.
        """
        coords_world = observation.coords_world
        gt_sdf = observation.gt_sdf
        gt_sdf_valid = observation.gt_sdf_valid
        gt_sdf_sign = observation.gt_sdf_sign
        model = self.get_grid_net(model_id)
        sdf_pred = self.query_sdf(model, corrections, coords_world)
        sdf_res = torch.where(gt_sdf_valid == 1, gt_sdf - sdf_pred, torch.zeros_like(sdf_pred))
        output = {
            'sdf_constraint': sdf_res,
            'sdf_coords': coords_world
        }
        if not skip_sign:
            fs_upper_constraint = torch.where(
                gt_sdf_sign == 1,
                F.relu(sdf_pred - gt_sdf),
                torch.zeros_like(sdf_pred)
            )  # linear cost if exceeding bound
            fs_lower_constraint = torch.where(
                gt_sdf_sign == 1,
                F.relu(self.trunc_dist - sdf_pred),
                torch.zeros_like(sdf_pred)
            )  # Option 2: linear loss if predicting smaller than truncation distance
            output['fs_constraint'] = torch.maximum(
                fs_upper_constraint,
                fs_lower_constraint
            )
            output['fs_upper_constraint'] = fs_upper_constraint
            output['fs_lower_constraint'] = fs_lower_constraint
        if not skip_eik:
            N = gt_sdf.shape[0] 
            bound = model.bound.detach().cpu().numpy()
            xs = np.reshape(np.random.uniform(bound[0,0], bound[0,1], N), (N,1))
            ys = np.reshape(np.random.uniform(bound[1,0], bound[1,1], N), (N,1))
            zs = np.reshape(np.random.uniform(bound[2,0], bound[2,1], N), (N,1))
            x_np = np.concatenate([xs, ys, zs], axis=1)
            eik_coords = torch.from_numpy(x_np).to(gt_sdf)
            eik_coords.requires_grad_(True)
            eik_sdfs = self.query_sdf(model, corrections, eik_coords)
            gradient = torch.autograd.grad(eik_sdfs, eik_coords, grad_outputs=torch.ones_like(eik_sdfs), create_graph=True)[0]
            output['eik_constraint'] = gradient.norm(dim=-1) - 1
        if not skip_smooth:
            coords_1 = coords_world
            coords_2 = coords_1 + torch.normal(0, smooth_std, size=coords_1.shape).to(coords_1)
            coords_1.requires_grad_(True)
            coords_2.requires_grad_(True)
            sdfs_1 = self.query_sdf(model, corrections, coords_1)
            sdfs_2 = self.query_sdf(model, corrections, coords_2)
            grad1 = torch.autograd.grad(sdfs_1, coords_1, grad_outputs=torch.ones_like(sdfs_1), create_graph=True)[0]
            grad2 = torch.autograd.grad(sdfs_2, coords_2, grad_outputs=torch.ones_like(sdfs_2), create_graph=True)[0]
            # grad1 = gradient3d(coords_1, model, method=self.grad_method, finite_diff_eps=self.finite_diff_eps)
            # grad2 = gradient3d(coords_2, model, method=self.grad_method, finite_diff_eps=self.finite_diff_eps)
            output['smooth_constraint'] = torch.where(
                gt_sdf_valid == 1,
                grad1 - grad2,
                torch.zeros_like(grad1)
            ) 
        return output
    
    def compute_encoder_inputs_from_residuals(self, input_residuals_dict, model_id: int, target_level:int, save=False):
        """Compute inputs to the encoder from the residual dict.
        """
        grid = self.get_grid_net(model_id)
        signals = []
        signals.append(utils.grid_pool_3d_avg(
            coords=input_residuals_dict['sdf_coords'],
            features=input_residuals_dict['sdf_constraint'],
            grid_bound=grid.bound,
            cell_size=grid.features[target_level].cell_size
        ).squeeze())  # H, W, D, 1 for 3D
        signals.append(utils.grid_pool_3d_avg(
            coords=input_residuals_dict['sdf_coords'],
            features=input_residuals_dict['fs_upper_constraint'],
            grid_bound=grid.bound,
            cell_size=grid.features[target_level].cell_size
        ).squeeze())  # H, W, D, 1 for 3D
        signals.append(utils.grid_pool_3d_avg(
            coords=input_residuals_dict['sdf_coords'],
            features=input_residuals_dict['fs_lower_constraint'],
            grid_bound=grid.bound,
            cell_size=grid.features[target_level].cell_size
        ).squeeze())  # H, W, D, 1 for 3D
        encoder_inputs = torch.stack(signals, dim=-1)
        encoder_inputs = encoder_inputs.permute([3, 2, 1, 0]).unsqueeze(0) 
        if save:
            key = f"encoder_inputs_model{model_id}_level{target_level}"
            self.intermediate_results[key] = encoder_inputs.clone().detach()
            key = f"residuals_coords_model{model_id}_level{target_level}"
            self.intermediate_results[key] = input_residuals_dict['sdf_coords'].clone().detach()
            key = f"residuals_values_model{model_id}_level{target_level}"
            self.intermediate_results[key] = input_residuals_dict['sdf_constraint'].clone().detach()
        return encoder_inputs
    
    def compute_encoder_outputs(self, model_id:int, encoder_inputs, target_level:int):
        """Compute the encoder output (i.e., predicted features) from encoder input dict.
        """
        grid = self.get_grid_net(model_id)
        target_spatial_size = grid.features[target_level].feature.shape[2:]
        pred_features = self.feature_encoders[target_level].predict(None, encoder_inputs, target_spatial_size)
        return pred_features
    
    def save_visualizations(self, model_id:int, save_dir, resolution=512):
        grid = self.get_grid_net(model_id)
        grid_corrections = self.stored_corrections_until_level(model_id, stop_level=grid.num_levels)
        bound_torch = grid.bound
        query_func = lambda x: self.query_sdf(grid, grid_corrections, x)
        utils_sdf.save_mesh(
            query_func, 
            bound_torch, 
            join(save_dir, f"model{model_id}_mesh.ply"), 
            resolution=resolution
        )
        utils_sdf.visualize_sdf_plane(
            query_func, 
            bound_torch, 
            resolution=resolution, 
            axis='z', 
            fig_path=join(save_dir, f"model{model_id}_sdf_plane.png")
        )
        # visualize encoder inputs
        color_bounds = (-0.25, 0.25)
        for level in range(grid.num_levels):
            key = f"encoder_inputs_model{model_id}_level{level}"
            if key in self.intermediate_results:
                encoder_inputs = self.intermediate_results[key].squeeze().permute([3,2,1,0])
                _, _, D, C = encoder_inputs.shape
                for ch in range(C):
                    encoder_inputs_ch_np = encoder_inputs[:, :, D//4, ch].detach().cpu().numpy()
                    utils.visualize_grid_scalar(
                        encoder_inputs_ch_np,
                        fig_path=join(save_dir, f"encoder_inputs_model{model_id}_level{level}_ch{ch}.png"),
                        cmap='seismic',
                        bounds=color_bounds,
                        show_colorbar=True,
                        show_title=False,
                        hide_axis=True
                    )

    def save_all_visualizations(self, save_dir, resolution=512):
        for model_id in range(len(self.grid_nets)):
            self.save_visualizations(model_id, save_dir, resolution)


class EncoderPretrainLoss(BaseLoss):
    """The loss used for pre-training encoders.
    """
    def __init__(self, target_level, sdf_weight=3e3, sign_weight=0, eik_weight=0, smooth_weight=0, trunc_dist=0.15, smooth_std=0.01, pred_std=0.1, dataset_groups=[], group_weight=1e3, reg_weight=0):
        super().__init__()
        self.sdf_weight = sdf_weight
        self.sign_weight = sign_weight
        self.eik_weight = eik_weight
        self.smooth_weight = smooth_weight
        self.smooth_std = smooth_std
        self.trunc_dist = trunc_dist
        self.target_level = target_level
        self.pred_std = pred_std
        self.latest_encoder_inputs = {0: {}, 1: {}}
        # self.dataset_groups = dataset_groups
        # self.reg_weight = reg_weight
        # self.group_weight = group_weight
        self.skip_eik = (self.eik_weight == 0)
        self.skip_smooth = (self.smooth_weight == 0)
    
    def compute_loss_from_residuals(self, residuals_dict):
        loss_dict = {'sdf': torch.mean(residuals_dict['sdf_constraint']**2) * self.sdf_weight}
        if self.sign_weight > 0:
            loss_dict['free_space'] = torch.mean(residuals_dict['fs_constraint']) * self.sign_weight
        if self.eik_weight > 0:
            loss_dict['eik'] = torch.mean(residuals_dict['eik_constraint']**2) * self.eik_weight
        if self.smooth_weight > 0:
            loss_dict['smooth'] = torch.mean(residuals_dict['smooth_constraint']**2) * self.smooth_weight
        return loss_dict

    def compute(self, model:Encoder, model_input: dict, gt: dict) -> dict:
        # dataset_index = model_input['dataset_index'].item()
        # dataset_key = f"dataset_{dataset_index}"
        model_id = model_input['dataset_index'].item()
        model_key = model.grid_key(model_id)
        grid = model.grid_nets[model_key]
        # Process inputs
        gt_sdf = gt['sdf'][0]
        gt_sdf_valid = gt['sdf_valid'][0]
        gt_sdf_sign = gt['sdf_signs'][0]
        coords_world = utils_geometry.batch_transform_to_world_frame(
            model_input['coords_frame'][0], 
            model_input['frame_indices'][0], 
            model_input['R_world_frame'][0],
            model_input['t_world_frame'][0],
            grid.rotation_corrections,
            grid.translation_corrections
        )
        observation = EncoderObservation(
            coords_world=coords_world,
            gt_sdf=gt_sdf,
            gt_sdf_sign=gt_sdf_sign,
            gt_sdf_valid=gt_sdf_valid
        )
        final_corrections = model.predict_corrections_until_level(
            model_id=model_id, stop_level=self.target_level+1, observation=observation,
            pred_std=self.pred_std, store_corrections=True
        )
        final_residuals = model.compute_residuals(
            model_id=model_id, corrections=final_corrections, observation=observation,
            skip_sign=False, skip_eik=self.skip_eik, skip_smooth=self.skip_smooth, smooth_std=self.smooth_std
        )
        loss_dict = self.compute_loss_from_residuals(final_residuals)
        # loss_consensus = self.compute_group_consensus_loss(model)
        # loss_dict.update(loss_consensus)
        # loss_regularization = self.compute_feature_regularization_loss(model)
        # loss_dict.update(loss_regularization)
        self.latest_loss_dict = loss_dict
        return loss_dict