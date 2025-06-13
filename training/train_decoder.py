import argparse
from os.path import join
import torch
from torch.utils.data import DataLoader
import numpy as np
from grid_opt.datasets.sdf_3d import *
from grid_opt.models.modules import MLPNet
from grid_opt.models.grid_modules import FeatureGrid
from grid_opt.configs import *
from grid_opt.loss import BaseLoss
import grid_opt.utils.utils as utils
import grid_opt.utils.utils_sdf as utils_sdf
import matplotlib.pyplot as plt


import logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to config file.', default='./configs/grid/pretrain.yaml')
parser.add_argument('--default_config', type=str, help='Path to config file.', default='./configs/base.yaml')
parser.add_argument('--save_dir', type=str, default='/home/yulun/code/grid_opt/results/training/train_decoder')
parser.add_argument('-e', '--epochs', type=int, default=1000)


def construct_shared_decoder(cfg):
    print(f"\n === Constructing shared decoder === \n")
    num_levels = cfg['model']['grid']['n_levels']
    level_dim = cfg['model']['grid']['feature_dim']
    cfg_decoder = cfg['model']['decoder']
    decoder = MLPNet(
        input_dim=num_levels * level_dim,
        output_dim=cfg_decoder['out_dim'],
        hidden_dim=cfg_decoder['hidden_dim'],
        hidden_layers=cfg_decoder['hidden_layers'],
        bias=True
    )
    print(decoder)
    return decoder


def construct_grid_for_dataset(dataset:PosedSdf3D, cfg):
    print(f"\n === Constructing grid for {dataset.meshfile} === \n")
    device = cfg['device']
    # Enlarge bound so that surface are strictly within bounds.
    bound = torch.from_numpy(dataset.get_inflated_bound()).float().to(device)
    num_levels = cfg['model']['grid']['n_levels']
    level_dim = cfg['model']['grid']['feature_dim']
    base_cell_size = cfg['model']['grid']['base_cell_size']
    scale_factor = cfg['model']['grid']['per_level_scale']
    grid_type = cfg['model']['grid']['type']
    feature_grids = torch.nn.ModuleList()
    for level in range(num_levels):
        cell_size = base_cell_size / (scale_factor**level)
        if grid_type == 'regular':
            grid = FeatureGrid(
                d = 3,
                fdim=level_dim,
                bound=bound,
                cell_size=cell_size,
                name=f"{level}",
                dtype=torch.float32,
                initial_feature=None,
                init_stddev=cfg['model']['grid']['init_stddev']
            )
        else:
            raise ValueError(f"Invalid grid type: {grid_type}")
        
        feature_grids.append(grid)  
    return feature_grids


class PretrainModel(torch.nn.Module):
    def __init__(self, cfg, datasets: BatchedSdf3D):
        super().__init__()
        self.num_levels = cfg['model']['grid']['n_levels']
        assert self.num_levels == 2
        # Initialize shared modules
        self.decoder = construct_shared_decoder(cfg)
        self.grid_type = cfg['model']['grid']['type']
        # Initialize dataset-dependent modules
        self.grid_dict = torch.nn.ModuleDict()
        for dataset_index in range(datasets.num_datasets):
            dataset_key = f"dataset_{dataset_index}"
            self.grid_dict[dataset_key] = construct_grid_for_dataset(
                datasets.datasets[dataset_index], cfg
            )
        # Enable training level by level
        self.ignore_levels = np.zeros(self.num_levels).astype(bool)
        
    
    def set_level_require_grad(self, level, require_grad):
        for dataset_key in self.grid_dict.keys():
            for param in self.grid_dict[dataset_key][level].parameters():
                param.requires_grad = require_grad

    def query_sdf(self, dataset_index, x):
        dataset_key = dataset_key = f"dataset_{dataset_index}"
        if self.grid_type == 'regular':
            feats = utils.grid_interp_regular(
                self.grid_dict[dataset_key], 
                x, 
                ignore_level=self.ignore_levels
            )
        else:
            raise ValueError(f"Invalid grid type: {self.grid_type}")
        pred_sdf = utils.grid_decode(feats, x, self.decoder, pos_invariant=True)
        return pred_sdf
    
    def save_result(self, save_dir):
        # Save shared decoder
        decoder_ckpt_file = join(save_dir, "decoder.pt")
        self.decoder.save(decoder_ckpt_file)

    def save_figures(self, dataset_index, save_dir):
        dataset_key = dataset_key = f"dataset_{dataset_index}"
        query_func = lambda x: self.query_sdf(dataset_index, x)
        utils_sdf.save_mesh(
            query_func, 
            model.grid_dict[dataset_key][0].bound, 
            join(save_dir, f"{dataset_key}.ply"), 
            resolution=512
        )
        utils_sdf.visualize_sdf_plane(
            query_func, 
            model.grid_dict[dataset_key][0].bound, 
            512, 
            axis='z', 
            fig_path=os.path.join(save_dir, f"{dataset_key}_sdf_plane.png")
        )




class PretrainLoss(BaseLoss):
    def __init__(self, sdf_weight=3e3, sign_weight=100.0, trunc_dist=0.15, beta=5.0):
        super().__init__()
        self.sdf_weight = sdf_weight
        self.sign_weight = sign_weight
        self.trunc_dist = trunc_dist
        self.beta = beta

    def compute(self, model:PretrainModel, model_input: dict, gt: dict) -> dict:
        dataset_index = model_input['dataset_index'].item()
        coords = model_input['coords_world_gt'][0]  # For now, we use  ground truth pose
        gt_sdf = gt['sdf'][0]
        gt_sdf_valid = gt['sdf_valid'][0]
        gt_sdf_sign = gt['sdf_signs'][0]
        assert coords.ndim == 2 and gt_sdf.ndim == 2
        pred_sdf = model.query_sdf(dataset_index, coords)
        sdf_constraint = torch.where(
            gt_sdf_valid == 1,
            pred_sdf - gt_sdf,
            torch.zeros_like(pred_sdf)
        )
        loss_dict = {'sdf': torch.mean(sdf_constraint**2) * self.sdf_weight}
        if self.sign_weight > 0:
            fs_upper_constraint = torch.where(
                gt_sdf_sign == 1,
                F.relu(pred_sdf - gt_sdf),
                torch.zeros_like(pred_sdf)
            )  # linear cost if exceeding bound
            # fs_lower_constraint = F.relu(torch.where(
            #     gt_sdf_sign == 1,
            #     torch.exp(-self.beta * pred_sdf) - 1.0, 
            #     torch.zeros_like(pred_sdf)
            # ))  # Option 1: exponential cost if predicting negative, same as iSDF
            fs_lower_constraint = torch.where(
                gt_sdf_sign == 1,
                F.relu(self.trunc_dist - pred_sdf),
                torch.zeros_like(pred_sdf)
            )  # Option 2: linear loss if predicting smaller than truncation distance
            fs_constraint = torch.maximum(
                fs_upper_constraint,
                fs_lower_constraint
            )
            loss_dict['free_space'] = torch.mean(fs_constraint) * self.sign_weight

        return loss_dict



if __name__ == "__main__":
    args = parser.parse_args()
    cfg = load_config(args.config, args.default_config)
    cfg['train']['epochs'] = args.epochs
    cfg['train']['learning_rate'] = 1e-3
    cond_mkdir(args.save_dir)
    trunc_dist = cfg['loss']['trunc_dist']
    assert trunc_dist > 0    
    ####################################
    ###    Datasets to pretrain on   ###
    ####################################
    train_datasets = [
        "/mnt/drive/Datasets/replica_v1/room_0/mesh_centered.ply",
        "/mnt/drive/Datasets/replica_v1/room_1/mesh_centered.ply",
        "/mnt/drive/Datasets/replica_v1/office_0/mesh_centered.ply",
        "/mnt/drive/Datasets/replica_v1/office_1/mesh_centered.ply",
        "/mnt/drive/Datasets/replica_v1/office_2/mesh_centered.ply",
        "/mnt/drive/Datasets/replica_v1/office_3/mesh_centered.ply"
    ]

    datasets = BatchPosedSdf3D(train_datasets, num_frames=128, trunc_dist=trunc_dist)
    train_loader = DataLoader(datasets, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
    val_loader = None
    
    # Initialize model
    model = PretrainModel(cfg, datasets)
    print("\n === Summary of pretraining model === \n")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
    num_levels = cfg['model']['grid']['n_levels']

    # Initialize loss
    loss = PretrainLoss(trunc_dist=trunc_dist)
    
    # Coarse training
    print("\n === Start coarse training === \n")
    cfg['train']['log_dir'] = join(args.save_dir, "coarse_training")
    cfg['train']['epochs'] = 200
    model.ignore_levels[1] = True
    coarse_trainer = Trainer(
        cfg['train'],
        model,
        loss,
        train_loader,
        val_loader,
        device=cfg['device'],
        dtype=torch.float32
    )
    coarse_trainer.train()

    # Fine training
    print("\n === Start fine training === \n")
    cfg['train']['log_dir'] = join(args.save_dir, "fine_training")
    cfg['train']['epochs'] = 200
    model.ignore_levels[1] = False
    model.set_level_require_grad(level=0, require_grad=False)  # Freeze coarse level features
    fine_trainer = Trainer(
        cfg['train'],
        model,
        loss,
        train_loader,
        val_loader,
        device=cfg['device'],
        dtype=torch.float32
    )
    fine_trainer.train()

    # Joint training
    print("\n === Start joint training === \n")
    cfg['train']['learning_rate'] = 1e-4
    cfg['train']['epochs'] = args.epochs
    cfg['train']['log_dir'] = join(args.save_dir, "joint_training")
    model.set_level_require_grad(level=0, require_grad=True)
    joint_trainer = Trainer(
        cfg['train'],
        model,
        loss,
        train_loader,
        val_loader,
        device=cfg['device'],
        dtype=torch.float32
    )
    joint_trainer.train()

    ####################################
    ###         Post training        ###
    ####################################
    model.save_result(args.save_dir)
    # Visualize trained mesh
    for dataset_index in range(datasets.num_datasets):
        model.save_figures(dataset_index, args.save_dir)
    
    


    
    






    
    





    