import argparse
from os.path import join
import torch
import numpy as np
from copy import deepcopy
import math
from grid_opt.models.encoder import Encoder, EncoderPretrainLoss
from grid_opt.datasets.sdf_3d import *
from grid_opt.configs import *
import grid_opt.utils.utils as utils
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to config file.', default='./configs/grid/pretrain.yaml')
parser.add_argument('--default_config', type=str, help='Path to config file.', default='./configs/base.yaml')
parser.add_argument('--save_dir', type=str, default='./results/training/train_encoders')
parser.add_argument('-e', '--epochs', type=int, default=1000)
parser.add_argument('--decoder_weights', type=str, help='Path to the pretrained decoder', default='./results/trained_decoders/decoder_indoor.pt')


def save_encoder_level(encoder: Encoder, level):
    ckpt_file = join(args.save_dir, f"feature_encoder_level_{level}.pt")
    torch.save(
        encoder.feature_encoders[level].state_dict(),
        ckpt_file
    )


def train_encoder_level(encoder: Encoder, cfg: dict, target_level: int):
    loss = EncoderPretrainLoss(
        target_level=target_level,
        sdf_weight=cfg['loss']['sdf_weight'],
        sign_weight=cfg['loss']['sign_weight'], 
        eik_weight=cfg['loss']['eik_weight'],
        smooth_weight=cfg['loss']['smooth_weight'], 
        smooth_std=cfg['loss']['smooth_std'],
        trunc_dist=cfg['loss']['trunc_dist'],
        pred_std=1e-3
    )
    encoder.lock_all_params()
    encoder.unlock_encoder_at_level(level=target_level)
    encoder.print_trainable_params()
    cfg['train']['learning_rate'] = 1e-3
    cfg['train']['epochs'] = args.epochs
    cfg['train']['verbose'] = True
    trainer = Trainer(
        cfg['train'],
        encoder,
        loss,
        train_loader,
        None,
        cfg['device'],
        torch.float32
    )
    trainer.train()
    save_encoder_level(encoder, target_level)

    # For visualization, set `pred_std` to 0
    loss.pred_std = 0
    trainer.run_eval(epoch=trainer.get_last_epoch())
    # Save intermediate training visualizations
    vis_dir = join(args.save_dir, f"level{target_level}_vis")
    for model_id in range(num_datasets):
        encoder.save_visualizations(model_id, vis_dir, resolution=128)


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = load_config(args.config, args.default_config)
    cfg['train']['log_dir'] = args.save_dir
    cfg['model']['decoder']['pretrained_model'] = args.decoder_weights
    # Add smoothness regularization to account for the simulated noise in the training dataset
    cfg['loss']['smooth_weight'] = 10.0  
    cond_mkdir(args.save_dir)    

    # Part I: Create training dataset
    train_datasets = [
        "/mnt/drive/Datasets/replica_v1/room_0/mesh_centered.ply",
        "/mnt/drive/Datasets/replica_v1/room_1/mesh_centered.ply",
        "/mnt/drive/Datasets/replica_v1/office_0/mesh_centered.ply",
        "/mnt/drive/Datasets/replica_v1/office_1/mesh_centered.ply",
        "/mnt/drive/Datasets/replica_v1/office_2/mesh_centered.ply",
        "/mnt/drive/Datasets/replica_v1/office_3/mesh_centered.ply",
    ]
    dataset_bounds = [
        [[-4.1, 4.1], [-2.5, 2.5], [-1.6, 1.6]],  # room 0
        [[-3.0, 3.3], [-1.9, 1.8], [-1.5, 1.5]],  # room 1
        [[-2.5, 2.5], [-2.2, 2.2], [-1.7, 1.7]],  # office 0
        [[-2.2, 2.3], [-1.2, 1.9], [-1.5, 1.5]],  # office 1
        [[-3.7, 3.7], [-2.1, 2.1], [-1.5, 1.5]],  # office 2
        [[-4.1, 4.1], [-2.6, 2.5], [-1.6, 1.6]],  # office 3
    ]
    num_datasets = len(train_datasets)
    # Simulate noisy poses and distance measurements
    # to account for possible errors during test time.
    datasets = BatchPosedSdf3D(
        train_datasets,
        num_frames=cfg['dataset']['num_frames'],
        trunc_dist=cfg['dataset']['trunc_dist'],
        frame_std_meter=0.005,     # 0.5 cm
        frame_std_rad=0.00872665,  # 0.5 degree
        distance_std=0.01,         # 1cm distance noise
        resample_poses_freq=50
    )
    train_loader = DataLoader(datasets, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

    # Part II: create grid nets and encoder models
    encoder = Encoder(cfg)
    for index in range(num_datasets):
        cfg_copy = deepcopy(cfg)
        cfg_copy['model']['grid']['bound'] = dataset_bounds[index]
        grid_net = cfg_model(cfg_copy)
        encoder.register_grid_model(grid_net)

    # Part III: train!
    for level in range(encoder.num_levels):
        train_encoder_level(encoder, cfg, target_level=level)
        
    