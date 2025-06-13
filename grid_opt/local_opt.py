from copy import deepcopy
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from grid_opt.configs import cfg_loss
from grid_opt.models.grid_net import GridNet
from grid_opt.models.grid_atlas import GridAtlas
from grid_opt.models.encoder import Encoder, EncoderObservation
from grid_opt.trainer import Trainer, GridTrainer
import grid_opt.utils.utils as utils
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def initialize_grid_net(
    grid: GridNet,
    init_mode='encode',
    encoder:Encoder=None,
    encoder_observation:EncoderObservation=None,
    encoder_stop_level:int=None
):
    assert isinstance(grid, GridNet)
    info = {'total_encoder_time': 0}
    if init_mode == 'zero':
        grid.zero_features()
        return grid, info
    elif init_mode == 'randn':
        grid.randn_features(std=1e-4)
        return grid, info
    # Initialize using encoder
    assert encoder is not None
    assert encoder_observation is not None
    if encoder_stop_level is None:
        encoder_stop_level = grid.num_levels
    grid.zero_features()
    model_id = encoder.register_grid_model(grid)
    timer = utils.PerfTimer(activate=True)
    timer.reset()
    corrections = encoder.predict_corrections_until_level(
        model_id=model_id,
        stop_level=encoder_stop_level,
        observation=encoder_observation,
        pred_std=0,
        store_corrections=False
    )
    cpu_time, gpu_time = timer.check()
    with torch.no_grad():
        for level in range(grid.num_levels):
            assert grid.features[level].feature.shape == corrections[level].shape
            grid.features[level].feature.copy_(corrections[level])
    info['total_encoder_time'] = gpu_time
    return grid, info


def optimize_grid_net(
    grid: GridNet,
    dataset: Dataset,
    cfg: dict,
    iterations=0,
    learning_rate=1e-3,
    eval_every=-1,
    eval_tuples=[],
    train_mode='joint',
    iterations_per_level=50
):
    train_loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)
    val_loader=None
    assert cfg['loss']['name'] == 'iSDF'
    finetune_loss = cfg_loss(cfg)
    cfg_train = deepcopy(cfg['train'])
    cfg_train['max_epochs_in_level'] = iterations_per_level
    cfg_train['relchange_tol'] = 0
    cfg_train['grid_training_mode'] = train_mode
    cfg_train['epochs'] = iterations
    cfg_train['learning_rate'] = learning_rate
    cfg_train['verbose'] = True
    cfg_train['eval_every'] = eval_every
    trainer = GridTrainer(
        cfg_train,
        grid,
        finetune_loss,
        train_loader,
        val_loader,
        cfg['device'],
        torch.float32
    )
    grid.print_trainable_params()
    for metric_name, metric_func in eval_tuples:
        trainer.register_eval_func(name=metric_name, func=metric_func)
    trainer.train()    
    info = {}
    info['trainer_epoch'] = trainer.train_dict['epochs']
    info['trainer_epoch_time'] = trainer.train_dict['epoch_time']
    info['trainer_total_loss'] = trainer.train_dict['total_loss']
    return grid, info


def initialize_grid_atlas(
    grid_atlas: GridAtlas,
    init_mode='encode',
    encoder:Encoder=None,
    encoder_observations:List[EncoderObservation]=None,
    encoder_stop_level:int=None
):
    info = {}
    for submap_id in range(grid_atlas.num_submaps):
        grid = grid_atlas.get_submap(submap_id)
        if init_mode == 'encode':
            encoder_obs = encoder_observations[submap_id]
        else:
            encoder_obs = None
        grid, submap_info = initialize_grid_net(
            grid, init_mode, encoder, encoder_obs, encoder_stop_level
        )
    return grid_atlas, info


def optimize_grid_atlas(
    grid_atlas: GridAtlas,
    dataset: Dataset,
    cfg: dict,
    iterations=0,
    learning_rate=0.0013,
    train_mode='coordinate'
):
    train_loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)
    val_loader = None
    assert cfg['loss']['name'] == 'iSDFSubmap'
    grid_atlas_loss = cfg_loss(cfg)
    cfg_copy = deepcopy(cfg)
    cfg_train = cfg_copy['train']
    cfg_train['max_epochs_in_level'] = 50
    cfg_train['relchange_tol'] = 0
    cfg_train['grid_training_mode'] = train_mode
    cfg_train['epochs'] = iterations
    cfg_train['learning_rate'] = learning_rate
    cfg_train['verbose'] = True
    cfg_train['eval_every'] = -1
    trainer = GridTrainer(
        cfg_train,
        grid_atlas,
        grid_atlas_loss,
        train_loader,
        val_loader,
        cfg['device'],
        torch.float32
    )
    grid_atlas.print_trainable_params()
    trainer.train()
    grid_atlas.print_keyframe_pose_info()
    grid_atlas.print_submap_pose_info()
    info = {}
    return grid_atlas, info