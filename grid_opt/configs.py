import yaml
import torch
from torch.utils.data import DataLoader
from grid_opt.models.grid_net import GridNet
from grid_opt.models.isdf.embedding import PostionalEncoding
from grid_opt.models.isdf.isdf import iSDF
from grid_opt.models.pointsdf.pointsdf import PointSDF
from grid_opt.loss import *
from grid_opt.loss_isdf import iSDFLoss, iSDFLossSubmap
from grid_opt.trainer import *
from grid_opt.datasets.sdf_2d import Sdf2D
from grid_opt.datasets.replicaCAD import ReplicaCAD
from grid_opt.datasets.scannet import ScanNet
from grid_opt.datasets.fastcamo import FastCaMo
from grid_opt.datasets.sdf_3d import Sdf3D, PosedSdf3D
from grid_opt.datasets.sdf_3d_lidar import PosedSdf3DLidar
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_config(path, default_path=None):
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    # load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.full_load(f)

    # check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # if yes, load this config first as default
    # if no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def cfg_model(cfg):
    """Configure model

    Args:
        cfg (_type_): _description_
    """
    device = cfg['device']
    dtype = torch.float32
    model_name = cfg['model']['name']
    
    if model_name == 'grid_net':
        model = GridNet(cfg=cfg['model'], device=device, dtype=dtype)

    elif model_name == 'iSDF':
        transform_file = cfg['dataset']['path'] + '/T_extent_to_scene.npy'
        # load the transform
        if os.path.exists(transform_file):
            transform = np.load(transform_file)
        else:
            transform = np.eye(4)
        transform = torch.from_numpy(transform).float().to(device)
        positional_encoding = PostionalEncoding(
            min_deg=0,
            max_deg=cfg['model']['positional_encoding']['n_embed_funcs'],
            scale=cfg['model']['positional_encoding']['scale_input'],
            transform=transform
        )
        model = iSDF(cfg=cfg['model'], 
                     positional_encoding=positional_encoding,
                     hidden_layers_block=cfg['model']['hidden_layers_block'],
                     scale_output=cfg['model']['scale_output'],
                     device=device, dtype=dtype)
        # model = SDFMap()

    elif model_name == 'pointSDF':
        model = PointSDF(cfg=cfg['model'], meshfile=cfg['dataset']['ref_mesh'], device=device, dtype=dtype)
    
    elif model_name == 'ngp':
        from grid_opt.models.grid_ngp import GridNGP
        model = GridNGP(cfg=cfg['model'], device=device, dtype=dtype)

    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.to(device)
    logger.info(f"Configured model: {model_name}.")
    return model


def cfg_loss(cfg):
    loss_name = cfg['loss']['name']
    if loss_name == 'Sdf2D':
        loss = SdfLoss2D(
            sdf_weight=cfg['loss']['sdf_weight']
        )
    elif loss_name == 'Sdf3D':
        loss = SdfLoss3D(
            sdf_weight=cfg['loss']['sdf_weight']
        )
    elif loss_name == 'iSDF':
        loss = iSDFLoss(
            model_name=cfg['model']['name'],
            trunc_weight=cfg['loss']['trunc_weight'],
            trunc_distance=cfg['loss']['trunc_distance'],
            noise_std=cfg['loss']['noise_std'],
            orien_loss=bool(cfg['loss']['orien_loss']),
            eik_weight=cfg['loss']['eik_weight'],
            grad_weight=cfg['loss']['grad_weight'],
            eik_apply_dist=cfg['loss']['eik_apply_dist'],
            smooth_weight=cfg['loss']['smooth_weight'],
            smooth_std=cfg['loss']['smooth_std'],
            loss_type=cfg['loss']['loss_type'],
            slam_mode=cfg['loss']['slam_mode'],
            pose_reg_weight=cfg['loss']['pose_reg_weight'],
            pose_thresh_m=cfg['loss']['pose_thresh_m'],
            pose_thresh_rad=cfg['loss']['pose_thresh_rad']
        )
    elif loss_name == 'iSDFSubmap':
        loss = iSDFLossSubmap(
            model_name=cfg['model']['name'],
            trunc_weight=cfg['loss']['trunc_weight'],
            trunc_distance=cfg['loss']['trunc_distance'],
            noise_std=cfg['loss']['noise_std'],
            orien_loss=bool(cfg['loss']['orien_loss']),
            eik_weight=cfg['loss']['eik_weight'],
            grad_weight=cfg['loss']['grad_weight'],
            eik_apply_dist=cfg['loss']['eik_apply_dist'],
            smooth_weight=cfg['loss']['smooth_weight'],
            smooth_std=cfg['loss']['smooth_std'],
            loss_type=cfg['loss']['loss_type'],
            feat_reg_weight=1.0,
            pose_reg_weight=cfg['loss']['pose_reg_weight'],
            pose_thresh_m=cfg['loss']['pose_thresh_m'],
            pose_thresh_rad=cfg['loss']['pose_thresh_rad'],
            slam_mode=cfg['loss']['slam_mode']
        )
    elif loss_name == 'Tsdf3D':
        loss = TsdfLoss3D(
            sdf_weight=cfg['loss']['sdf_weight'],
            sign_weight=cfg['loss']['sign_weight'],
            eik_weight=cfg['loss']['eik_weight'],
            trunc_dist=cfg['loss']['trunc_dist'],
            grad_method=cfg['loss']['grad_method'],
            finite_diff_eps=cfg['loss']['finite_diff_eps']
        )
    elif loss_name == 'PosedSdf3D':
        loss = PosedSdfLoss3D(
            sdf_weight=cfg['loss']['sdf_weight'],
            sign_weight=cfg['loss']['sign_weight'],
            eik_weight=cfg['loss']['eik_weight'],
            smooth_weight=cfg['loss']['smooth_weight'],
            smooth_std=cfg['loss']['smooth_std'],
            trunc_dist=cfg['loss']['trunc_dist'],
            grad_method=cfg['loss']['grad_method'],
            finite_diff_eps=cfg['loss']['finite_diff_eps'],
            loss_type=cfg['loss']['type'],
        )
    elif loss_name == 'PosedSdf3DSubmap':
        loss = PosedSdfLoss3DSubmap(cfg)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
    
    logger.info(f"Configured loss: {loss_name}.")
    return loss
    

def cfg_dataset(cfg):
    dataset_name = cfg['dataset']['name']
    if dataset_name == 'Sdf2D':
        dataset = Sdf2D(
            mapfile=cfg['dataset']['path'],
            batch_size=cfg['train']['batch_size'],
            samples_near=cfg['dataset']['samples_near'],  
            samples_unif=cfg['dataset']['samples_uniform']
        )
        trainloader = DataLoader(dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
        valloader = None
    elif dataset_name == 'Sdf3D':
        dataset = Sdf3D(
            meshfile=cfg['dataset']['path'], 
            batch_size=cfg['train']['batch_size'],
            normalize=cfg['dataset']['normalize'],
            trunc_dist=cfg['dataset']['trunc_dist']
        )
        trainloader = DataLoader(dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
        valloader = None
    elif dataset_name == 'ReplicaCAD':
        dataset = ReplicaCAD(cfg)
        trainloader = DataLoader(dataset, shuffle=False, batch_size=1)
        valloader = None
    elif dataset_name == 'ScanNet':
        dataset = ScanNet(cfg)
        trainloader = DataLoader(dataset, shuffle=False, batch_size=1)
        valloader = None
    elif dataset_name == 'FastCaMo':
        dataset = FastCaMo(cfg)
        trainloader = DataLoader(dataset, shuffle=False, batch_size=1)
        valloader = None
    elif dataset_name == 'PosedSdf3D':
        dataset = PosedSdf3D(
            meshfile=cfg['dataset']['path'],
            num_frames=cfg['dataset']['num_frames'],
            trunc_dist=cfg['dataset']['trunc_dist'],
            frame_std_meter=cfg['dataset']['frame_std_meter'],
            frame_std_rad=cfg['dataset']['frame_std_rad'],
            distance_std=cfg['dataset']['distance_std']
        )
        trainloader = DataLoader(dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
        valloader = None
    elif dataset_name == 'PosedSdf3DLidar':
         dataset = PosedSdf3DLidar(
             lidar_folder=cfg['dataset']['path'],
             pose_file_gt=cfg['dataset']['pose_gt'],
             pose_file_init=cfg['dataset']['pose_init'],
             num_frames=cfg['dataset']['num_frames'],
             trunc_dist=cfg['dataset']['trunc_dist'],
             distance_std=cfg['dataset']['distance_std'],
             frame_samples=cfg['dataset']['frame_samples'],
             frame_batchsize=cfg['dataset']['frame_batchsize'],
             num_frames_per_submap=cfg['dataset']['num_frames_per_submap'],
             bound=cfg['dataset']['bound']
         )
         trainloader = DataLoader(dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
         valloader = None
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    logger.info(f"Configured dataset: {dataset_name}.")
    return trainloader, valloader


def cfg_trainer(cfg, model=None):
    device = cfg['device']
    dtype = torch.float32
    train_loader, val_loader = cfg_dataset(cfg)
    loss = cfg_loss(cfg)
    if model is None:
        model = cfg_model(cfg)
    if 'trainer' not in cfg['train'].keys():
        cfg['train']['trainer'] = 'base'
    if cfg['train']['trainer'] == 'base':
        TrainerClass = Trainer
    elif cfg['train']['trainer'] == 'grid':
        TrainerClass = GridTrainer
    else: 
        raise ValueError(f"Invalid trainer type: {cfg['train']['trainer']}.")
    trainer = TrainerClass(
        cfg['train'],
        model,
        loss,
        train_loader,
        val_loader,
        device,
        dtype
    )
    logger.info(f"Configured trainer: {cfg['train']['trainer']}.")
    # Save cfg to log
    cfg_out_file = os.path.join(cfg['train']['log_dir'], 'cfg.yaml')
    logger.info(f"Save used config to {cfg_out_file}")
    with open(cfg_out_file, 'w') as file:
        yaml.dump(cfg, file, default_flow_style=False)

    return trainer, model