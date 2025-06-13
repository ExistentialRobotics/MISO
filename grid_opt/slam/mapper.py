import torch
from os.path import join
from copy import deepcopy
from typing import List
from grid_opt.configs import *
from grid_opt.models.grid_net import GridNet
from grid_opt.datasets.submap_dataset import SubmapDataset
from grid_opt.loss import MisoLossMapping
import grid_opt.utils.utils_sdf as utils_sdf
import logging
logger = logging.getLogger(__name__)


def eval_func(epoch, cfg, model, loss_func, train_dataloader, val_dataloader):
    """A evaluation function to be provided to the trainer.
    """
    assert isinstance(model, GridNet)
    fig_path = join(cfg['log_dir'], f"sdf_plane_epoch{epoch}.png")
    utils_sdf.visualize_sdf_plane(
        model, model.bound, axis='z', fig_path=fig_path,
        show_colorbar=False, show_title=True, hide_axis=True,
        title=f"Iteration {epoch}"
    )
    return 0


class Mapper:
    """
    A class for mapping keyframes in a SLAM system.
    """
    def __init__(
            self,
            model: GridNet,              # We always perform mapping in a single GridNet
            dataset: SubmapDataset,
            cfg: dict,
        ):
        assert isinstance(model, GridNet) or isinstance(model, GridNGP), f"Invalid model type {type(model)}."
        self.grid = model
        self.dataset = dataset
        self.train_loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)
        self.cfg = cfg
        cfg_map = cfg['mapping']
        self.lr = cfg_map['learning_rate']
        self.verbose = cfg_map['verbose']
        if 'disable' in cfg_map and cfg_map['disable']:
            self.disable = True
        else:
            self.disable = False
        logger.info(f"Initializing mapper.")
        logger.info(f"  - learning rate: {self.lr}.")
        logger.info(f"  - verbose: {self.verbose}.")
        logger.info(f"  - disable optimization: {self.disable}.")
        self.loss_fn = MisoLossMapping(
            weight_sdf=cfg_map['weight_sdf'],
            weight_eik=cfg_map['weight_eik'],
            weight_fs=cfg_map['weight_fs'],
            loss_type=cfg_map['loss_type'],
            trunc_dist=cfg_map['trunc_dist'],
            finite_diff_eps=cfg_map['finite_diff_eps'],
            grad_method=cfg_map['grad_method'],
            eik_trunc_dist=cfg_map['eik_trunc_dist'],
        )


    def mapping(self, mapping_kfs, iterations=10, level_iterations=5):
        """
        Map the specified keyframes in the SLAM system.
        """
        if self.disable: return
        logger.info(f"Mapping frames: {mapping_kfs}.")
        # Only make the specified keyframes trainable
        self.grid.unlock_feature()
        self.grid.lock_pose()
        # Only sample from the specified keyframes
        self.dataset.select_keyframes(mapping_kfs)
        # Config and optimize!
        cfg_copy = deepcopy(self.cfg)  
        cfg_train = cfg_copy['train']
        cfg_train['max_epochs_in_level'] = level_iterations
        cfg_train['epochs'] = iterations
        cfg_train['learning_rate'] = self.lr
        cfg_train['verbose'] = self.verbose
        trainer = GridTrainer(
            cfg_train,
            self.grid,
            self.loss_fn,
            self.train_loader,
            None,
            self.cfg['device'],
            torch.float32
        )
        if self.verbose:
            self.grid.print_trainable_params()
        # trainer.register_eval_func("visualize", eval_func)
        trainer.train()
        if self.verbose: 
            self.grid.print_kf_pose_info()
            self.grid.print_feature_info()