import torch
import math
from copy import deepcopy
from grid_opt.configs import *
from grid_opt.models.grid_atlas import GridAtlas
from grid_opt.datasets.submap_dataset import SubmapDataset
from grid_opt.align.miso import align_multiple_submaps_hierarchical 
import logging
logger = logging.getLogger(__name__)


class Fuser:
    """
    A class for aligning and fusing multiple submaps in a grid atlas.
    """
    def __init__(
        self,
        model: GridAtlas,              
        dataset: SubmapDataset,
        cfg: dict,
    ):
        assert isinstance(model, GridAtlas), "Model must be an instance of GridAtlas."
        self.model = model
        self.dataset = dataset
        self.train_loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)
        self.cfg = cfg
    

    def align(self):
        cfg_align = self.cfg['align']
        verbose = cfg_align.get('verbose', False)
        save_iterations = cfg_align.get('save_iterations', False)
        pose_thresh_m = cfg_align.get('pose_thresh_m', 10.0)  
        pose_thresh_deg = cfg_align.get('pose_thresh_deg', 45.0)
        align_info = align_multiple_submaps_hierarchical(
            grid_atlas=self.model,
            dataset=self.dataset,
            level_iters=cfg_align['level_iters'],
            finetune_iters=cfg_align['finetune_iters'],
            level_thresh=0,
            lr=cfg_align['learning_rate'],
            align_loss=cfg_align['loss_type'],
            stability_thresh=cfg_align['stability_thresh'],
            subsample_points=cfg_align['subsample_points'],
            latent_levels=cfg_align['latent_levels'],
            skip_finetune=cfg_align['skip_finetune'],
            pose_reg_weight=cfg_align['pose_reg_weight'],
            pose_thresh_m=pose_thresh_m,
            pose_thresh_rad=math.radians(pose_thresh_deg),
            verbose=verbose,
            save_iterations=save_iterations
        )
        self.model.print_submap_pose_info()
        return align_info

    
    def fuse(
            self,
            feat_lr=1e-3,
            submap_pose_lr=1e-4,
            kf_pose_lr=1e-4,
            iterations=10,
        ):
        self.dataset.unselect_keyframes()
        for submap_id in range(self.model.num_submaps):
            self.model.unlock_submap(submap_id)
        self.model.unlock_submap_pose()
        param_groups = []
        if feat_lr > 0:
            param_groups.append({'params': self.model.params_for_all_features(), 'lr': feat_lr})
        else:
            for param in self.model.params_for_all_features():
                param.requires_grad_(False)
        if submap_pose_lr > 0:
            param_groups.append({'params': self.model.params_for_all_submap_poses(), 'lr': submap_pose_lr})
        else:
            for param in self.model.params_for_all_submap_poses():
                param.requires_grad_(False)
        if kf_pose_lr > 0:
            param_groups.append({'params': self.model.params_for_all_kf_poses(), 'lr': kf_pose_lr})
        else:
            for param in self.model.params_for_all_kf_poses():
                param.requires_grad_(False)
        if len(param_groups) == 0:
            logger.warning("No parameters to optimize. Please check the learning rates.")
            return
        # Config and optimize!
        cfg_copy = deepcopy(self.cfg)  
        cfg_map = cfg_copy['mapping']
        cfg_train = cfg_copy['train']
        cfg_train['epochs'] = iterations
        cfg_train['verbose'] = True
        loss_func = MisoLossFusion(
            weight_sdf=cfg_map['weight_sdf'],
            weight_eik=cfg_map['weight_eik'],
            weight_fs=cfg_map['weight_fs'],
            loss_type=cfg_map['loss_type'],
            trunc_dist=cfg_map['trunc_dist'],
            finite_diff_eps=cfg_map['finite_diff_eps'],
            grad_method=cfg_map['grad_method'],
            eik_trunc_dist=cfg_map['eik_trunc_dist'],
            gm_scale_sdf=cfg_map['gm_scale_sdf'],   # Experimental,
            use_stability=False
        )
        trainer = Trainer(
            cfg_train,
            self.model,
            loss_func,
            self.train_loader,
            None,
            self.cfg['device'],
            torch.float32
        )
        self.model.print_trainable_params()
        # Reset optimizer
        adam_opt = optim.Adam(param_groups, lr=1e-3)
        trainer.set_external_optimizer(adam_opt)
        trainer.train()
        self.model.print_keyframe_pose_info()
        self.model.print_submap_pose_info()