import os
from tqdm.autonotebook import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .utils.utils import cond_mkdir, PerfTimer, prepare_batch
import time

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Trainer(object):
    """
    A generic trainer class adapted from NGLOD: 
    https://github.com/nv-tlabs/nglod

    Base class for the trainer.

    The default overall flow of things:

    init()
    |- set_dataset()
    |- set_network()
    |- set_optimizer()
    |- set_renderer()
    |- set_logger()

    train():
        for every epoch:
            pre_epoch()

            iterate()
                step()

            post_epoch()
            |- log_tb()
            |- render_tb()
            |- save_model()
            |- resample()

            validate()

    Each of these submodules can be overriden, or extended with super().

    """

    #######################
    # __init__
    #######################
    
    def __init__(self, 
                 cfg, 
                 model, 
                 loss_func,
                 train_dataloader,
                 val_dataloader=None, 
                 device='cuda:0', 
                 dtype=torch.float32):
        """Constructor.
        """
        self.cfg = cfg 
        self.verbose = cfg['verbose']
        self.model = model
        self.loss_func = loss_func
        
        # Set device to use
        self.use_cuda = torch.cuda.is_available()
        self.device = device
        if self.verbose:
            logger.info(f'Using {device} with CUDA v{torch.version.cuda}.')
        
        # Dataloaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Optimizer
        self.model.to(self.device)
        self.set_optimizer()

        # Logging
        self.set_logging()
        

    def set_optimizer(self):
        """
        Override this function to use custom optimizers. (Or, just add things to this switch)
        """
        # Load pretrained model if provided:
        if self.cfg['pretrained_model'] is not None:
            logger.info(f"Loading pretrained model: {self.cfg['pretrained_model']}")
            checkpoint = torch.load(self.cfg['pretrained_model'])
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # Set geometry optimizer
        if self.cfg['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg['learning_rate'])
        elif self.cfg['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.cfg['learning_rate'])
        elif self.cfg['optimizer'] == 'lbfgs':
            raise NotImplementedError("LBFGS optimizer not implemented yet.")
        else:
            raise ValueError(f"Invalid optimizer: {self.cfg['optimizer']}.")
        
        if self.verbose:
            logger.info(f"Configured {self.cfg['optimizer']} optimizer with lr={self.cfg['learning_rate']}.")

    def set_external_optimizer(self, optimizer):
        logger.info(f"Using external optimizer: {optimizer}.")
        self.optimizer = optimizer

    def set_logging(self):
        # Custom evaluation 
        self.eval_metric = self.cfg['eval_metric']
        self.eval_best_loss = None
        self.eval_every = self.cfg['eval_every']
        self.ckpt_every = self.cfg['ckpt_every']
        self.log_dir = self.cfg['log_dir']
        self.ckpt_dir = os.path.join(self.log_dir, 'ckpt')
        self.tb_dir = os.path.join(self.log_dir, 'tensorboard')
        cond_mkdir(self.log_dir)
        cond_mkdir(self.ckpt_dir)
        cond_mkdir(self.tb_dir)
        self.train_dict = {'epochs': [], 'elapsed_time': [], 'epoch_time': [], 'total_loss': []}
        self.val_dict = {'epochs': [], 'total_loss': []}
        self.custom_eval_dict = {'epochs': []}
        self.custom_eval_funcs = dict()
        self.writer = SummaryWriter(self.tb_dir)
        self.timer = PerfTimer(activate=True)

    
    def get_last_epoch(self):
        if len(self.train_dict['epochs']) > 0:
            return self.train_dict['epochs'][-1]
        else:
            return 0


    def pre_epoch(self, epoch):
        """
        Override this function to change the pre-epoch preprocessing.
        This function runs once before the epoch.
        """
        if self.eval_every > 0 and epoch % self.eval_every == 0:
            self.run_eval(epoch)


    def post_epoch(self, epoch):
        """
        Override this function to change the post-epoch post processing.

        By default, this function logs to Tensorboard, renders images to Tensorboard, saves the model,
        and resamples the dataset.

        To keep default behaviour but also augment with other features, do 
          
          super().post_epoch(self, epoch)

        in the derived method.
        """

        # Save check point
        if self.ckpt_every > 0 and epoch % self.ckpt_every == 0:
            self.save_model(epoch, f"ckpt_{epoch}")
        

    
    #######################
    # train
    #######################
    
    def train(self):
        """
        Override this if some very specific training procedure is needed.
        """
        self.total_steps = 0
        self.train_start_time = time.process_time()
        self.total_epoch_time = 0
        epoch = 0
        while epoch < self.cfg['epochs']:   
            
            self.pre_epoch(epoch)
            
            self.train_epoch(epoch)

            self.post_epoch(epoch)

            epoch += 1
        
        if self.eval_every > 0: self.run_eval(epoch)
        if self.ckpt_every > 0: self.save_model(epoch, "final")


    def train_epoch(self, epoch):
        """
        Override this if there is a need to override the dataset iteration.
        """
        self.model.train()
        cpu_time, gpu_time = 0, 0
        for step, (model_input, gt) in enumerate(self.train_dataloader):
            self.timer.reset()
            model_input, gt = prepare_batch(model_input, gt)
            self.optimizer.zero_grad()
            
            # Loss 
            total_loss = 0.
            loss_dict = self.loss_func.compute(self.model, model_input, gt)
            for loss_name, loss in loss_dict.items():
                single_loss = loss.mean()
                total_loss += single_loss

            # Backward step
            if not torch.isnan(total_loss):
                total_loss.backward(retain_graph=False)
                self.optimizer.step()
            else:
                logger.warning(f"Loss at epoch {epoch} is nan! Skip backward step.")

            # Logging
            self.total_steps += 1
            if self.verbose and step % 10 == 0:
                logger.info(f"Train epoch {epoch} step {step} | train_loss={total_loss.item():.2e}.")
            step_cpu_time, step_gpu_time = self.timer.check()
            cpu_time += step_cpu_time
            gpu_time += step_gpu_time
        self.total_epoch_time += gpu_time

    
    def relative_param_change(self, epoch, params_list):
        self.params_curr = [param.clone().detach() for param in params_list]
        if self.params_prev is None:
            self.params_prev = self.params_curr
            return np.inf
        num_sq = 0
        den_sq = 0
        for idx in range(len(params_list)):
            num_sq += torch.sum((self.params_curr[idx] - self.params_prev[idx])**2)
            den_sq += torch.sum((self.params_prev[idx])**2)
        self.params_prev = self.params_curr
        return torch.sqrt(num_sq / den_sq)
        
    

    #######################
    # eval
    #######################
    def register_eval_func(self, name, func):
        self.custom_eval_funcs[name] = func
        self.custom_eval_dict[name] = []

    
    def run_eval(self, epoch):
        self.eval(epoch, 'train')
        self.eval(epoch, 'val')
        # Run custom evaluation
        self.custom_eval_dict['epochs'].append(epoch)
        for name, func in self.custom_eval_funcs.items():
            self.custom_eval_dict[name].append(
                func(epoch, self.cfg, self.model, self.loss_func, self.train_dataloader, self.val_dataloader)
            )


    def eval(self, epoch, mode='train'):
        self.model.eval()
        eval_dict = {}
        if mode == 'train':
            dataloader = self.train_dataloader
            target_dict = self.train_dict
        elif mode == 'val':
            dataloader = self.val_dataloader
            target_dict = self.val_dict
        else:
            raise ValueError(f"Invalid eval mode: {mode}!")
        if dataloader is None:
            return
        # with tqdm(total=len(dataloader)) as pbar:
        for step, (model_input, gt) in enumerate(dataloader):
            model_input, gt = self.prepare_batch(model_input, gt)
            loss_dict = self.loss_func.compute(self.model, model_input, gt)
            for loss_name, loss in loss_dict.items():
                single_loss = loss.mean()
                if loss_name not in eval_dict.keys():
                    eval_dict[loss_name] = []
                eval_dict[loss_name].append(single_loss.item())

        target_dict['epochs'].append(epoch) 
        total_loss = 0.0
        for loss_name in eval_dict.keys():
            loss_avg = np.mean(np.asarray(eval_dict[loss_name]))
            if loss_name not in target_dict.keys():
                target_dict[loss_name] = []
            target_dict[loss_name].append(loss_avg)
            total_loss += loss_avg
            self.writer.add_scalar(f"{mode}/{loss_name}", loss_avg, epoch)
            if self.verbose:
                logger.info(f"Epoch {epoch} {mode} {loss_name}: {loss_avg:.2e}")
        target_dict['total_loss'].append(total_loss)
        # if self.verbose:
        #     logger.info(f"Epoch {epoch} {mode} total loss: {total_loss:.2e}")
        
        # Train timing
        if mode == 'train':
            target_dict['elapsed_time'].append(time.process_time() - self.train_start_time) 
            target_dict['epoch_time'].append(self.total_epoch_time)

        # Save best model so far
        if mode == 'val' and self.eval_metric is not None:
            if self.eval_best_loss is None or self.eval_best_loss > target_dict[self.eval_metric][-1]:
                self.eval_best_loss = target_dict[self.eval_metric][-1]
                self.save_model(epoch, 'best_model')
                logger.info(f"NEW best model: epoch = {epoch}, eval {self.eval_metric} loss: {self.eval_best_loss:.2e}.")
            else:
                logger.info(f"Best so far {self.eval_metric} loss: {self.eval_best_loss:.2e}.")
            


    def save_model(self, epoch, ckpt_name):
        ckpt_file = os.path.join(self.ckpt_dir, f"{ckpt_name}.pt")
        torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_dict': self.train_dict,
                'val_dict': self.val_dict
            }, 
            ckpt_file
        )
        # Support custom save method implemented in the model class
        if hasattr(self.model, 'save') and callable(self.model.save):
            self.model.save(self.ckpt_dir, ckpt_name)

    
    #######################
    # Helper functions
    #######################

    def prepare_batch(self, model_input, gt):
        model_input = {key: value.to(self.device) for key, value in model_input.items()}
        # TODO: move to derived SDF trainer class?
        if 'coords' in model_input.keys():
            model_input['coords'].requires_grad_(True)
        gt = {key: value.to(self.device) for key, value in gt.items()}
        return model_input, gt
    

    def plot_training_curve(self, ax, x_key, y_key, yscale='log', v_keys=[], label=None):
        if label is None:
            label = y_key
        ax.plot(self.train_dict[x_key], self.train_dict[y_key], label=label)
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        if len(v_keys) > 0:
            for key in v_keys:
                if key not in self.train_dict.keys():
                    logger.warning(f"Key {key} not in training dict!")
                    continue
                ax.axvline(x=self.train_dict[key], label=key, color='gray', linestyle='--')
        ax.legend()
        ax.set_title(f"loss term: {y_key}")
        ax.set_yscale(yscale)


###############################
#####    Grid Trainer    ######
###############################


class GridTrainer(Trainer):
    def __init__(self, 
                 cfg, 
                 model, 
                 loss_func,
                 train_dataloader,
                 val_dataloader=None, 
                 device='cuda:0', 
                 dtype=torch.float32):
        """Constructor.
        """
        self.cfg = cfg 
        self.verbose = cfg['verbose']
        self.model = model
        self.loss_func = loss_func
        
        # Set device to use
        self.use_cuda = torch.cuda.is_available()
        self.device = device
        if self.verbose: logger.info(f'Using {device} with CUDA v{torch.version.cuda}.')
        
        # Dataloaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Optimizer
        self.model.to(self.device)
        self.set_optimizer()

        # Logging
        self.set_logging()

    
    def reset_convergence_check(self):
        self.params_prev = None
        self.params_curr = None
        self.relchange = np.inf
        self.epochs_in_level = 0

    
    def set_optimizer(self):
        self.relchange_tol = self.cfg['relchange_tol']
        self.max_epochs_in_level = self.cfg['max_epochs_in_level']
        self.grid_training_mode = self.cfg['grid_training_mode']

        # Load pretrained model if provided:
        if self.cfg['pretrained_model'] is not None:
            logger.info(f"Loading pretrained model: {self.cfg['pretrained_model']}")
            checkpoint = torch.load(self.cfg['pretrained_model'])
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # Set geometry optimizer
        if self.cfg['optimizer'] == 'adam':
            optim_class = optim.Adam
        elif self.cfg['optimizer'] == 'sgd':
            optim_class = optim.SGD
        else:
            raise ValueError(f"Invalid optimizer: {self.cfg['optimizer']}.")
        
        # Create optimizer for each grid level
        self.level_optimizers = []
        if self.grid_training_mode != 'joint':
            for level in range(self.model.num_levels):
                optimizer = optim_class(self.model.params_at_level(level), lr=self.cfg['learning_rate']) 
                self.level_optimizers.append(optimizer)

        # Create a joint optimizer for final finetune
        self.joint_optimizer = optim_class(self.model.parameters(), lr=self.cfg['learning_rate']) 
        
        # Set active optimizer
        self.reset_convergence_check()
        if self.grid_training_mode == 'coordinate' or self.grid_training_mode == 'coordinate+joint':
            self.active_level = 0
            self.optimizer = self.level_optimizers[0]
        elif self.grid_training_mode == 'joint':
            self.active_level = self.model.num_levels
            self.optimizer = self.joint_optimizer
        else:
            raise ValueError(f"Invalid grid training mode: {self.grid_training_mode}")
        
        if self.verbose:
            logger.info(f"Using grid training mode: {self.grid_training_mode}.")
            logger.info(f"Configured {len(self.level_optimizers)} {self.cfg['optimizer']} optimizers with lr={self.cfg['learning_rate']}.")

    
    def pre_epoch(self, epoch):
        """
        Override this function to change the pre-epoch preprocessing.
        This function runs once before the epoch.
        """
        super().pre_epoch(epoch)
        # Detect convergence at current level
        if self.relchange < self.relchange_tol or self.epochs_in_level >= self.max_epochs_in_level:
            if self.active_level < self.model.num_levels:
                self.train_dict[f'level{self.active_level}_last_epoch'] = epoch
                # self.save_model(epoch, f'level{self.active_level}_ckpt')
                if self.verbose: logger.info(f"Level {self.active_level}: relchange {self.relchange:.1e}, epochs_in_level={self.epochs_in_level}, epoch={epoch}.")
                self.active_level += 1
                if self.active_level >= self.model.num_levels:
                    if self.grid_training_mode == 'coordinate':
                        if self.verbose: logger.info(f"Continue at finest level.")
                    if self.grid_training_mode == 'coordinate+joint':
                        if self.verbose: logger.info(f"Switching to joint training.")
                        self.optimizer = self.joint_optimizer
                else:
                    if self.verbose: logger.info(f"Switching to level {self.active_level}.")
                    self.optimizer = self.level_optimizers[self.active_level]
                # for level in range(self.model.num_levels):
                #     logger.info(f"Level {level} norm = {self.model.features[level].norm()}")
                self.reset_convergence_check()
        self.epochs_in_level += 1

    
    def eval(self, epoch, mode='train'):
        super().eval(epoch, mode)
        if mode == 'train':
            self.relchange = self.relative_param_change(epoch, self.model.params_at_level(self.active_level))
            if self.verbose:
                logger.info(f"Epoch {epoch}: curr_level={self.active_level}, rel_change={self.relchange:.4f}.")
            if 'relchange' not in self.train_dict.keys():
                self.train_dict['relchange'] = []
            self.train_dict['relchange'].append(self.relchange)