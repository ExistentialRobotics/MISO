import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.utils import normalize_coordinates
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class BaseNet(nn.Module):
    """A basic interface for a model.
    Actual models will derive from this class.

    Args:
        nn (_type_): _description_
    """
    def __init__(self,
        cfg: dict, 
        device = 'cpu',
        dtype = torch.float32
    ):
        super().__init__()
        self.cfg = cfg
        self.d = self.cfg['spatial_dim']
        self.device = device
        self.dtype = dtype
        assert self.d == 2 or self.d == 3
        cfg_grid = self.cfg['grid']

        # Load bound as d-by-2 tensor of min and max bound
        self.bound = torch.tensor(
            cfg_grid['bound'], 
            device=device,
            dtype=dtype
        )   
        assert self.bound.shape == (self.d, 2)
        # logger.debug(f"Grid model defined for bound: \n {self.bound}")
        

    def forward(self, x: torch.Tensor):
        """Predict the field value at input positions x in the *global/world* frame

        Args:
            x (torch.Tensor): input positions x in the *global* frame
        """
        raise NotImplementedError
    

    def params_at_level(self, level):
        raise NotImplementedError
    
    
    def print_trainable_params(self):
        print("\n === Trainable parameters === ")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}")
        print("=== END trainable parameters === \n ")