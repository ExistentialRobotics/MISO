import os
import sys
import time
import numpy as np
import pprint
import argparse
import torch
from enum import Enum

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def gradient3d(x, f, method='finitediff', finite_diff_eps=1e-2, create_graph=True):
    assert x.ndim == 2
    assert x.shape[-1] == 3

    if method == 'finitediff':
        eps_x = torch.tensor([finite_diff_eps, 0.0, 0.0], device=x.device, dtype=x.dtype)
        eps_y = torch.tensor([0.0, finite_diff_eps, 0.0], device=x.device, dtype=x.dtype)
        eps_z = torch.tensor([0.0, 0.0, finite_diff_eps], device=x.device, dtype=x.dtype)

        grad = torch.cat([f(x + eps_x) - f(x - eps_x),
                        f(x + eps_y) - f(x - eps_y),
                        f(x + eps_z) - f(x - eps_z)], dim=-1)
        grad = grad / (finite_diff_eps * 2.0)
    elif method == 'autograd':
        # with torch.enable_grad():
        # x = x.requires_grad_(True)
        assert x.requires_grad, "requires_grad need to be true for autograd!"
        y = f(x)
        grad = torch.autograd.grad(y, x, 
                                    grad_outputs=torch.ones_like(y), create_graph=create_graph)[0]

    else:
        raise ValueError("Unknown method: {}".format(method))

    return grad


def gradient2d(x, f, method='finitediff', finite_diff_eps=1e-2, create_graph=True):
    """Given 2D spatial coordinates x and a scalar-valued field f
    Estimate the gradient of f(x) evaluated at x using finite diff.

    Args:
        x (_type_): (N, 2) or (H, W, 2) tensor specifying input 2D coordinates
        f (_type_): _description_
        finite_diff_eps:  epsilon used for finite diff
        method: method used to compute gradient

    Returns:
        g (_type_): (N, 2) or (H, W, 2) tensor where g[i, :] or g[i, j, :] is the corresponding gradient
    """
    assert x.shape[-1] == 2

    if method == 'finitediff':
        eps_x = torch.tensor([finite_diff_eps, 0.0], device=x.device, dtype=x.dtype)
        eps_y = torch.tensor([0.0, finite_diff_eps], device=x.device, dtype=x.dtype)

        grad = torch.cat([f(x + eps_x) - f(x - eps_x),
                        f(x + eps_y) - f(x - eps_y)], dim=-1)
        grad = grad / (finite_diff_eps * 2.0)
    elif method == 'autograd':
        # with torch.enable_grad():
        # x = x.requires_grad_(True)
        assert x.requires_grad, "requires_grad need to be true for autograd!"
        y = f(x)
        grad = torch.autograd.grad(y, x, 
                                    grad_outputs=torch.ones_like(y), create_graph=create_graph)[0]

    else:
        raise ValueError("Unknown method: {}".format(method))

    return grad