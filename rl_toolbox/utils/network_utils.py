import numpy as np
from scipy.signal import lfilter

import torch
from torch import nn


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(0, len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def update_target_net(net: nn.Module, target_net: nn.Module, tau=1.0):
    net_state = net.state_dict()
    target_state = target_net.state_dict()

    for key in net_state:
        target_state[key] = net_state[key] * tau + target_state[key] * (1.0 - tau)
    target_net.load_state_dict(target_state)


def freeze_grad(net: nn.Module):
    for p in net.parameters():
        p.requires_grad = False


def unfreeze_grad(net: nn.Module):
    for p in net.parameters():
        p.requires_grad = True


def as_tensor32(x, device=None):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def as_ndarray32(x):
    return np.as_tensor(x, dtype=np.float32)
