import numpy as np

import torch
from torch import nn

from rl_toolbox.utils.network_utils import mlp


class DistQNet(nn.Module):
    def __init__(
        self, obs_space_size, action_space_size, hidden_sizes, activation
    ) -> None:
        super().__init__()

    def select_action(self, obs, eps=None):
        eps = eps or 0.0
        if np.random.rand() < eps:
            action = np.random.choice(self.actions)
        else:
            with torch.no_grad():
                qvalues = self(obs)
                action = torch.argmax(qvalues).cpu().item()
        return action


class SimpleQNet(DistQNet):
    def __init__(
        self, obs_space_size, action_space_size, hidden_sizes, activation=nn.ReLU
    ) -> None:
        super().__init__(obs_space_size, action_space_size, hidden_sizes, activation)
        if isinstance(activation, str):
            activation = getattr(nn, activation)

        sizes = [obs_space_size] + list(hidden_sizes) + [action_space_size]
        self.actions = np.arange(action_space_size, dtype=np.int32)

        self.q_net = mlp(sizes, activation, nn.Identity)

    def forward(self, obs):
        return self.q_net(obs)


class DuelingQNet(DistQNet):
    def __init__(
        self, obs_space_size, action_space_size, hidden_sizes, activation=nn.ReLU
    ) -> None:
        super().__init__(obs_space_size, action_space_size, hidden_sizes, activation)

        sizes = [obs_space_size] + list(hidden_sizes) + [action_space_size]
        self.actions = np.arange(action_space_size, dtype=np.int32)

        self.layers = mlp(
            sizes[:-1], activation=activation, output_activation=activation
        )

        self.v_layer = nn.Linear(sizes[-2], 1)
        self.a_layer = nn.Linear(sizes[-2], sizes[-1])

    def forward(self, obs):
        x = self.layers(obs)
        values = self.v_layer(x)
        advs = self.a_layer(x)
        return values + advs - advs.mean(-1, keepdim=True)


class ContQNet(nn.Module):
    """continuous Q network"""

    def __init__(
        self, obs_space_size, action_space_size, hidden_sizes, activation
    ) -> None:
        super().__init__()
