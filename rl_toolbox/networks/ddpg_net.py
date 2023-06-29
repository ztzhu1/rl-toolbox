import numpy as np

import torch
from torch import nn

from rl_toolbox.utils.network_utils import as_tensor32, mlp

from .qnet import ContQNet


class DDPGActor(nn.Module):
    def __init__(
        self,
        obs_space_size,
        action_space_size,
        hidden_sizes,
        action_ranges,
        activation=nn.ReLU,
    ) -> None:
        super().__init__()

        for i, action_range in enumerate(action_ranges):
            if np.isscalar(action_range):
                action_ranges[i] = (-action_range, action_range)

        action_ranges = as_tensor32(action_ranges)
        self.action_ranges = action_ranges
        self.action_range_ratio = action_ranges[:, 1] - action_ranges[:, 0]
        self.action_range_base = action_ranges[:, 0]

        self.p_net = mlp(
            [obs_space_size] + hidden_sizes + [action_space_size], activation, nn.Tanh
        )

        self.action_space_size = action_space_size

    def forward(self, obs):
        action = self.p_net(obs)
        return action * self.action_range_ratio + self.action_range_base

    def clip_action(self, action):
        return np.clip(action, self.action_ranges[:, 0], self.action_ranges[:, 1])

    def get_action(self, obs, noise_scale=None):
        noise_scale = noise_scale or 0.0
        with torch.no_grad():
            action = self(obs).cpu().numpy()
        action += np.random.randn(self.action_space_size) * noise_scale
        return action

    def sample_action(self):
        ratio = self.action_range_ratio.cpu().numpy()
        base = self.action_range_base.cpu().numpy()
        return ratio * np.random.rand(self.action_space_size) + base


class DDPGCritic(ContQNet):
    def __init__(
        self, obs_space_size, action_space_size, hidden_sizes, activation=nn.ReLU
    ) -> None:
        super().__init__(obs_space_size, action_space_size, hidden_sizes, activation)
        if isinstance(activation, str):
            activation = getattr(nn, activation)

        sizes = [obs_space_size + action_space_size] + list(hidden_sizes) + [1]

        self.q_net = mlp(sizes, activation, nn.Identity)

    def forward(self, obs, action):
        qvalues = self.q_net(torch.cat((obs, action), axis=-1))
        return qvalues.squeeze(-1)


class DDPGActorCritic(nn.Module):
    def __init__(
        self,
        obs_space_size,
        action_space_size,
        hidden_sizes,
        action_ranges,
        activation=nn.ReLU,
    ) -> None:
        super().__init__()

        self.actor = DDPGActor(
            obs_space_size, action_space_size, hidden_sizes, action_ranges, activation
        )
        self.critic = DDPGCritic(
            obs_space_size, action_space_size, hidden_sizes, activation
        )

    def forward(self, obs):
        action = self.actor(obs)
        return self.critic(obs, action)
