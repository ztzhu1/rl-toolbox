import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from rl_toolbox.utils.network_utils import mlp


class Actor(nn.Module):
    def get_policy(self, obs):
        raise NotImplementedError()

    def log_prob_from_policy(self, policy, action):
        raise NotImplementedError()

    def log_prob(self, obs, action):
        policy = self.get_policy(obs)
        return self.log_prob_from_policy(policy, action)


class CategoricalActor(Actor):
    def __init__(
        self, obs_space_size, action_space_size, hidden_sizes, activation=nn.Tanh
    ) -> None:
        super().__init__()

        sizes = [obs_space_size] + list(hidden_sizes) + [action_space_size]
        self.logits_net = mlp(sizes, activation, nn.Identity)

    def get_policy(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def log_prob_from_policy(self, policy, action):
        return policy.log_prob(action)


class GuassianActor(Actor):
    def __init__(
        self,
        obs_space_size,
        action_space_size,
        hidden_sizes,
        activation=nn.Tanh,
        init_sigma=0.5,
    ) -> None:
        super().__init__()

        sizes = [obs_space_size] + list(hidden_sizes) + [action_space_size]
        self.mu_net = mlp(sizes, activation, nn.Identity)
        self.sigmas = nn.Parameter(
            init_sigma * torch.ones(action_space_size, dtype=torch.float32)
        )

    def get_policy(self, obs):
        mus = self.mu_net(obs)
        return Normal(mus, self.sigmas)

    def log_prob_from_policy(self, policy, action):
        return policy.log_prob(action).sum(axis=-1)


class Critic(nn.Module):
    def __init__(self, obs_space_size, hidden_sizes, activation=nn.Tanh) -> None:
        super().__init__()

        sizes = [obs_space_size] + list(hidden_sizes) + [1]
        self.v_net = mlp(sizes, activation, nn.Identity)

    def forward(self, obs):
        return self.v_net(obs).squeeze(axis=-1)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_space_size,
        action_space_size,
        hidden_sizes,
        continuous,
        activation=nn.Tanh,
    ) -> None:
        super().__init__()

        if continuous:
            self.actor = GuassianActor(
                obs_space_size, action_space_size, hidden_sizes, activation
            )
        else:
            self.actor = CategoricalActor(
                obs_space_size, action_space_size, hidden_sizes, activation
            )
        self.critic = Critic(obs_space_size, hidden_sizes, activation)

    def select_action(self, obs):
        with torch.no_grad():
            policy = self.actor.get_policy(obs)
            action = policy.sample()
            logp = self.actor.log_prob_from_policy(policy, action)
        return action.cpu().numpy(), logp.cpu().numpy()

    def est_value(self, obs):
        with torch.no_grad():
            value = self.critic(obs).cpu().numpy()
        return value
