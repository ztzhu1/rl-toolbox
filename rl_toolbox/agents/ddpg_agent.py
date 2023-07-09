from copy import deepcopy

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.tensor_specs import ContinuousBox
from torchrl.modules import Actor

from rl_toolbox.agents.agent import Agent
from rl_toolbox.utils.backend import check_notebook, get_device
from rl_toolbox.utils.env_utils import make_env
from rl_toolbox.utils.network_utils import mlp, soft_update

in_notebook = check_notebook()
if in_notebook:
    from IPython import display

device = get_device()


class DDPGAgent(Agent):
    """
    refer to
    - https://pytorch.org/rl/tutorials/coding_ddpg.html
    - https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    - https://keras.io/examples/rl/ddpg_pendulum/
    """

    def __init__(
        self, env_name="Pendulum-v1", cp_dir=None, save_buf=False, **cfg
    ) -> None:
        vis_value_names = [
            "loss_actor",
            "loss_critic",
            "episode_reward",
            "avg_frame",
        ]
        saved_attr_names = [
            "env_name",
            "curr_epoch",
            "vis_value_names",
            "env",
            "actor",
            "target_actor",
            "opt_actor",
            "critic",
            "target_critic",
            "opt_critic",
        ]
        super().__init__(
            env_name, cp_dir, save_buf, vis_value_names, saved_attr_names, **cfg
        )

    def one_epoch(self):
        cfg = self._cfg
        clip_grad_norm = cfg["clip_grad_norm"]
        batch_size = cfg["batch_size"]
        steps_per_epoch = cfg["steps_per_epoch"]

        losses_actor = []
        losses_critic = []
        episode_rewards = []
        num_trajs = 0
        for _ in range(steps_per_epoch):
            # interact with env
            data = self._collector.next()
            self._buf.extend(data)
            # log info
            data_next = data["next"]
            if data_next["done"][-1].item():
                episode_rewards.append(
                    data_next["episode_reward"][-1].detach().cpu().item()
                )
                num_trajs += 1
            # update network if buffer contains enough data
            if len(self._buf) >= batch_size:
                data = self._buf.sample(batch_size)
                data_next = data["next"]
                obs = data["observation"]
                act = data["action"]
                next_obs = data_next["observation"]
                not_done = ~data_next["done"]
                actor = self._actor.module
                target_actor = self._target_actor.module

                # opt critic
                q = self._critic(obs, act)
                q_next_target = self._target_critic(next_obs, target_actor(next_obs))
                q_target = (
                    data_next["reward"] + self._cfg["gamma"] * q_next_target * not_done
                )
                loss_critic = getattr(F, cfg["criterion"])(q, q_target)

                self._opt_critic.zero_grad()
                loss_critic.backward()
                if clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self._critic.parameters(), clip_grad_norm)
                self._opt_critic.step()

                losses_critic.append(loss_critic.detach().cpu().item())

                # opt actor
                self._critic.requires_grad_(False)
                q = self._critic(obs, actor(obs))
                loss_actor = -q.mean()

                self._opt_actor.zero_grad()
                loss_actor.backward()
                if clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self._actor.parameters(), clip_grad_norm)
                self._opt_actor.step()

                losses_actor.append(loss_actor.detach().cpu().item())
                self._critic.requires_grad_(True)

                # soft update target net
                tau = cfg["tau"]
                soft_update(self._actor, self._target_actor, tau)
                soft_update(self._critic, self._target_critic, tau)

        log = dict()
        log["loss_actor"] = np.mean(losses_actor)
        log["loss_critic"] = np.mean(losses_critic)
        if len(episode_rewards) == 0:
            log["episode_reward"] = 0.0
        else:
            log["episode_reward"] = np.mean(episode_rewards)
        num_trajs = np.max([num_trajs, 1])
        log["avg_frame"] = cfg["steps_per_epoch"] / num_trajs
        return log

    @classmethod
    def from_checkpoint(cls, cp_dir, epochs, render_mode=None, **env_kwargs):
        def human_render_hook(cfg):
            # Disable noise
            cfg["noise_scale"] = 0.0

        return super().from_checkpoint(
            cp_dir, epochs, render_mode, human_render_hook, **env_kwargs
        )

    def _init_all(self):
        self._init_env()
        self._init_actor()
        self._init_critic()
        self._init_opt()
        self._init_collector()
        self._init_buf()
        self._init_vis()

    def _init_env(self):
        cfg = self._cfg
        self._env = make_env(
            self._env_name,
            cfg["init_stats_param"],
            seed=cfg["seed"],
            device=device,
            **cfg["env_kwargs"],
        )

        # DDPG only supports continuous action space
        assert isinstance(self._env.action_spec.space, ContinuousBox)
        self._cont_act = True

    def _init_actor(self):
        cfg = self._cfg

        obs_dim = self._env.observation_spec["observation"].shape[0]
        act_dim = self._env.action_spec.shape[0]

        actor = DDPGActor(
            obs_dim,
            cfg["hidden_sizes"],
            act_dim,
            cfg["activation"],
            self._env.action_spec.space,
            cfg["noise_scale"],
        )
        actor = Actor(actor, in_keys=["observation"], out_keys=["action"])
        self._actor = actor.to(device)

        target_actor = deepcopy(actor)
        target_actor.requires_grad_(False)
        self._target_actor = target_actor.to(device)

    def _init_critic(self):
        cfg = self._cfg

        obs_dim = self._env.observation_spec["observation"].shape[0]
        act_dim = self._env.action_spec.shape[0]

        critic = DDPGCritic(
            obs_dim,
            act_dim,
            cfg["obs_encoding_sizes"],
            cfg["act_encoding_sizes"],
            cfg["hidden_sizes"],
            cfg["activation"],
            cfg["encoding_activation"],
        )
        self._critic = critic.to(device)

        target_critic = deepcopy(critic)
        target_critic.requires_grad_(False)
        self._target_critic = target_critic.to(device)

    def _init_opt(self):
        self._opt_actor = Adam(self._actor.parameters(), lr=self._cfg["lr_actor"])
        self._opt_critic = Adam(self._critic.parameters(), lr=self._cfg["lr_critic"])
        # TODO (ztzhu): use lr_scheduler

    def _init_collector(self):
        cfg = self._cfg
        self._collector = SyncDataCollector(
            self._env,
            self._actor,
            frames_per_batch=1,
            total_frames=cfg["steps_per_epoch"] * cfg["epochs"],
            max_frames_per_traj=cfg["max_steps_per_traj"],
        )

    def _init_buf(self):
        # TODO (ztzhu): use prioritized replay buffer
        cfg = self._cfg
        self._buf = ReplayBuffer(
            storage=LazyTensorStorage(cfg["buf_cap"]),
            sampler=SamplerWithoutReplacement(),
            batch_size=cfg["batch_size"],
        )

    def _get_dflt_cfg(self):
        cfg = dict()
        cfg = {
            # --- training ---
            "device": device,
            "seed": 56,
            "epochs": 100,
            "save_cp_freq": 5,
            # --- network ---
            "hidden_sizes": [256, 256],
            "activation": "ReLU",
            "gamma": 0.99,
            "criterion": "mse_loss",
            "clip_grad_norm": 1.0,
            # --- env ---
            "init_stats_param": 1000,
            "max_steps_per_traj": -1,
            "steps_per_epoch": 1024,
            "env_kwargs": {},
            # --- DDPG specific ---
            "batch_size": 128,
            "buf_cap": 100_000,
            "lr_actor": 0.0004,
            "lr_critic": 0.001,
            "tau": 0.005,
            "obs_encoding_sizes": [32, 32],
            "act_encoding_sizes": [32, 32],
            "encoding_activation": "ReLU",
            "noise_scale": 0.05,
        }
        cfg["num_batch"] = int(cfg["steps_per_epoch"] // cfg["batch_size"])
        return cfg


class DDPGActor(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden_sizes,
        act_dim,
        activation,
        action_space: ContinuousBox,
        noise_scale,
    ) -> None:
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self._norm_actor = mlp(sizes, activation, nn.Tanh)
        self._act_min = action_space.minimum.to(device)
        self._act_max = action_space.maximum.to(device)
        self._noise_scale = noise_scale

    def forward(self, obs):
        norm_act = self._norm_actor(obs)
        norm_act = norm_act + torch.randn_like(norm_act) * self._noise_scale
        norm_act.clamp_(-1.0, 1.0)
        return (
            norm_act * (self._act_max - self._act_min) / 2.0
            + (self._act_max + self._act_min) / 2.0
        )


class DDPGCritic(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        obs_encoding_sizes,
        act_encoding_sizes,
        hidden_sizes,
        activation,
        encoding_activation,
    ) -> None:
        super().__init__()
        self._obs_encoder = mlp(
            [obs_dim] + obs_encoding_sizes,
            activation=encoding_activation,
            output_activation=nn.Identity,
        )
        self._act_encoder = mlp(
            [act_dim] + act_encoding_sizes,
            activation=encoding_activation,
            output_activation=nn.Identity,
        )
        self._qnet = mlp(
            [obs_encoding_sizes[-1] + act_encoding_sizes[-1]] + hidden_sizes + [1],
            activation=activation,
            output_activation=nn.Identity,
        )

    def forward(self, obs, action):
        x = torch.cat([self._obs_encoder(obs), self._act_encoder(action)], dim=-1)
        return self._qnet(x)
