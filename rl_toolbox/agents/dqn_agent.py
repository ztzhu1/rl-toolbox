from copy import deepcopy
import os
import pathlib

from matplotlib import pyplot as plt
import numpy as np

from tensordict.tensordict import TensorDict
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.tensor_specs import DiscreteBox
from torchrl.modules import EGreedyWrapper, QValueActor

from rl_toolbox.agents.agent import Agent
from rl_toolbox.utils.backend import check_notebook, get_device
from rl_toolbox.utils.env_utils import make_env
from rl_toolbox.utils.network_utils import mlp, soft_update
from rl_toolbox.visualization.monitor import plot_value

in_notebook = check_notebook()
if in_notebook:
    from IPython import display

device = get_device()


class DQNAgent(Agent):
    """
    refer to
    - https://pytorch.org/rl/tutorials/coding_dqn.html
    """

    def __init__(
        self, cp_dir=None, env_name="MountainCar-v0", save_buf=False, **cfg
    ) -> None:
        vis_value_names = [
            "loss",
            "episode_reward",
            "avg_frame",
        ]
        saved_attr_names = [
            "env_name",
            "curr_epoch",
            "vis_value_names",
            "env",
            "actor",
            "opt",
        ]
        super().__init__(
            cp_dir, env_name, save_buf, vis_value_names, saved_attr_names, **cfg
        )

    def one_epoch(self):
        cfg = self._cfg
        clip_grad_norm = cfg["clip_grad_norm"]
        batch_size = cfg["batch_size"]
        steps_per_epoch = cfg["steps_per_epoch"]

        losses = []
        episode_rewards = []
        num_trajs = 0
        for _ in range(steps_per_epoch):
            # interact with env
            data = self._collector.next()
            # If two actions have the same Q value, the one-hot action tensor
            # will contain two `1`s. We select the first `1` as `GymEnv` does.
            action = data["action"]
            row, col = torch.where(action > 0)
            action[row[1:], col[1:]] = 0
            data["action_value"] /= len(row)
            # store data
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

                self._opt.zero_grad()
                loss = self._get_loss(data)
                loss.backward()
                if clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self._actor.parameters(), clip_grad_norm)
                self._opt.step()
                self._actor.step()
                soft_update(self._actor, self._target_actor, cfg["tau"])

                losses.append(loss.detach().cpu().item())

        log = dict()
        log["loss"] = np.mean(losses)
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
            # Disable epilson greedy
            cfg["eps_init"] = 0.0
            cfg["eps_end"] = 0.0

        return super().from_checkpoint(
            cp_dir, epochs, render_mode, human_render_hook, **env_kwargs
        )

    def _get_loss(self, data: TensorDict):
        data_next = data["next"]
        cfg = self._cfg
        gamma = cfg["gamma"]
        reward = data_next["reward"]
        not_done = ~data_next["done"]
        mask = data["action"].bool()
        qvalue = (
            self._actor.td_module(data)["action_value"].masked_select(mask).view(-1, 1)
        )
        next_qvalue = self._target_actor(data_next)["chosen_action_value"]
        qvalue_target = reward + gamma * next_qvalue * not_done
        loss = getattr(F, cfg["criterion"])(qvalue, qvalue_target)
        return loss

    def _init_all(self):
        self._init_env()
        self._init_actor()
        self._init_target_actor()
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

        # DQN only supports discrete action space
        assert isinstance(self._env.action_spec.space, DiscreteBox)
        self._cont_act = False

    def _init_actor(self):
        cfg = self._cfg

        obs_dim = self._env.observation_spec["observation"].shape[0]
        act_dim = self._env.action_spec.shape[0]
        sizes = [obs_dim] + cfg["hidden_sizes"] + [act_dim]

        actor = mlp(sizes, getattr(nn, cfg["activation"]), nn.Identity)
        # init actor network parameters
        for i in range(0, len(actor), 2):
            nn.init.zeros_(actor[i].bias)
            if i < len(actor) - 2:
                nn.init.orthogonal_(actor[i].weight, np.sqrt(2))
            else:
                nn.init.orthogonal_(actor[i].weight, 0.01)

        actor = QValueActor(actor, spec=self._env.action_spec)
        actor = EGreedyWrapper(
            actor,
            eps_init=cfg["eps_init"],
            eps_end=cfg["eps_end"],
            annealing_num_steps=cfg["steps_per_epoch"] * cfg["epochs"],
        )
        self._actor = actor.to(device)

    def _init_target_actor(self):
        target_actor = deepcopy(self._actor.td_module)
        target_actor.requires_grad_(False)
        self._target_actor = target_actor.to(device)

    def _init_opt(self):
        self._opt = Adam(self._actor.parameters(), lr=self._cfg["lr"])
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
        )

    def _get_dflt_cfg(self):
        cfg = dict()
        cfg = {
            # --- training ---
            "device": device,
            "seed": 56,
            "epochs": 600,
            "save_cp_freq": 20,
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
            # --- DQN specific ---
            "batch_size": 128,
            "buf_cap": 100_000,
            "lr": 0.0002,
            "eps_init": 0.3,
            "eps_end": 0.05,
            "tau": 0.005,
            "double": False,
            "dueling": False,
        }
        cfg["num_batch"] = int(cfg["steps_per_epoch"] // cfg["batch_size"])
        return cfg
