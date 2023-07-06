from copy import deepcopy
import json
import os
import pathlib

import keyboard
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

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

from rl_toolbox.utils.backend import check_notebook, get_device
from rl_toolbox.utils.cp_utils import load_checkpoint, search_cp_file_name
from rl_toolbox.utils.env_utils import make_env
from rl_toolbox.utils.network_utils import mlp
from rl_toolbox.visualization.monitor import plot_value

in_notebook = check_notebook()
if in_notebook:
    from IPython import display

device = get_device()


class DQNAgent:
    """
    refer to
    - https://pytorch.org/rl/tutorials/coding_dqn.html
    """

    def __init__(
        self, cp_dir=None, env_name="LunarLander-v2", save_buf=False, **cfg
    ) -> None:
        if (
            cp_dir is not None
            and cfg["save_cp_freq"] is not None
            and os.path.exists(cp_dir)
        ):
            raise FileExistsError(f"Directory {cp_dir} already exists!")
        self._cp_dir = cp_dir
        self._env_name = env_name
        self._save_buf = save_buf
        self._cfg = self._get_dflt_cfg()
        self._update_cfg(cfg)

        self._curr_epoch = 0
        self._vis_value_names = [
            "loss",
            "episode_reward",
            "avg_frame",
        ]

    def learn(self):
        self._init_all()

        pbar = tqdm(total=self._cfg["epochs"] - self._curr_epoch)
        while self._curr_epoch < self._cfg["epochs"]:
            self._curr_epoch += 1

            log = self.one_epoch()
            self._process_log(log)
            self._update_vis()

            if self._need_to_save():
                self._save_checkpoint()

            pbar.update()

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
                self._update_target_net()

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
        cfg = torch.load(os.path.join(cp_dir, "config.pt"), map_location="cpu")
        log = pd.read_csv(os.path.join(cp_dir, "log.csv"))
        data = torch.load(search_cp_file_name(cp_dir, epochs), map_location="cpu")
        print(json.dumps(cfg, sort_keys=True, indent=4, default=str))
        cfg["init_stats_param"] = data["env"]

        agent = cls(cp_dir=None, env_name=data["env_name"], **cfg)
        agent._curr_epoch = data["curr_epoch"]
        agent._vis_value_names = data["vis_value_names"]
        agent._log = log[:epochs]

        if render_mode is not None:
            # testing mode
            cfg = agent._cfg
            assert render_mode == "human"
            cfg["env_kwargs"].update(env_kwargs)
            cfg["env_kwargs"]["render_mode"] = render_mode
            cfg["epochs"] = 10 * cfg["steps_per_epoch"]
            cfg["steps_per_epoch"] = 1
            if cfg["max_steps_per_traj"] < 0:
                cfg["max_steps_per_traj"] = 1000
            # Disable epilson greedy
            cfg["eps_init"] = 0.0
            cfg["eps_end"] = 0.0

        agent._init_all()
        agent._update_vis()

        agent._actor.load_state_dict(data["actor"])
        agent._opt.load_state_dict(data["opt"])
        if render_mode != "human" and "buf" in data:
            agent._buf = data["buf"]

        return agent

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

    def _update_target_net(self):
        new_params = self._actor.parameters()
        target_sd = self._target_actor.state_dict()
        tau = self._cfg["tau"]
        with torch.no_grad():
            for k, target_p, new_p in zip(
                target_sd.keys(), target_sd.values(), new_params
            ):
                target_sd[k] = target_p * (1.0 - tau) + new_p * tau
        self._target_actor.load_state_dict(target_sd)

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

    def _init_vis(self):
        if self._vis_value_names is None:
            return

        n = len(self._vis_value_names)
        if hasattr(self, "_fig"):
            plt.close(self._fig)
        fig = plt.figure(figsize=[5.5 * n, 4])
        self._fig = fig
        self._axes = dict()
        for i, name in enumerate(self._vis_value_names):
            self._axes[name] = fig.add_subplot(1, n, i + 1)
        self._display_handle = None
        if in_notebook:
            self._display_handle = display.display(fig, display_id=True)

    def _update_vis(self):
        if self._vis_value_names is None:
            return
        if self._log is None:
            return

        for name in self._vis_value_names:
            plot_value(
                self._axes[name],
                self._curr_epoch,
                self._log[self._log[name].notna()][name].to_numpy(),
                name,
                self._display_handle,
            )

    def _process_log(self, log: dict):
        # lazy init
        if not hasattr(self, "_log"):
            self._log = pd.DataFrame(columns=log.keys())

        self._log = self._log.append(log, ignore_index=True)

    def _need_to_save(self):
        save_cp_freq = self._cfg["save_cp_freq"]
        return (
            self._cp_dir is not None
            and save_cp_freq is not None
            and self._curr_epoch % save_cp_freq == 0
        )

    def _save_checkpoint(self):
        cfg_path = os.path.join(self._cp_dir, f"config.pt")
        log_path = os.path.join(self._cp_dir, f"log.csv")
        data_path = os.path.join(self._cp_dir, f"epochs_{self._curr_epoch}.pt")

        pathlib.Path(self._cp_dir).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(cfg_path):
            torch.save(self._cfg, cfg_path)
        self._log.to_csv(log_path, index=False)

        data = {
            "entry_point": "ppo_agent.PPOAgent",
            "env_name": self._env_name,
            "curr_epoch": self._curr_epoch,
            "vis_value_names": self._vis_value_names,
            "env": self._env.state_dict(),
            "actor": self._actor.state_dict(),
            "opt": self._opt.state_dict(),
        }
        if self._save_buf:
            data["buf"] = self._buf
        torch.save(data, data_path)

    def _get_dflt_cfg(self):
        cfg = dict()
        cfg = {
            # --- training ---
            "device": device,
            "seed": 56,
            "epochs": 600,
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

    def _update_cfg(self, cfg: dict):
        for key in cfg.keys():
            if key not in self._cfg:
                raise KeyError(f"Invalid key {key} in cfg")

        self._cfg.update(cfg)
        cfg = self._cfg

        num_batch = int(cfg["steps_per_epoch"] // cfg["batch_size"])
        cfg["num_batch"] = num_batch
        steps_per_epoch = num_batch * cfg["batch_size"]
        if steps_per_epoch != cfg["steps_per_epoch"]:
            print(
                f"\x1b[1;33mWarning: `steps_per_epoch` is truncated to {steps_per_epoch}!\x1b[0m"
            )
            cfg["steps_per_epoch"] = steps_per_epoch
        if cfg["steps_per_epoch"] < cfg["batch_size"]:
            raise ValueError(
                "`steps_per_epoch` (%d) < `batch_size` (%d)!"
                % (cfg["steps_per_epoch"], cfg["batch_size"])
            )


def test_agent(cp_dir, epochs, **env_kwargs):
    agent = DQNAgent.from_checkpoint(cp_dir, epochs, "human", **env_kwargs)

    for data in agent._collector:
        data_next = data["next"]
        if data_next["done"] or data_next["truncated"]:
            print(
                f"step_count: %d, traj_reward: %lf"
                % (
                    data_next["step_count"].cpu().item(),
                    data_next["episode_reward"].cpu().item(),
                )
            )
        if keyboard.is_pressed("Esc"):
            break
    plt.close(agent._fig)
