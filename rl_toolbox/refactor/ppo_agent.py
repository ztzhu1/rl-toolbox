import os
import pathlib

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.tensordict import TensorDict
import torch
from torch import nn
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F
from torch.optim import Adam
from torchrl.collectors import SyncDataCollector
from torchrl.data.tensor_specs import ContinuousBox, DiscreteBox
from torchrl.modules import (
    OneHotCategorical,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from rl_toolbox.utils.backend import check_notebook, get_device
from rl_toolbox.utils.cp_utils import load_checkpoint
from rl_toolbox.utils.env_utils import make_env
from rl_toolbox.utils.network_utils import mlp
from rl_toolbox.visualization.monitor import plot_value

in_notebook = check_notebook()
if in_notebook:
    from IPython import display

device = get_device()


class PPOAgent:
    def __init__(self, cp_dir=None, env_name="LunarLander-v2", **cfg) -> None:
        self._curr_epoch = 0
        self._cp_dir = cp_dir
        self._env_name = env_name
        self._cfg = cfg
        self._set_dflt_cfg()
        self._set_ppo_dflt_cfg()
        self._vis_value_names = [
            "loss",
            "episode_reward",
            "kl_div",
            "avg_frame",
        ]

    def learn(self):
        self._init_all()

        pbar = tqdm(total=self._cfg["epochs"] - self._curr_epoch)
        start_epoch = self._curr_epoch
        for epoch, data in enumerate(self._collector, 1 + start_epoch):
            self._curr_epoch = epoch

            log = self.one_epoch(data)
            self._process_log(log)
            self._update_vis()

            if self._need_to_save():
                self._save_checkpoint()

            pbar.update()

    def one_epoch(self, data):
        """
        refer to
        - https://github.com/openai/baselines/blob/master/baselines/ppo2/ppo2.py
        - https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html
        - https://pytorch.org/rl/tutorials/coding_ppo.html
        """
        cfg = self._cfg
        clip_grad_norm = cfg["clip_grad_norm"]
        batch_size = cfg["batch_size"]
        steps_per_epoch = cfg["steps_per_epoch"]

        with torch.no_grad():
            self._adv_module(data)

        losses = []
        old_dist = self._actor.build_dist_from_params(data)
        for _ in range(cfg["opt_epochs"]):
            indexes = torch.randperm(steps_per_epoch, device=data.device)
            for i in range(0, steps_per_epoch, batch_size):
                batch_data = data[indexes[i : i + batch_size]].clone()
                loc = batch_data["advantage"].mean().item()
                scale = batch_data["advantage"].std().clamp_min(1e-6).item()
                batch_data["advantage"] = (batch_data["advantage"] - loc) / scale

                self._opt.zero_grad()
                loss_vals = self._loss_module(batch_data)
                loss = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                loss.backward()
                if clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self._actor.parameters(), clip_grad_norm)
                    nn.utils.clip_grad_norm_(self._critic.parameters(), clip_grad_norm)
                self._opt.step()

                losses.append(loss.detach().cpu().item())

        curr_dist = self._actor.get_dist(data, params=self._loss_module.actor_params)
        kl_div = kl_divergence(old_dist, curr_dist).mean().detach().cpu().item()

        log = dict()
        log["loss"] = np.mean(losses)
        log["episode_reward"] = (
            data.get(("next", "episode_reward"))[-1].detach().cpu().item()
        )
        log["kl_div"] = kl_div
        num_trajs = len(torch.where(data["step_count"] == 0)[0])
        num_trajs = np.max([num_trajs, 1])
        log["avg_frame"] = cfg["steps_per_epoch"] / num_trajs
        return log

    def _init_all(self):
        self._init_env()
        self._init_actor()
        self._init_critic()
        self._init_opt()
        self._init_adv_module()
        self._init_loss_module()
        self._init_collector()
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
        if isinstance(self._env.action_spec.space, DiscreteBox):
            self._cont_act = False
        else:
            assert isinstance(self._env.action_spec.space, ContinuousBox)
            self._cont_act = True

    def _init_actor(self):
        cfg = self._cfg

        if self._cont_act:
            out_dim_factor = 2
            out_layer = NormalParamExtractor
            actor_out_keys = ["loc", "scale"]
        else:
            out_dim_factor = 1
            out_layer = nn.Identity
            actor_out_keys = ["logits"]

        obs_dim = self._env.observation_spec["observation"].shape[0]
        act_dim = self._env.action_spec.shape[0]
        sizes = [obs_dim] + cfg["hidden_sizes"] + [out_dim_factor * act_dim]

        actor = mlp(sizes, cfg["activation"], out_layer)
        # init actor network parameters
        for i in range(0, len(actor), 2):
            nn.init.zeros_(actor[i].bias)
            if i < len(actor) - 2:
                nn.init.orthogonal_(actor[i].weight, np.sqrt(2))
            else:
                nn.init.orthogonal_(actor[i].weight, 0.01)
        # make actor a TensorDictModule
        actor = TensorDictModule(
            actor, in_keys=["observation"], out_keys=actor_out_keys
        )
        if self._cont_act:
            act_min = self._env.action_spec.space.minimum
            act_max = self._env.action_spec.space.maximum
            actor = ProbabilisticActor(
                actor,
                in_keys=actor_out_keys,
                out_keys=["action"],
                distribution_class=TanhNormal,
                distribution_kwargs={"min": act_min, "max": act_max},
                return_log_prob=True,
            )
        else:
            actor = ProbabilisticActor(
                actor,
                in_keys=actor_out_keys,
                out_keys=["action"],
                distribution_class=OneHotCategorical,
                return_log_prob=True,
            )

        self._actor = actor.to(device)

    def _init_critic(self):
        cfg = self._cfg

        obs_dim = self._env.observation_spec["observation"].shape[0]
        sizes = [obs_dim] + cfg["hidden_sizes"] + [1]

        critic = mlp(sizes)
        # init critic network parameters
        for i in range(0, len(critic), 2):
            nn.init.zeros_(critic[i].bias)
            if i < len(critic) - 2:
                nn.init.orthogonal_(critic[i].weight, np.sqrt(2))
            else:
                nn.init.orthogonal_(critic[i].weight, 1)
        # make critic a TensorDictModule
        critic = ValueOperator(critic, in_keys=["observation"])

        self._critic = critic

    def _init_opt(self):
        cfg = self._cfg
        self._opt = Adam(
            [
                {"params": self._actor.parameters(), "lr": cfg["lr_actor"]},
                {"params": self._critic.parameters(), "lr": cfg["lr_critic"]},
            ]
        )
        # TODO (ztzhu): use lr_scheduler

    def _init_adv_module(self):
        cfg = self._cfg
        self._adv_module = GAE(
            gamma=cfg["gamma"],
            lmbda=cfg["lam"],
            value_network=self._critic,
            average_gae=True,
        )

    def _init_loss_module(self):
        cfg = self._cfg
        self._loss_module = ClipPPOLoss(
            self._actor,
            self._critic,
            clip_epsilon=cfg["clip_ratio_eps"],
            entropy_bonus=True,
            entropy_coef=cfg["entropy_coef"],
            loss_critic_type=cfg["criterion"],
            normalize_advantage=False,  # we will normalize it at each minibatch
        )

    def _init_collector(self):
        cfg = self._cfg
        self._collector = SyncDataCollector(
            self._env,
            self._actor,
            frames_per_batch=cfg["steps_per_epoch"],
            total_frames=cfg["steps_per_epoch"] * cfg["epochs"],
            max_frames_per_traj=cfg["max_steps_per_traj"],
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
        filename = os.path.join(self._cp_dir, f"epochs_{self._curr_epoch}.pt")
        if os.path.exists(filename):
            print("\x1b[1;33mWarning: Overwriting %s!\x1b[0m" % (filename))

        pathlib.Path(self._cp_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self, filename)

    def _set_dflt_cfg(self):
        self._cfg["device"] = device

        self._cfg.setdefault("seed", 56)

        self._cfg.setdefault("epochs", 100)
        self._cfg.setdefault("save_cp_freq", 2)

        self._cfg.setdefault("hidden_sizes", [256, 256])
        self._cfg.setdefault("activation", nn.Tanh)

        self._cfg.setdefault("gamma", 0.99)
        self._cfg.setdefault("criterion", "l2")
        self._cfg.setdefault("clip_grad_norm", 1.0)

        self._cfg.setdefault("init_stats_param", 1000)
        self._cfg.setdefault("max_steps_per_traj", -1)
        self._cfg.setdefault("env_kwargs", dict())

    def _set_ppo_dflt_cfg(self):
        self._cfg.setdefault("opt_epochs", 20)

        self._cfg.setdefault("lr_actor", 0.0004)
        self._cfg.setdefault("lr_critic", 0.001)

        self._cfg.setdefault("lam", 0.95)
        self._cfg.setdefault("clip_ratio_eps", 0.2)
        self._cfg.setdefault("entropy_coef", 0.01)

        self._cfg.setdefault("batch_size", 64)
        self._cfg.setdefault("steps_per_epoch", 8192)

        num_batch = int(self._cfg["steps_per_epoch"] // self._cfg["batch_size"])
        self._cfg["num_batch"] = num_batch
        steps_per_epoch = num_batch * self._cfg["batch_size"]
        if steps_per_epoch != self._cfg["steps_per_epoch"]:
            print(f"Warning: `steps_per_epoch` is truncated to {steps_per_epoch}!")
            self._cfg["steps_per_epoch"] = steps_per_epoch
