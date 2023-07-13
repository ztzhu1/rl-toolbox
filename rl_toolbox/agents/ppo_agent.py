import numpy as np

from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
import torch
from torch import nn
from torch.distributions.kl import kl_divergence
from torch.optim import Adam
from torchrl.collectors import SyncDataCollector
from torchrl.data.tensor_specs import ContinuousBox, DiscreteBox
from torchrl.modules import (
    LSTMModule,
    OneHotCategorical,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from rl_toolbox.agents.agent import Agent
from rl_toolbox.utils.backend import check_notebook, get_device
from rl_toolbox.utils.env_utils import make_env
from rl_toolbox.utils.network_utils import mlp

in_notebook = check_notebook()
if in_notebook:
    from IPython import display

device = get_device()


class PPOAgent(Agent):
    """
    refer to
    - https://github.com/openai/baselines/blob/master/baselines/ppo2/ppo2.py
    - https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html
    - https://pytorch.org/rl/tutorials/coding_ppo.html
    - https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    """

    def __init__(self, env_name="LunarLander-v2", cp_dir=None, **cfg) -> None:
        vis_value_names = [
            "loss",
            "episode_reward",
            "kl_div",
            "avg_frame",
        ]
        saved_attr_names = [
            "env_name",
            "curr_epoch",
            "vis_value_names",
            "env",
            "actor",
            "opt",
            "adv_module",
            "loss_module",
        ]
        super().__init__(
            env_name, cp_dir, False, vis_value_names, saved_attr_names, **cfg
        )

    def one_epoch(self):
        data = self._collector.next()
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
            data["next"]["episode_reward"]
            .masked_select(data["next"]["done"])
            .mean()
            .detach()
            .cpu()
            .item()
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
        lstm_hs = cfg["lstm_hidden_size"]

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
        sizes = [lstm_hs or obs_dim] + cfg["hidden_sizes"] + [out_dim_factor * act_dim]

        actor = mlp(sizes, getattr(nn, cfg["activation"]), out_layer)
        # init actor network parameters
        for i in range(0, len(actor), 2):
            nn.init.zeros_(actor[i].bias)
            if i < len(actor) - 2:
                nn.init.orthogonal_(actor[i].weight, np.sqrt(2))
            else:
                nn.init.orthogonal_(actor[i].weight, 0.01)
        # make actor a TensorDictModule
        if lstm_hs is None:
            actor = TensorDictModule(
                actor, in_keys=["observation"], out_keys=actor_out_keys
            )
        else:
            lstm_module = LSTMModule(
                obs_dim,
                lstm_hs,
                in_keys=["observation", "rs_h", "rs_c"],
                out_keys=["intermediate", ("next", "rs_h"), ("next", "rs_c")],
            )
            actor = TensorDictModule(
                actor, in_keys=["intermediate"], out_keys=actor_out_keys
            )
            actor = TensorDictSequential(lstm_module, actor)
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

        critic = mlp(sizes, getattr(nn, cfg["activation"]), nn.Identity)
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
        # TODO (ztzhu): use parallel collector
        cfg = self._cfg
        self._collector = SyncDataCollector(
            self._env,
            self._actor,
            frames_per_batch=cfg["steps_per_epoch"],
            total_frames=cfg["steps_per_epoch"] * cfg["epochs"],
            max_frames_per_traj=cfg["max_steps_per_traj"],
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
            "activation": "Tanh",
            "gamma": 0.99,
            "criterion": "l2",
            "clip_grad_norm": 1.0,
            # --- env ---
            "init_stats_param": 1000,
            "max_steps_per_traj": -1,
            "steps_per_epoch": 8192,
            "env_kwargs": {},
            # --- PPO specific ---
            "lstm_hidden_size": None,
            "opt_epochs": 20,
            "batch_size": 128,
            "lr_actor": 0.0004,
            "lr_critic": 0.001,
            "lam": 0.95,
            "clip_ratio_eps": 0.2,
            "entropy_coef": 0.01,
        }
        cfg["num_batch"] = int(cfg["steps_per_epoch"] // cfg["batch_size"])
        return cfg
