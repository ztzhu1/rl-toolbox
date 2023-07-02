from abc import ABC, abstractmethod

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import trange

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import torch
from torch import nn
import torch.nn.functional as F

from rl_toolbox.utils.backend import check_notebook, get_device
from rl_toolbox.utils.model_utils import save_checkpoint
from rl_toolbox.visualization.monitor import plot_value

in_notebook = check_notebook()
if in_notebook:
    from IPython import display

device = get_device()


class Agent(ABC):
    def __init__(self, model_dir, env=None, args_in_name=None, **config) -> None:
        self.model_dir = model_dir
        self.args_in_name = args_in_name
        config["env"] = env
        self.config = config
        self.params_to_save = ["log", "continuous", "__class__"]
        self.vis_value_names = None  # should be overwritten

        self.overwrite_default_config()
        self.set_default_config()
        self.set_default_extra_config()
        if model_dir is not None:
            self.check_config_keys()
            print("env: %s" % self.config["env"])

    @staticmethod
    @abstractmethod
    def get_model(obs_space_size, action_space_size, hidden_sizes, config, data=None):
        raise NotImplementedError()

    def run(self):
        self.seed(self.config["seed"])
        self.init_env()
        self.init_agents()
        self.init_optimizers()
        self.init_buf()
        self.init_vis()

        for epoch in trange(1, self.config["epochs"] + 1):
            self.curr_epoch = epoch

            log = self.one_epoch()
            self.process_log(log)
            self.update_vis()

            if self.needs_to_save():
                self.save_checkpoint()

    @abstractmethod
    def one_epoch(self) -> dict:
        raise NotImplementedError()

    def seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def init_env(self):
        config = self.config
        env = config["env"]
        if isinstance(env, str):
            env = gym.make(
                env,
                max_episode_steps=config["max_steps_per_traj"],
                render_mode=None,
                **config["env_config"],
            )
        self.env = env

        self.obs_space_size = env.observation_space.shape[0]

        if isinstance(env.action_space, Box):
            self.continuous = True
            self.action_space_size = env.action_space.shape[0]
        elif isinstance(env.action_space, Discrete):
            self.continuous = False
            self.action_space_size = env.action_space.n
        else:
            raise ValueError()
        self.config["continuous"] = self.continuous

    @abstractmethod
    def init_agents(self):
        raise NotImplementedError()

    @abstractmethod
    def init_optimizers(self):
        raise NotImplementedError()

    @abstractmethod
    def init_buf(self):
        raise NotImplementedError()

    def init_vis(self):
        if self.vis_value_names is None:
            return

        n = len(self.vis_value_names)
        fig = plt.figure(figsize=[5.5 * n, 4])
        self.fig = fig
        self.axes = dict()
        for i, name in enumerate(self.vis_value_names):
            self.axes[name] = fig.add_subplot(1, n, i + 1)
        self.display_handle = None
        if in_notebook:
            self.display_handle = display.display(fig, display_id=True)

    def update_vis(self):
        if self.vis_value_names is None:
            return

        for name in self.vis_value_names:
            plot_value(
                self.axes[name],
                self.curr_epoch,
                self.log[self.log[name].notna()][name].to_numpy(),
                name,
                self.display_handle,
            )

    def process_log(self, log: dict):
        # lazy init
        if not hasattr(self, "log"):
            self.log = pd.DataFrame(columns=log.keys())

        self.log = self.log.append(log, ignore_index=True)

    def save_checkpoint(self):
        names = self.params_to_save
        params_to_save = {name: getattr(self, name) for name in names}
        save_checkpoint(
            self.model_dir,
            self.curr_epoch,
            self.config,
            self.args_in_name,
            **params_to_save,
        )

    def needs_to_save(self):
        save_model_freq = self.config["save_model_freq"]
        return save_model_freq is not None and self.curr_epoch % save_model_freq == 0

    def set_default_env(self):
        config = self.config
        if config["env"] is None:
            config["env"] = "LunarLander-v2"
        env = config["env"]
        if isinstance(env, str):
            if env == "LunarLander-v2":
                config.setdefault("env_config", {"enable_wind": False})
        else:
            config.setdefault("env_config", {})

    def overwrite_default_config(self):
        pass

    def set_default_config(self):
        cfg = self.config

        cfg.setdefault("seed", 56)

        cfg.setdefault("epochs", 600)
        cfg.setdefault("save_model_freq", 50)

        self.set_default_env()

        cfg.setdefault("hidden_sizes", [128, 128])
        cfg.setdefault("activation", nn.ReLU)

        cfg.setdefault("max_steps_per_traj", 1000)

        cfg.setdefault("gamma", 0.99)
        cfg.setdefault("criterion", F.mse_loss)
        cfg.setdefault("clip_grad_value", None)

    def set_default_extra_config(self):
        pass

    def check_config_keys(self):
        subclass = getattr(self, "__class__")
        valid_keys = list(subclass(None).config.keys())
        for key in self.config.keys():
            if key not in valid_keys:
                raise KeyError(f"`{key}` is an invalid key!")
