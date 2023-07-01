from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
import torch.nn.functional as F
from torchrl.data.tensor_specs import ContinuousBox, DiscreteBox
from torchrl.modules import (
    OneHotCategorical,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)

from rl_toolbox.utils.backend import get_device
from rl_toolbox.utils.env_utils import make_env
from rl_toolbox.utils.network_utils import mlp

device = get_device()


class PPOAgent:
    def __init__(self, env_name="LunarLander-v2", **cfg) -> None:
        self.initialized = False

        self._env_name = env_name
        self._cfg = cfg

    def learn(self):
        self._init_all()

    def _init_all(self):
        if self.initialized:
            raise RuntimeError("Cannot re-initialize agent!")
        else:
            self.initialized = True

        self._init_dflt_cfg()
        self._init_env()
        self._init_actor()
        # self._init_critic()
        # self._init_opt()
        # self._init_loss()

    def _init_dflt_cfg(self):
        self._set_dflt_cfg()
        self._set_ppo_dflt_cfg()

    def _init_env(self):
        cfg = self._cfg
        self._env = make_env(
            self._env_name,
            cfg["init_stats_param"],
            cfg["seed"],
            device=device,
            **cfg["env_kwargs"]
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

    def _set_dflt_cfg(self):
        self._cfg["device"] = device

        self._cfg.setdefault("seed", 56)

        self._cfg.setdefault("epochs", 600)
        self._cfg.setdefault("save_model_freq", 50)

        self._cfg.setdefault("hidden_sizes", [128, 128])
        self._cfg.setdefault("activation", nn.Tanh)

        self._cfg.setdefault("gamma", 0.99)
        self._cfg.setdefault("criterion", F.mse_loss)
        self._cfg.setdefault("clip_grad_norm", 1.0)

        self._cfg.setdefault("init_stats_param", 1000)
        self._cfg.setdefault("max_steps_per_traj", 1000)
        self._cfg.setdefault("env_kwargs", dict())

    def _set_ppo_dflt_cfg(self):
        self._cfg.setdefault("epochs_actor", 80)
        self._cfg.setdefault("epochs_critic", 80)

        self._cfg.setdefault("lr_actor", 0.0004)
        self._cfg.setdefault("lr_critic", 0.001)

        self._cfg.setdefault("lam", 0.95)
        self._cfg.setdefault("clip_ratio_eps", 0.2)

        self._cfg.setdefault("steps_per_epoch", 5000)

