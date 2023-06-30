from collections import OrderedDict
from typing import Union

import numpy as np

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import Compose, DoubleToFloat, ObservationNorm, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator

from rl_toolbox.utils.backend import get_device
from rl_toolbox.utils.network_utils import mlp

device = get_device()


def make_env(
    env_name,
    seed=0,
    init_stats_param: Union[int, OrderedDict] = 1000,
    check_env=False,
    **env_cfg
):
    env = GymEnv(env_name, **env_cfg)
    env = TransformedEnv(
        env,
        Compose(
            ObservationNorm(in_keys=["observation"], standard_normal=True),
            DoubleToFloat(in_keys=["observation"]),
        ),
    )
    env.set_seed(seed)
    init_env(env, init_stats_param)

    if check_env:
        check_env_specs(env)

    return env


def init_env(env, init_stats_param=1000):
    t = None
    for _t in env.transform:
        if isinstance(_t, ObservationNorm):
            t = _t
            break
    if t is None:
        return

    if isinstance(init_stats_param, int):
        if not t.initialized:
            t.init_stats(num_iter=init_stats_param)
    else:
        assert isinstance(init_stats_param, OrderedDict)
        t.load_state_dict(init_stats_param)


def make_actor(hidden_sizes, env):
    sizes = (
        [env.observation_spec["observation"].shape[0]]
        + hidden_sizes
        + [env.action_spec.shape[0] * 2]  # loc and scale
    )
    actor = mlp(sizes, nn.Tanh, nn.Identity)
    actor.append(NormalParamExtractor())
    actor = TensorDictModule(actor, ["observation"], ["loc", "scale"])
    actor = ProbabilisticActor(
        actor,
        ["loc", "scale"],
        ["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.action_spec.space.minimum,
            "max": env.action_spec.space.maximum,
        },
        return_log_prob=True,
    )
    return actor


def make_critic(hidden_sizes, env):
    sizes = [env.observation_spec["observation"].shape[0]] + hidden_sizes + [1]
    critic = mlp(sizes, nn.Tanh, nn.Identity)
    critic = ValueOperator(module=critic, in_keys=["observation"])
    return critic


def make_collector(env, actor, epochs, steps_per_epoch, max_steps_per_traj):
    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=steps_per_epoch,
        total_frames=steps_per_epoch * epochs,
        max_frames_per_traj=max_steps_per_traj,
        reset_at_each_iter=True,
        split_trajs=False,
    )
    return collector


def make_buf(steps_per_epoch):
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(steps_per_epoch),
        sampler=SamplerWithoutReplacement(),
    )
    return replay_buffer
