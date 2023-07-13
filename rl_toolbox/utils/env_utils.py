from collections import OrderedDict
from typing import Optional, Union

from torchrl.envs import (
    Compose,
    ObservationNorm,
    RewardSum,
    StepCounter,
    InitTracker,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs


def make_env(
    env_name="LunarLander-v2",
    init_stats_param: Optional[Union[int, OrderedDict]] = None,
    seed=0,
    device=None,
    **env_cfg
):
    env = GymEnv(env_name, frame_skip=1, **env_cfg)
    if init_stats_param is not None:
        compose = Compose(
            ObservationNorm(in_keys=["observation"], standard_normal=True),
            RewardSum(),
            StepCounter(),
            InitTracker(),
        )
    else:
        compose = Compose(
            RewardSum(),
            StepCounter(),
            InitTracker(),
        )
    env = TransformedEnv(
        env,
        compose,
        device=device,
    )

    env.set_seed(seed)

    if init_stats_param is not None:
        init_env(env, init_stats_param)
        check_env_specs(env)

    return env


def init_env(env, init_stats_param: Optional[Union[int, OrderedDict]] = 1000):
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
        t.register_buffer(
            "standard_normal", init_stats_param["transforms.0.standard_normal"]
        )
        t.register_buffer("loc", init_stats_param["transforms.0.loc"])
        t.register_buffer("scale", init_stats_param["transforms.0.scale"])
