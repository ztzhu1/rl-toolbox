from collections import OrderedDict
from typing import Optional, Union

from torchrl.envs import Compose, DoubleToFloat, ObservationNorm, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs


def make_env(
    env_name="LunarLander-v2",
    init_stats_param: Optional[Union[int, OrderedDict]] = None,
    seed=0,
    device=None,
    **env_cfg
):
    env = GymEnv(env_name, **env_cfg)
    env = TransformedEnv(
        env,
        Compose(
            ObservationNorm(in_keys=["observation"], standard_normal=True),
            DoubleToFloat(in_keys=["observation"]),
        ),
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
        t.load_state_dict(init_stats_param)