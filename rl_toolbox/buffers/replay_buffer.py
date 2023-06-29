import numpy as np
from numpy_ringbuffer import RingBuffer

import torch

from rl_toolbox.utils.network_utils import as_tensor32


class ReplayBuffer:
    def __init__(
        self, cap, obs_space_size, action_space_size, continuous=False
    ) -> None:
        self.obs_space_size = obs_space_size
        self.action_space_size = action_space_size
        self.cap = cap
        self.continuous = continuous

        self.obses = RingBuffer(cap, (np.float32, obs_space_size), allow_overwrite=True)
        if not continuous:
            self.actions = RingBuffer(cap, np.int64, allow_overwrite=True)
        else:
            self.actions = RingBuffer(
                cap, (np.float32, action_space_size), allow_overwrite=True
            )
        self.rewards = RingBuffer(cap, np.float32, allow_overwrite=True)
        self.next_obses = RingBuffer(
            cap, (np.float32, obs_space_size), allow_overwrite=True
        )
        self.terminated = RingBuffer(cap, np.float32, allow_overwrite=True)

    def __len__(self):
        return len(self.rewards)

    def push(self, obs, action, reward, next_obs, terminated):
        self.obses.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_obses.append(next_obs)
        self.terminated.append(terminated)

    def sample(self, n, device=None):
        curr_size = len(self)
        if n > curr_size:
            raise ValueError()
        indexes = np.random.choice(
            np.arange(curr_size, dtype=np.int32), n, replace=False
        )
        names = ["obses", "actions", "rewards", "next_obses", "terminated"]
        datas = (getattr(self, i)[indexes] for i in names)
        if device is not None:
            datas_tensor = tuple()
            for i, data in enumerate(datas):
                if names[i] != "actions" or self.continuous:
                    data = as_tensor32(data, device)
                else:
                    data = torch.as_tensor(data, dtype=torch.int64, device=device)
                datas_tensor += (data,)
            datas = datas_tensor
        return datas
