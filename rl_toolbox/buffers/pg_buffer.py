import numpy as np

from rl_toolbox.utils.network_utils import as_tensor32, discount_cumsum


class PGBuffer:
    def __init__(
        self, gamma, lam, steps_per_epoch, obs_space_size, action_space_size, continuous
    ) -> None:
        self.gamma = gamma
        self.lam = lam
        self.steps_per_epoch = steps_per_epoch
        self.obs_space_size = obs_space_size
        self.action_space_size = action_space_size
        self.buf_size = steps_per_epoch
        self.start = 0
        self.next = 0

        self.obses = np.zeros((self.buf_size, obs_space_size), dtype=np.float32)
        self.values = np.zeros(self.buf_size, dtype=np.float32)
        if continuous:
            self.actions = np.zeros(
                (self.buf_size, action_space_size), dtype=np.float32
            )
        else:
            self.actions = np.zeros(self.buf_size, dtype=np.float32)
        self.logps = np.zeros(self.buf_size, dtype=np.float32)
        self.rewards = np.zeros(self.buf_size, dtype=np.float32)
        self.advs = np.zeros(self.buf_size, dtype=np.float32)
        self.rets = np.zeros(self.buf_size, dtype=np.float32)
        self.traj_lens = []

    def reset(self):
        self.start = self.next = 0
        self.traj_lens = []

    def store(self, obs, value, action, logp, reward):
        i = self.next
        self.obses[i] = obs
        self.values[i] = value
        self.actions[i] = action
        self.logps[i] = logp
        self.rewards[i] = reward
        self.next += 1

    def done(self, last_value=0.0):
        traj_slice = slice(self.start, self.next)
        traj_values = np.append(self.values[traj_slice], last_value)
        traj_rewards = np.append(self.rewards[traj_slice], last_value)
        traj_deltas = (
            traj_rewards[:-1] + self.gamma * traj_values[1:] - traj_values[:-1]
        )
        traj_advs = discount_cumsum(traj_deltas, self.gamma * self.lam)
        traj_rets = discount_cumsum(traj_rewards, self.gamma)[:-1]

        self.advs[traj_slice] = traj_advs
        self.rets[traj_slice] = traj_rets

        self.start = self.next
        self.traj_lens.append(len(traj_advs))

    def get_data(self, device=None):
        i = self.next
        data_names = ["obses", "logps", "actions", "advs", "rets"]
        data = []
        for name in data_names:
            data.append(as_tensor32(getattr(self, name)[:i], device))
        return data
