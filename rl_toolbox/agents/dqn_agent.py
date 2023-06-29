import numpy as np

from torch import nn
from torch.optim import Adam

from rl_toolbox.buffers.replay_buffer import ReplayBuffer
from rl_toolbox.networks.qnet import DuelingQNet, SimpleQNet
from rl_toolbox.utils.backend import get_device
from rl_toolbox.utils.network_utils import as_tensor32, freeze_grad, update_target_net

from .agent import Agent

device = get_device()


class DQNAgent(Agent):
    def __init__(self, model_name, env=None, args_in_name=None, **config) -> None:
        super().__init__(model_name, env, args_in_name, **config)
        self.params_to_save += ["policy_net", "target_net", "opt", "opt_step"]
        self.vis_value_names = ["loss", "ep avg reward"]

    @staticmethod
    def get_model(obs_space_size, action_space_size, config, data=None):
        training = data is None
        if not config["dueling"]:
            policy_net = SimpleQNet(
                obs_space_size,
                action_space_size,
                config["hidden_sizes"],
                config["activation"],
            ).to(device)
            target_net = SimpleQNet(
                obs_space_size,
                action_space_size,
                config["hidden_sizes"],
                config["activation"],
            ).to(device)
        else:
            policy_net = DuelingQNet(
                obs_space_size,
                action_space_size,
                config["hidden_sizes"],
                config["activation"],
            ).to(device)
            target_net = DuelingQNet(
                obs_space_size,
                action_space_size,
                config["hidden_sizes"],
                config["activation"],
            ).to(device)
        if not training:
            policy_net.load_state_dict(data["policy_net"])
            policy_net.eval()
            return policy_net
        return policy_net, target_net

    def one_epoch(self):
        env = self.env
        buf = self.buf
        cfg = self.config
        policy_net = self.policy_net
        target_net = self.target_net
        losses = []
        rewards = []

        obs, _ = env.reset()
        for t in range(cfg["max_steps_per_traj"]):
            obs_tensor = as_tensor32(obs).to(device)
            eps = self.decayed_eps()
            action = policy_net.select_action(obs_tensor, eps)
            new_obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            buf.push(obs, action, reward, new_obs, terminated)

            obs = new_obs

            if len(buf) >= cfg["batch_size"]:
                loss = self.optimize()
                losses.append(loss)
                if self.opt_step % cfg["target_update_freq"] == 0:
                    update_target_net(policy_net, target_net, cfg["tau"])

            if terminated or truncated:
                break

        # log infos
        log = dict()
        log["loss"] = None if len(losses) == 0 else np.mean(losses)
        log["ep avg reward"] = np.mean(rewards)

        return log

    def init_agents(self):
        self.policy_net, self.target_net = __class__.get_model(
            self.obs_space_size, self.action_space_size, self.config
        )
        freeze_grad(self.target_net)

    def init_optimizers(self):
        self.opt = Adam(self.policy_net.parameters(), self.config["lr"])
        self.opt_step = 0

    def init_buf(self):
        cfg = self.config
        self.buf = ReplayBuffer(
            cfg["buf_cap"],
            self.obs_space_size,
            self.action_space_size,
            self.continuous,
        )

    def set_default_extra_config(self):
        cfg = self.config

        cfg.setdefault("epochs_critic", 80)
        cfg.setdefault("batch_size", 128)

        cfg.setdefault("double", False)
        cfg.setdefault("dueling", False)

        cfg.setdefault("lr", 0.0002)

        cfg.setdefault("target_update_freq", 1)
        cfg.setdefault("tau", 0.005)

        cfg.setdefault("eps_start", 0.2)
        cfg.setdefault("eps_end", 0.1)
        cfg.setdefault("eps_decay", 100.0)

        cfg.setdefault("buf_cap", 50000)

    def decayed_eps(self):
        cfg = self.config
        eps_start = cfg["eps_start"]
        eps_end = cfg["eps_end"]
        eps_decay = cfg["eps_decay"]
        return eps_end + (eps_start - eps_end) * np.exp(-self.opt_step / eps_decay)

    def optimize(self):
        cfg = self.config
        batch_size = cfg["batch_size"]
        gamma = cfg["gamma"]
        criterion = cfg["criterion"]
        clip_grad_value = cfg["clip_grad_value"]
        policy_net = self.policy_net
        target_net = self.target_net

        self.opt.zero_grad()

        obses, actions, rewards, next_obses, terminated = self.buf.sample(
            batch_size, device
        )
        qvalues = policy_net(obses).gather(1, actions.view(-1, 1)).squeeze(-1)
        if cfg["double"]:
            next_actions = policy_net(next_obses).argmax(axis=1)
            target_qvalues = (
                target_net(next_obses).gather(1, next_actions.view(-1, 1)).squeeze(-1)
            )
        else:
            target_qvalues = target_net(next_obses).max(axis=1)[0]
        qvalues_expected = rewards + gamma * target_qvalues * (1.0 - terminated)

        loss = criterion(qvalues, qvalues_expected)
        loss.backward()
        if clip_grad_value is not None:
            nn.utils.clip_grad_value_(policy_net.parameters(), clip_grad_value)
        self.opt.step()

        self.opt_step += 1

        return loss.detach().cpu().item()
