import numpy as np

import torch

from rl_toolbox.utils.backend import get_device
from rl_toolbox.utils.network_utils import as_tensor32, freeze_grad, unfreeze_grad

from .a2c_agent import A2CAgent

device = get_device()


class PPOAgent(A2CAgent):
    def one_epoch(self):
        env = self.env
        ac = self.ac
        buf = self.buf
        cfg = self.config
        steps_per_epoch = cfg["steps_per_epoch"]
        epsilon = cfg["epsilon"]

        obs, _ = env.reset()
        buf.reset()
        for t in range(steps_per_epoch):
            obs_tensor = as_tensor32(obs).to(device)
            action, logp = ac.select_action(obs_tensor)
            value = ac.est_value(obs_tensor)
            new_obs, reward, terminated, truncated, _ = env.step(action)
            buf.store(obs, value, action, logp, reward)

            obs = new_obs

            if t == steps_per_epoch - 1:
                truncated = True
            done = terminated or truncated
            if done:
                if terminated:
                    value = 0.0
                else:
                    value = ac.est_value(as_tensor32(obs).to(device))

                buf.done(value)
                obs, _ = env.reset()

        obses, logps_old, actions, advs, rets = buf.get_data()

        # update actor
        freeze_grad(ac.critic)
        losses_actor = []
        for _ in range(cfg["epochs_actor"]):
            self.opt_actor.zero_grad()

            logps = ac.actor.log_prob(obses, actions)
            ratio = torch.exp(logps - logps_old)
            unclipped = ratio * advs
            clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advs
            loss_actor = -torch.min(unclipped, clipped).mean()

            loss_actor.backward()
            self.opt_actor.step()

            losses_actor.append(loss_actor.detach().cpu().item())
        unfreeze_grad(ac.critic)

        # update critic
        freeze_grad(ac.actor)
        losses_critic = []
        for _ in range(cfg["epochs_critic"]):
            self.opt_critic.zero_grad()

            values = ac.critic(obses)
            loss_critic = cfg["criterion"](values, rets)

            loss_critic.backward()
            self.opt_critic.step()

            losses_critic.append(loss_critic.detach().cpu().item())
        unfreeze_grad(ac.actor)

        # log infos
        log = dict()
        log["loss_actor"] = np.mean(losses_actor)
        log["loss_critic"] = np.mean(losses_critic)
        log["avg_returns"] = self.get_avg_returns()

        return log

    def set_default_extra_config(self):
        cfg = self.config

        cfg.setdefault("epochs_actor", 80)
        cfg.setdefault("epochs_critic", 80)

        cfg.setdefault("lr_actor", 0.0004)
        cfg.setdefault("lr_critic", 0.001)

        cfg.setdefault("lam", 0.95)
        cfg.setdefault("epsilon", 0.2)

        cfg.setdefault("steps_per_epoch", cfg["max_steps_per_traj"] * 5)
