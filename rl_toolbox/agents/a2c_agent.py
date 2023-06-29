import numpy as np

from torch import nn
from torch.optim import Adam

from rl_toolbox.buffers.pg_buffer import PGBuffer
from rl_toolbox.networks.pg_net import ActorCritic
from rl_toolbox.utils.backend import get_device
from rl_toolbox.utils.network_utils import as_tensor32, freeze_grad, unfreeze_grad

from .agent import Agent

device = get_device()


class A2CAgent(Agent):
    def __init__(self, model_name, env=None, args_in_name=None, **config) -> None:
        super().__init__(model_name, env, args_in_name, **config)
        self.params_to_save += [
            "ac",
            "opt_actor",
            "opt_critic",
        ]
        self.vis_value_names = ["loss_actor", "loss_critic", "avg_returns"]

    @staticmethod
    def get_model(obs_space_size, action_space_size, config, data=None):
        model = ActorCritic(
            obs_space_size,
            action_space_size,
            config["hidden_sizes"],
            config["continuous"],
            config["activation"],
        ).to(device)
        if data is not None:
            model.load_state_dict(data["ac"])
            model.eval()
        return model

    def one_epoch(self):
        env = self.env
        ac = self.ac
        buf = self.buf
        cfg = self.config
        steps_per_epoch = cfg["steps_per_epoch"]

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

        obses, logps, actions, advs, rets = buf.get_data()

        # update actor
        freeze_grad(ac.critic)
        self.opt_actor.zero_grad()

        logps = ac.actor.log_prob(obses, actions)
        loss_actor = -(logps * advs).mean()

        loss_actor.backward()
        self.opt_actor.step()

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
        log["loss_actor"] = loss_actor.detach().cpu().item()
        log["loss_critic"] = np.mean(losses_critic)
        log["avg_returns"] = self.get_avg_returns()

        return log

    def init_agents(self):
        self.ac = __class__.get_model(
            self.obs_space_size,
            self.action_space_size,
            self.config,
        )

    def init_optimizers(self):
        self.opt_actor = Adam(self.ac.actor.parameters(), self.config["lr_actor"])
        self.opt_critic = Adam(self.ac.critic.parameters(), self.config["lr_critic"])

    def init_buf(self):
        cfg = self.config
        self.buf = PGBuffer(
            cfg["gamma"],
            cfg["lam"],
            cfg["steps_per_epoch"],
            self.obs_space_size,
            self.action_space_size,
            self.continuous,
        )

    def overwrite_default_config(self):
        self.config.setdefault("epochs", 300)
        self.config.setdefault("activation", nn.Tanh)

    def set_default_extra_config(self):
        cfg = self.config

        cfg.setdefault("epochs_critic", 80)

        cfg.setdefault("lr_actor", 0.0004)
        cfg.setdefault("lr_critic", 0.001)

        cfg.setdefault("lam", 0.98)

        cfg.setdefault("steps_per_epoch", cfg["max_steps_per_traj"] * 5)

    def get_avg_returns(self):
        rets = []
        i = 0
        for j in self.buf.traj_lens:
            i += j
            rets.append(self.buf.rets[i - 1])
        return np.mean(rets)
